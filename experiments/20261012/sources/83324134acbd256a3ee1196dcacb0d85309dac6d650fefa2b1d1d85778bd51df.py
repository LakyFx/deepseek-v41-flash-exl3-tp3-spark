"""Opt-in scheduler chunk telemetry. No tensor access, GPU sync or prompt logging."""
import hashlib
import collections
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

SCHEMA = "dgx.chunk-prefill.v1"


def describe(scheduler, output, credited):
    """Use pre-execution positions from SchedulerOutput, not advanced requests."""
    for rid in getattr(output, "finished_req_ids", ()):
        credited.discard(rid)
    starts = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    starts.update(zip(cached.req_ids, cached.num_computed_tokens))
    computed = hits = prefill_requests = 0
    cache_known = True
    for rid, scheduled in output.num_scheduled_tokens.items():
        req = scheduler.requests.get(rid)
        if req is None or rid not in starts:
            raise ValueError("Missing scheduled request/position")
        prompt = req.num_prompt_tokens
        new = min(scheduled, max(0, prompt - starts[rid]))
        if new <= 0:
            continue
        computed += new
        prefill_requests += 1
        if rid not in credited:
            credited.add(rid)
            stats = getattr(req, "prefill_stats", None)
            if getattr(req, "num_preemptions", 0) > 0:
                pass  # Recompute is real work, not another prefix-cache hit.
            elif stats is None:
                cache_known = False
            else:
                hits += min(prompt, max(0, stats.num_cached_tokens))
    return {"computed_tokens": computed, "cached_tokens": hits if cache_known else None,
            "prefill_requests": prefill_requests,
            "other_scheduled_tokens": max(0, output.total_num_scheduled_tokens-computed)}


class ChunkMeter:
    def __init__(self, path, model, clock=time.monotonic, wall=time.time):
        self.path, self.model, self.clock, self.wall = Path(path), model, clock, wall
        self.credited, self.pending = set(), {}
        self.events = queue.Queue(maxsize=128)
        self.seq = 0
        self.dropped = 0
        self.thread = None

    def begin(self, scheduler, output):
        description = describe(scheduler, output, self.credited)
        # Scalar-only request shape metadata, no prompt text or token IDs.
        try:
            from diaglog import emit
            cached=output.scheduled_cached_reqs
            positions={r.req_id:r.num_computed_tokens for r in output.scheduled_new_reqs}
            positions.update(zip(cached.req_ids,cached.num_computed_tokens))
            should_log=bool(description["computed_tokens"]) or self.clock()-getattr(self,"_diag_last",0)>=1
            if should_log:self._diag_last=self.clock()
            if should_log: emit("scheduler_batch", total_scheduled=output.total_num_scheduled_tokens,
                 requests=[dict(id=hashlib.sha256(rid.encode()).hexdigest()[:16],
                    scheduled=n,computed_before=positions.get(rid),
                    prompt_tokens=scheduler.requests[rid].num_prompt_tokens,
                    preemptions=getattr(scheduler.requests[rid],"num_preemptions",None))
                    for rid,n in output.num_scheduled_tokens.items() if rid in scheduler.requests])
        except Exception:pass
        if not description["computed_tokens"]:
            return
        self.seq += 1
        event = dict(description, seq=self.seq, started_at=self.wall(),
                     start_monotonic=self.clock())
        self.pending[id(output)] = event
        self.emit(("start", event.copy()))

    def complete(self, output):
        event = self.pending.pop(id(output), None)
        if event is None:
            return
        duration = self.clock()-event.pop("start_monotonic")
        if duration <= 0:
            return
        event.update(completed_at=self.wall(), elapsed_seconds=duration,
                     compute_tps=event["computed_tokens"]/duration)
        hits = event["cached_tokens"]
        event["including_cache_tps"] = (
            (event["computed_tokens"]+hits)/duration if hits is not None else None)
        self.emit(("complete", event))

    def emit(self, event):
        try:
            from diaglog import emit
            emit("scheduler_chunk", event_type=event[0], chunk=event[1].copy())
        except Exception:
            pass
        if self.thread is None:
            self.thread = threading.Thread(target=self.writer, daemon=True,
                                           name="chunk-telemetry")
            self.thread.start()
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def writer(self):
        active, last, recent = {}, None, collections.deque(maxlen=64)
        write_errors = 0
        while True:
            try:
                kind, event = self.events.get(timeout=1)
                if kind == "start":
                    event.pop("start_monotonic", None)
                    active[event["seq"]] = event
                else:
                    active.pop(event["seq"], None)
                    last = event
                    recent.append(event)
                if self.events.qsize():
                    continue
            except queue.Empty:
                pass
            now = self.wall()
            # Bounded observability only; no request IDs/content are persisted.
            active = {k:v for k,v in active.items() if now-v["started_at"] < 600}
            payload = {"schema": SCHEMA, "model": self.model, "pid": os.getpid(),
                       "published_at": now, "measurement": "dispatch_to_completion",
                       "includes_pipeline_wait": True, "dropped_events": self.dropped,
                       "active_chunks": list(active.values()), "last_chunk": last,
                       "recent_chunks": list(recent)}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError as exc:
                write_errors += 1
                if write_errors == 1 or write_errors % 60 == 0:
                    logging.getLogger(__name__).warning("Chunk telemetry write failed: %s", exc)


_meters = {}


def begin(core, output):
    path = os.environ.get("DGX_CHUNK_TELEMETRY_PATH")
    if not path:
        return
    try:
        meter = _meters.get(id(core))
        if meter is None:
            config = core.vllm_config.model_config
            model = getattr(config, "served_model_name", None) or config.model
            meter = _meters[id(core)] = ChunkMeter(path, model)
        meter.begin(core.scheduler, output)
    except Exception:
        # Instrumentation must not fail inference; unsupported is not zero.
        pass


def complete(core, output):
    try:
        meter = _meters.get(id(core))
        if meter is not None:
            meter.complete(output)
    except Exception:
        pass
