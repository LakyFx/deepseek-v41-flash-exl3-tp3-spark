"""Armed-only cold/hot, long, C8 and event-injected mixed request drivers.

Measured bodies come from the frozen corpus. These drivers never manage
services or clear caches; a successful controller cache-reset receipt is a
separate requirement for calling any prime cold.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import threading
import time


def require_armed(client):
    if getattr(client, "armed", False) is not True:
        raise RuntimeError("load driver requires explicitly armed client")


def body_for(entry, *, campaign, group, slot, output_tokens):
    if entry.get("measured_prompt_tokens", 0) <= 0 or not entry.get("tokenization_receipt_sha256"):
        raise ValueError("measured frozen corpus entry required")
    body = copy.deepcopy(entry["body"])
    if body.get("reasoning_effort") != "max":
        raise ValueError("performance corpus must retain main max reasoning")
    # Version deliberately excluded. Input body is identical for prime/hot.
    body["cache_salt"] = hashlib.sha256(f"{campaign}:{group}:{slot}".encode()).hexdigest()
    body["max_tokens"] = output_tokens
    return body


def request_row(client, entry, body, *, phase, slot, deadline, on_first_delta=None):
    started = time.monotonic()
    row = {"case_id": entry["id"], "phase": phase, "slot": slot,
           "started_monotonic": started, "input_tokens_expected": entry["measured_prompt_tokens"],
           "request_body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
           "complete": False, "usage": None, "error": None}
    try:
        call = client.complete(body, deadline=deadline, on_first_delta=on_first_delta)
        reply = call["reply"]
        counts = reply.usage_counts()
        if counts["prompt_tokens"] != entry["measured_prompt_tokens"]:
            raise ValueError("runtime tokenization differs from frozen corpus")
        first = reply.first_any_delta
        row.update(complete=True, usage=counts, request_id=call["request_id"],
                   first_any_delta_monotonic=first, first_visible_delta_monotonic=reply.first_visible_delta,
                   sse_chunk_times=list(reply.delta_times), finish_reason=reply.finish_reason,
                   assistant=reply.message(), ttft_seconds=first-started if first is not None else None)
    except (ValueError, KeyError, TypeError, TimeoutError, OSError) as exc:
        row["error"] = type(exc).__name__ + ":" + str(exc)[:300]
    row["ended_monotonic"] = time.monotonic()
    row["wall_seconds"] = row["ended_monotonic"] - started
    # TTFT includes queue/sampling/network. Do not label prompt/TTFT as native
    # prefill throughput or completion/(end-firstdelta) as exact GPU decode.
    row["request_output_tokens_per_wall_second"] = (
        row["usage"]["completion_tokens"] / row["wall_seconds"] if row["complete"] else None)
    return row


def dispatch_cohort(client, jobs, *, deadline, save_completed=None):
    require_armed(client)
    if not 1 <= len(jobs) <= 8 or time.monotonic() >= deadline:
        raise ValueError("bounded nonexpired cohort required")
    started = time.monotonic()
    barrier = threading.Barrier(len(jobs))
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futures = []
    results = {}

    def execute(job):
        barrier.wait(timeout=max(0.001, deadline-time.monotonic()))
        return request_row(client, **job, deadline=deadline)

    def collect(done):
        for future in done:
            row = future.result()
            results[future] = row
            if save_completed:
                save_completed(row)
    try:
        futures = [pool.submit(execute, job) for job in jobs]
        pending = set(futures)
        while pending and time.monotonic() < deadline:
            done, pending = wait(pending, timeout=max(0, deadline-time.monotonic()), return_when=FIRST_COMPLETED)
            collect(done)
        if pending:
            client.abort_all_owned()
            done, pending = wait(pending, timeout=5)
            collect(done)
        if pending:
            raise TimeoutError("cohort threads remain after owned abort")
        rows = [results[f] for f in futures]
        wall = time.monotonic()-started
        complete = all(row["complete"] for row in rows)
        return {"rows": rows, "wall_seconds": wall, "usage_complete": complete,
                "aggregate_output_tokens_per_wall_second": (
                    sum(row["usage"]["completion_tokens"] for row in rows)/wall if complete else None),
                "concurrency_scope": "HTTP request overlap; GPU concurrency requires engine trace"}
    except BaseException:
        barrier.abort()
        client.abort_all_owned()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_cold_hot(client, entries, *, campaign, group, deadline, output_tokens=128,
                 cold_reset_receipt=None, save_completed=None, measure=None):
    require_armed(client)
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("independent distinct prefixes required")
    jobs = [{"entry": entry, "body": body_for(entry, campaign=campaign, group=group, slot=i, output_tokens=1),
             "slot": i, "phase": "prime"} for i, entry in enumerate(entries)]
    def execute(phase):
        run = lambda: dispatch_cohort(client,jobs,deadline=deadline,save_completed=save_completed)
        return measure((group,phase),len(jobs),deadline,run) if measure else run()
    prime = execute("prime")
    result = {"regime": group, "prime": prime, "hot": None,
              "cold_reset_receipt": cold_reset_receipt,
              "cold_status": "RESET_PROVEN" if cold_reset_receipt else "COLD_NOT_PROVEN",
              "prime_wall_separate_from_hot": True}
    if not prime["usage_complete"]:
        result["error"] = "prime incomplete; hot skipped"
        return result
    prime_counts=[row["usage"] for row in prime["rows"]]
    result["cold_cache_verified"] = bool(cold_reset_receipt) and all(count["cached_tokens"]==0 for count in prime_counts)
    for job in jobs:
        job["phase"] = "hot"
        job["body"]["max_tokens"] = output_tokens
    result["hot"] = execute("hot")
    result["hot_cache_verified"] = result["hot"]["usage_complete"] and all(
        row["usage"]["cached_tokens"]>=row["usage"]["prompt_tokens"]*0.9 for row in result["hot"]["rows"])
    return result


def run_mixed(client, leaders, injected, *, campaign, deadline, leader_output=1024,
              injected_output=128, cold_reset_receipt=None, save_completed=None):
    require_armed(client)
    if len(leaders) != 3 or len({row["id"] for row in leaders} | {injected["id"]}) != 4:
        raise ValueError("three warm leaders and one independent injected prefix required")
    prime_jobs = [{"entry": entry, "body": body_for(entry, campaign=campaign, group="mixed", slot=i,
                  output_tokens=1), "phase": "prime", "slot": i} for i, entry in enumerate(leaders)]
    warm = dispatch_cohort(client, prime_jobs, deadline=deadline, save_completed=save_completed)
    if not warm["usage_complete"]:
        return {"regime": "mixed", "warmup": warm, "error": "warmup incomplete", "injected": None}
    event = threading.Event()
    trigger_lock = threading.Lock()
    trigger = {}
    def first_delta(when, response_id):
        with trigger_lock:
            if not trigger:
                trigger.update(monotonic=when, response_id=response_id)
                event.set()
    jobs = [{"entry": entry, "body": body_for(entry, campaign=campaign, group="mixed", slot=i,
             output_tokens=leader_output), "phase": "mixed-leader", "slot": i, "on_first_delta": first_delta}
            for i, entry in enumerate(leaders)]
    pool = ThreadPoolExecutor(max_workers=1)
    started = time.monotonic()
    future = pool.submit(dispatch_cohort, client, jobs, deadline=deadline, save_completed=save_completed)
    injected_row = None
    try:
        while not event.wait(timeout=min(0.1, max(0, deadline-time.monotonic()))):
            if future.done() or time.monotonic() >= deadline:
                break
        if event.is_set() and not future.done() and time.monotonic() < deadline:
            injected_row = request_row(client, injected,
                body_for(injected, campaign=campaign, group="mixed-injected", slot=0, output_tokens=injected_output),
                phase="mixed-injected", slot=3, deadline=deadline)
            if save_completed:
                save_completed(injected_row)
        leader_rows = future.result(timeout=max(0.001, deadline-time.monotonic())+5)
        active_at_injection = sum(row["started_monotonic"] <= injected_row["started_monotonic"] < row["ended_monotonic"]
                                  for row in leader_rows["rows"]) if injected_row else 0
        warm_verified=leader_rows["usage_complete"] and all(
            row["usage"]["cached_tokens"]>=row["usage"]["prompt_tokens"]*0.9 for row in leader_rows["rows"])
        injected_cold=bool(injected_row and injected_row["complete"] and injected_row["usage"]["cached_tokens"]==0)
        return {"regime": "mixed", "warmup": warm, "trigger": trigger or None,
                "leaders": leader_rows, "injected": injected_row,
                "active_http_leaders_at_injection": active_at_injection,
                "warm_leader_cache_verified":warm_verified,
                "injected_cold_cache_verified":injected_cold,
                "cache_regime_verified":bool(warm_verified and injected_cold and active_at_injection),
                "cold_reset_receipt": cold_reset_receipt,
                "wall_seconds": time.monotonic()-started,
                "injection_status": "FIRST_ACTUAL_SSE_DELTA" if active_at_injection else "NO_OVERLAP_NOT_A_SPEEDUP",
                "gap_scope": "SSE chunk gaps, not per-token ITL"}
    except BaseException:
        client.abort_all_owned()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
