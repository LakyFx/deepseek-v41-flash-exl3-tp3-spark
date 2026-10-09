"""Bind engine deltas to one closed-window workload, not arbitrary traffic.

Transport and generation/lease proof are injected by the fleet adapter. No
inference or remote service action is reachable by constructing this monitor.
"""
from __future__ import annotations

import time

from .metrics import GAUGES, compare, parse


class MeasurementOwnershipLost(RuntimeError):
    pass


class NativeMonitor:
    def __init__(self, fetch, guard, *, save, armed=False, clock=time.monotonic,
                 pause=time.sleep, settle_seconds=3):
        self.fetch, self.guard, self.save = fetch, guard, save
        self.armed, self.clock, self.pause = armed, clock, pause
        if not 0 < settle_seconds <= 10:
            raise ValueError("bounded metric settling required")
        self.settle_seconds = settle_seconds

    def snapshot(self):
        if self.armed is not True:
            raise RuntimeError("native monitor is unarmed")
        try:
            proof = self.guard()
        except Exception as exc:
            raise MeasurementOwnershipLost("closed-window guard failed") from exc
        if not isinstance(proof, dict) or proof.get("exclusive") is not True or not proof.get("campaign_id"):
            raise MeasurementOwnershipLost("closed ingress and campaign ownership proof missing")
        value = self.fetch()
        if not value.get("engine_generation") or not isinstance(value.get("metrics"), str):
            raise ValueError("bound engine generation/metrics required")
        return {"generation": value["engine_generation"], "campaign_id": proof["campaign_id"],
                "values": parse(value["metrics"]), "monotonic": self.clock()}

    def __call__(self, identity, expected_requests, deadline, run):
        before = self.snapshot()
        if any(before["values"][key] != 0 for key in GAUGES):
            raise ValueError("native measurement starts with unfinished requests")
        if self.clock() >= deadline:
            raise TimeoutError("native measurement deadline before requests")
        result = None
        error = None
        try:
            result = run()
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record = {"schema": "dgx.closed-workload-native-metrics.v1", "identity": identity,
                      "before": before, "workload_error": error, "metrics": None}
            fatal = None
            try:
                count = expected_requests(result) if callable(expected_requests) and result is not None else expected_requests
                if type(count) is not int or count < 1:
                    raise ValueError("actual positive request-attempt count required")
                settle_end = min(deadline, self.clock()+self.settle_seconds)
                while True:
                    after = self.snapshot()
                    if after["campaign_id"] != before["campaign_id"]:
                        raise MeasurementOwnershipLost("measurement ownership changed")
                    if after["generation"] != before["generation"]:
                        raise MeasurementOwnershipLost("engine generation changed during measurement")
                    values = compare(before["values"], after["values"], expected_requests=count,
                        same_generation=before["generation"] == after["generation"], exclusive=True)
                    if values["coverage_verified"] or self.clock() >= settle_end:
                        break
                    # More completed requests is overlap, not a delayed scrape.
                    if values["observed_requests"] is not None and values["observed_requests"] > count:
                        break
                    if after["values"].get("request_generation_tokens_count") is None:
                        break
                    self.pause(min(0.05, max(0, settle_end-self.clock())))
                record.update(after=after, metrics=values)
                if result is not None:
                    result["native_metrics"] = values
            except Exception as exc:
                if isinstance(exc,MeasurementOwnershipLost):
                    fatal=exc
                record["measurement_error"] = type(exc).__name__+":"+str(exc)[:300]
                if result is not None:
                    result["native_metrics"] = {"coverage_verified": False,
                        "missing_or_overlapping_metrics": "UNKNOWN_NOT_ZERO_GAIN",
                        "measurement_error": record["measurement_error"]}
            # Persist metrics independently, also if the workload raised and its
            # per-request partial rows are all that survived. Disk failure is
            # fatal: proceeding without durable evidence is not a useful test.
            self.save(record)
            if fatal is not None:
                raise fatal


def workflow_attempts(result):
    return sum(row["request_attempts"] for row in result["tasks"])
