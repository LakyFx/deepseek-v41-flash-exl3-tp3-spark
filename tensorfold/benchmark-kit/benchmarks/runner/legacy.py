"""Original sixteen cold/hot C1..C4 cells or frozen six-cell screen.

Same eight private inputs, original salts, max effort, temperature0/seed42,
256-output cap and one-token exact hot priming. Source-set staging and cold
reset evidence are supplied by the future fleet controller.
"""
from __future__ import annotations

import copy
import hashlib
import time

from .load_regimes import dispatch_cohort, require_armed

FULL = tuple((phase, category, concurrency) for phase in ("cold", "hot")
             for concurrency in range(1, 5) for category in ("code", "prose"))
SCREEN = (("cold", "code", 3), ("cold", "prose", 3),
          ("hot", "code", 1), ("hot", "prose", 1),
          ("hot", "code", 3), ("hot", "prose", 3))


def jobs_for(entries, *, campaign, category, concurrency, phase, output_tokens):
    jobs = []
    for slot, entry in enumerate(entries[:concurrency]):
        body = copy.deepcopy(entry["body"])
        # Retain exact engine-input template controls, including original tools.
        body.update(temperature=0, seed=42, max_tokens=output_tokens)
        body.pop("max_completion_tokens", None)
        body["cache_salt"] = hashlib.sha256(f"{campaign}:{category}:{concurrency}:{slot}".encode()).hexdigest()
        jobs.append({"entry": entry, "body": body, "phase": phase, "slot": slot})
    return jobs


def run_legacy(client, groups, *, campaign, deadline, full=False, cold_reset_receipt=None,
               save_completed=None, measure=None):
    require_armed(client)
    if not cold_reset_receipt:
        raise ValueError("legacy cold cells require controller-proven cache reset")
    started = time.monotonic()
    result = {"schema": "dgx.expanded-campaign-legacy.v1", "full16": full, "cells": [],
              "hot_priming": [], "complete": False, "cold_reset_receipt": cold_reset_receipt,
              "sampling": {"temperature": 0, "seed": 42, "max_tokens": 256}}

    def execute(jobs, identity):
        run = lambda: dispatch_cohort(client, jobs, deadline=deadline, save_completed=save_completed)
        return measure(identity, len(jobs), deadline, run) if measure else run()

    for phase, category, concurrency in FULL if full else SCREEN:
        if time.monotonic() >= deadline:
            result["error"] = "deadline; remaining cells missing, not zero gain"
            break
        if phase == "hot":
            prime = execute(jobs_for(groups[category], campaign=campaign, category=category,
                concurrency=concurrency, phase="prime", output_tokens=1), ("prime", category, concurrency))
            result["hot_priming"].append(prime)
            if not prime["usage_complete"]:
                result["error"] = "hot priming incomplete"
                break
        cell = execute(jobs_for(groups[category], campaign=campaign, category=category,
            concurrency=concurrency, phase=phase, output_tokens=256), (phase, category, concurrency))
        counts = [row["usage"] for row in cell["rows"] if row["complete"]]
        valid_cache = len(counts) == concurrency and all(
            count["cached_tokens"] == 0 if phase == "cold" else
            count["cached_tokens"] >= count["prompt_tokens"]*0.9 for count in counts)
        cell.update(phase=phase, category=category, concurrency=concurrency, cache_verified=valid_cache)
        result["cells"].append(cell)
        if not cell["usage_complete"] or not valid_cache:
            result["error"] = "incomplete or unverified cold/hot cell"
            break
    else:
        result["complete"] = True
    result["wall_seconds"] = time.monotonic()-started
    result["native_component_metrics_status"] = "MONITOR_ATTACHED" if measure else "NOT_MEASURED"
    return result
