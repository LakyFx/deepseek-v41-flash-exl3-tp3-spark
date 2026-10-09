"""Evidence-based workload comparisons; a context-specific gain is retained."""
from __future__ import annotations

import math


def workflow_summary(rows: list[dict], *, wall_seconds: float | None = None) -> dict:
    families = {}
    for row in rows:
        family = row["family"]
        totals = families.setdefault(family, {"attempts": 0, "correct": 0,
                                              "elapsed_seconds": 0.0,
                                              "tool_calls": 0})
        duration = row["elapsed_seconds"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("positive finite workflow duration required")
        totals["attempts"] += 1
        totals["correct"] += int(row.get("correct") is True)
        totals["elapsed_seconds"] += duration
        totals["tool_calls"] += row.get("tool_calls", 0)
    for totals in families.values():
        totals["success_rate"] = totals["correct"] / totals["attempts"]
        totals["correct_tasks_per_task_minute"] = totals["correct"] * 60 / totals["elapsed_seconds"]
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("positive finite cohort wall time required")
    return {"families": families,
            "cohort_wall_seconds": wall_seconds,
            "correct_tasks_per_wall_minute": (sum(row.get("correct") is True for row in rows)
                                               * 60 / wall_seconds) if wall_seconds else None,
            "single_blended_score": None,
            "note": "Report code/tool/analysis separately; aborted or failed work stays in denominators."}


def compare_cases(baseline: list[dict], candidate: list[dict]) -> dict:
    def index(rows):
        indexed = {}
        for row in rows:
            key = (row["case_id"], row["family"], row["regime"])
            if key in indexed:
                raise ValueError("duplicate benchmark comparison identity")
            indexed[key] = row
        return indexed
    left, right = index(baseline), index(candidate)
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            rows.append({"case_id": key[0], "family": key[1], "regime": key[2],
                         "status": "UNPAIRED_NOT_ZERO_GAIN"})
            continue
        if a.get("correct") is not True or b.get("correct") is not True:
            rows.append({"case_id": key[0], "family": key[1], "regime": key[2],
                         "status": "QUALITY_OR_COMPLETION_FAILURE",
                         "baseline_correct": a.get("correct"), "candidate_correct": b.get("correct")})
            continue
        before, after = a["elapsed_seconds"], b["elapsed_seconds"]
        if any(not math.isfinite(x) or x <= 0 for x in (before, after)):
            raise ValueError("invalid measured time")
        rows.append({"case_id": key[0], "family": key[1], "regime": key[2],
                     "status": "PAIRED_SCREEN", "speedup": before / after,
                     "time_saved_seconds": before - after,
                     "baseline_seconds": before, "candidate_seconds": after})
    return {"cases": rows, "automatic_performance_rejection": False,
            "inference": "One run screens gains; long-context/sampled/mixed regimes retain their own results."}
