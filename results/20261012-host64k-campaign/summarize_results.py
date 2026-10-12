"""Summarize immutable X11c campaign result envelopes for review.

Run from this folder after copying result.<arm>.pass1.json envelopes here.
The two benchmark suites deliberately keep their own measurement units.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = (
    "B0", "N1", "N2", "N3", "N4", "N5", "N6", "N7a", "N7b",
    "N8a", "N8b", "N9", "C1", "C2", "C3", "C4",
    "C5-indexer", "C5-sp512", "C5-sp256",
)
OWEN_POINTS = (
    "code-c1", "code-c2", "code-c4", "code-c8",
    "prose-c1", "prose-c2", "prose-c4", "prose-c8",
)


def nested(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def extract(arm):
    path = ROOT / f"result.{arm}.pass1.json"
    if not path.exists():
        return None
    envelope = json.loads(path.read_bytes())
    if envelope["arm"] != arm or envelope["pass_number"] != 1:
        raise ValueError(f"wrong result envelope: {path}")
    result = envelope["result"]
    if result["arm"] != arm:
        raise ValueError(f"wrong result payload: {path}")
    hermes = result["paired"]["hermes"]
    owen = result["paired"]["owen"]
    phases = {phase["id"]: phase for phase in hermes["phases"]}
    if not all(name in phases for name in
               ("workflow-C1", "workflow-C3", "64K-C1", "128K-C3", "8K-C8")):
        raise ValueError(f"missing Hermes phase: {arm}")

    def h(*keys):
        return nested(phases[keys[0]]["result"], *keys[1:])

    row = {
        "arm": arm,
        "quality": nested(result, "quality", "correct") is True,
        "hermes_complete": hermes["complete"] is True,
        "owen_complete": owen["complete"] is True,
        "curve_complete": nested(result, "curve", "complete") is True,
        "h_c1_decode": h("workflow-C1", "native_metrics", "weighted_request_decode_tokens_per_second"),
        "h_c3_decode": h("workflow-C3", "native_metrics", "weighted_request_decode_tokens_per_second"),
        "h_c8_wall": h("8K-C8", "hot", "aggregate_output_tokens_per_wall_second"),
        "h_64k_prefill": h("64K-C1", "prime", "native_metrics", "weighted_request_computed_prefill_tokens_per_second"),
        "h_128k_prefill": h("128K-C3", "prime", "native_metrics", "weighted_request_computed_prefill_tokens_per_second"),
        "h_128k_complete": phases["128K-C3"]["complete"] is True,
    }
    for point in OWEN_POINTS:
        measured = [sample["aggregate_decode_tps"] for sample in owen["rows"]
                    if sample.get("point") == point and sample.get("sample") != "warmup"]
        if len(measured) != 3:
            raise ValueError(f"Owen {point} does not have three measured samples: {arm}")
        row["o_" + point.replace("-", "_")] = statistics.mean(measured)
    for length in (32768, 262144):
        measured = [sample["prompt_tokens_per_ttft_second"] for sample in owen["rows"]
                    if sample.get("nominal_tokens") == length]
        if len(measured) != 2:
            raise ValueError(f"Owen prefill {length} does not have two repeats: {arm}")
        row[f"o_{length // 1024}k_prefill"] = statistics.mean(measured)
    return row


def fmt(value):
    return "n/a" if value is None else f"{value:.1f}"


def show(title, rows, columns):
    print(f"\n{title}")
    print("| " + " | ".join(label for label, _ in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for _, key in columns:
            value = row["arm"] if key == "arm" else fmt(row[key])
            if key == "h_128k_prefill" and not row["h_128k_complete"] and value != "n/a":
                value += "*"
            cells.append(value)
        print("| " + " | ".join(cells) + " |")


def main():
    rows = [row for arm in ARMS if (row := extract(arm)) is not None]
    show("Hermes (engine decode C1/C3, hot wall C8, computed cold prefill)", rows,
         (("Arm", "arm"), ("C1", "h_c1_decode"), ("C3", "h_c3_decode"),
          ("C8", "h_c8_wall"), ("64K", "h_64k_prefill"),
          ("128K", "h_128k_prefill")))
    show("Owen code (mean of three measured samples)", rows,
         (("Arm", "arm"), ("C1", "o_code_c1"), ("C2", "o_code_c2"),
          ("C4", "o_code_c4"), ("C8", "o_code_c8")))
    show("Owen prose and TTFT-based cold prefill", rows,
         (("Arm", "arm"), ("C1", "o_prose_c1"), ("C2", "o_prose_c2"),
          ("C4", "o_prose_c4"), ("C8", "o_prose_c8"),
          ("32K", "o_32k_prefill"), ("256K", "o_256k_prefill")))
    print("\nIncomplete suite coverage: " + ", ".join(
        row["arm"] for row in rows if not all(row[key] for key in
            ("quality", "hermes_complete", "owen_complete", "curve_complete"))) or
        "\nIncomplete suite coverage: none")
    print("128K-C3 incomplete: " + ", ".join(
        row["arm"] for row in rows if not row["h_128k_complete"]) or
        "\n128K-C3 incomplete: none")
    print("* 128K cold prime prefill was measured, but the full 128K-C3 phase did not finish.")
    print(f"Result envelopes present: {len(rows)}/{len(ARMS)}")


if __name__ == "__main__":
    main()
