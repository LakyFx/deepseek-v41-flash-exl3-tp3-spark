# Matched completed-workflow benchmark

This publishes the authored synthetic task used for the 1.2.2TF / Bert1 / xTF /
1.2.3 / A–G / CF / DEF / CDEF comparison. It is separate from the older public
surrogate benchmark. **All 78 initial request body hashes match the frozen
campaign**, including its C10–C12 namespace continuation. Czech task wording,
tools, sampler fields, code cases and accounting pages retain their semantics.

The task repairs a small pure Python `merge_events`, runs actual semantic tests,
reads two synthetic accounting pages, deduplicates records and submits checked
totals plus a short Czech explanation. The restricted evaluator rejects unsupported
syntax and checks non-mutation/copies, tie-breaking and ordering. It does not
score all prose semantics. Tool rejections remain in the trace and measured time.

## Inspect offline

```sh
python3 -m benchmarks.workflow.run
python3 -m unittest discover -s tests
python3 tools/render_workflow_report.py
```

The first command checks all frozen hashes and prints preparation information;
it makes **zero model requests**. Source imports/transport packaging are portable
adaptations; the private SSH deployment/remote process monitor is not distributed.
The public runner does not implement that fleet controller or its memory monitor.

## Run in your own exclusive window

Prepare/qualify one [native recipe](../../docs/RECIPES.md). Close external
admission and drain existing traffic using your own deployment controls; start
the loopback benchmark backend. The runner does not change services or drain for
you. Provide a fresh JSON receipt that truthfully records that state:

```json
{
  "arm": "cdef",
  "campaign_id": "matched-workflow-20261020-f2ad3fb73c11",
  "profile_sha256": "SHA256_OF_YOUR_SELECTED_PROFILE_FILE",
  "exclusive_window": true,
  "public_api_closed": true,
  "engine_generation": "YOUR_VERIFIED_DEPLOYMENT_ID",
  "ranks_verified": [0, 1, 2],
  "user_go_reference": "YOUR_APPROVED_TEST_WINDOW",
  "created_epoch": 0
}
```

Set created_epoch to the current Unix time; the receipt must be no older than
900 seconds at campaign start. The receipt is an operator assertion, not an
automatic verification of remote ranks. Use your actual loopback port:

```sh
python3 -m benchmarks.workflow.run --run --exclusive --arm cdef \
  --port 8080 --profile tensorfold/profiles/1.2.4.json \
  --receipt /srv/bench-receipt.json --output /srv/bench-cdef-new
```

One C1–C12 wave sequence, 78 tasks, fresh output directory; no automatic retries,
no overwrites and no automatic API reopening. Each wave requires an idle backend;
request/token counter mismatches or unresolved owned disconnects stop the run.
Failures and partial cells remain available. Per-task protocol guard is 16 turns;
the 120-second design target is not a success deadline. Socket stall timeout is
900 seconds. Use operator monitoring appropriate to your hardware.

Result files contain generated task transcripts; keep your own results private
until reviewed. The published [numeric evidence](../../results/20261021-workflow/)
omits assistant text, reasoning, tool arguments, addresses and runtime state.
The post-prefill metric is an upper bound; [definitions and limitations](../../docs/WORKFLOW-BENCHMARK-20261021.md)
explain why microsecond host events are not GPU first-token computation latency.
