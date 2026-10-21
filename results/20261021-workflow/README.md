# Public evidence — 1.2.4 workflow campaign

[Readable report](../../docs/WORKFLOW-BENCHMARK-20261021.md) · [recipe chooser](../../docs/RECIPES.md).
All 14 complete scored passes are included; preliminary monitor-aborted G attempts
are disclosed in the report rather than mixed into completed-pass averages.

| File | Scope |
| --- | --- |
| [aggregates.csv](aggregates.csv) | 14 recipe totals, weighted native decode, pooled request median/peak, median/best wave aggregate and timing coverage |
| [cells.csv](cells.csv) | 168 C1–C12 cells: decode, prefill, output lengths, latency percentiles, task times and repaired tools |
| [requests.csv](requests.csv) | 4,846 requests with numeric usage/timing, local task/request index; no generated content |
| [tasks.csv](tasks.csv) | 1,092 task outcomes, durations, request/tool/error counts |
| [native-counters.csv](native-counters.csv) | Native counter deltas by recipe/cell, including independent token/request/decode denominators |
| [tool-error-categories.csv](tool-error-categories.csv) | Rejected fixture/protocol/accounting/submission attempts, including repaired attempts |
| [request-speculation.csv](request-speculation.csv) | Per-request draft/accepted/round counters; not unique GPU rounds |
| [learner-summary.csv](learner-summary.csv) | Available aggregate learner counters and their cell coverage; blanks are unavailable, not zero |
| [workload-bands.csv](workload-bands.csv) | Retained C1 / C2–C6 / C7–C12 group summaries |
| [metadata.json](metadata.json) | Settings, limitations, row counts and all 168 original source-cell SHA256 digests |
| [SHA256.json](SHA256.json) | Exact byte hashes of the public CSV/JSON evidence |

Arm names: `original` = 1.2.2TF; `bert1` = Bert1; `xtf` = historical xTF/1.2.3;
`baseline` = new instrumented 1.2.3 reference; `cdef` = current 1.2.4. Other arm
names map directly to the uppercase selectors in the recipe chooser.

Native request decode and whole-wave aggregate use different denominators.
“Peak wave” is the maximum completed-wave mean; “peak request” is the largest
completed-request native rate, with its C and output length recorded. Neither is
an instantaneous GPU sample. `native_prefill_tps` uses computed tokens divided
by complete prompt-pass wall time, including prefix restoration and setup for
this task's short cached continuations. It is effective workflow prefill, not
sustained long cold-input throughput. The report includes the initial/continuation
breakdown explaining the approximately 856 tok/s CDEF value.

`post_prefill_first_token_upper_bound_seconds` excludes queue/prefill but includes
server preparation. Source “exact” host events are renamed to
`host_prefilled_to_token_event_seconds` and `host_prefilled_to_delivery_seconds`:
they start after first-token selection and are not exact GPU computation latency.
End-to-end TTFT remains available as a separate diagnostic and is not substituted
for the main post-prefill value.

Private transcripts, reasoning text, generated code/tool arguments, authentication,
host addresses and runtime process state are excluded. Source hashes bind the
export to retained private evidence, without publishing that material. Numeric
totals and all report tables can be independently checked using these public files.
Run `python3 tools/render_workflow_report.py` to regenerate Markdown.
