"""Regenerate the 1.2.4 Markdown tables offline from public numeric evidence."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/20261021-workflow'
LABELS=dict(original='1.2.2TF',bert1='Bert1',xtf='xTF / 1.2.3 (historical)',baseline='1.2.3 (new reference)',
            **{a:a.upper() for a in ['a','b','c','d','e','f','g','cf','def']},cdef='**1.2.4 / CDEF**')


def read(name):
    with (DATA/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def number(value,digits=2):
    return f'{float(value):.{digits}f}' if value not in ('',None) else '—'


def duration(value):
    value=float(value)
    return f'{int(value//60)}:{value%60:05.2f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def render():
    aggregates=read('aggregates.csv');cells=read('cells.csv')
    by={(r['arm'],int(r['C'])):r for r in cells}
    text='''# Completed-workflow benchmark: 1.2.2TF → 1.2.4

**Main recipe: [1.2.4 / CDEF](DS41F-1.2.4.md).** One matched-initial-input pass
per recipe, C1–C12, 78 scored tool workflows per recipe. All 14 recipes completed
78/78 tasks: **168 cells, 1,092 correct tasks, 4,846 model requests**.
Reasoning and tool-call tokens are included. No new model run was needed to
produce this report. [Choose a recipe](RECIPES.md) · [exact model, quant and
sampling](BENCHMARK-SETTINGS-20261020.md) · [CSV evidence](../results/20261021-workflow/).

## What changed in practice

Against the oldest matched benchmark, **1.2.4 used 14.35% less wave time**
(1,154.48 vs 1,347.87 s), or **16.75% more correct tasks/minute**. At C6,
weighted native decode increased **21.66%**, from 23.61 to 28.72 tok/s per request.
Against the new 1.2.3 reference, summed wave time fell **10.58%**.

These are observed whole-recipe results. Different output lengths and repair
paths contribute to elapsed time. CDEF wins total completed-work time here;
it does not lead every concurrency or every token-rate metric. C and CF remain
useful alternatives. Rejected tool attempts are counted, including ones repaired
before successful completion. They are not evidence of changed model quality.

## Aggregate concurrent work

All rates below are **generated completion tokens / complete wave wall time**.
The denominator includes queueing, prefill, tool execution and repairs. A wave
starts C workflows together; the number of active model streams varies as tools
run. “Peak wave” is the fastest **whole completed wave mean**, not an instantaneous
GPU decode peak. “Median wave” gives each of the 12 waves equal weight.

'''
    text+=table(['Recipe','All waves, tok/s','Median wave, tok/s','Peak wave, tok/s (C)','78 tasks, min:s','Correct','Rejected tools'],[
        [LABELS[a['arm']],number(a['workflow_completion_tps']),number(a['wave_aggregate_tps_median']),number(a['best_wave_aggregate_tps'])+' (C'+a['best_wave_C']+')',duration(a['wave_seconds']),a['completion'],a['tool_errors']] for a in aggregates])
    text+='''
## Decode per request and first token after prefill

Native request decode excludes prefill. **Weighted** divides native generated
tokens by summed request decode seconds; overlapping requests contribute separate
seconds. **Median** and **peak** are over completed requests, not GPU samples.
Peak requests can be short; their lengths and concurrency are in aggregates.csv.

**Post-prefill latency** is the comparable median **upper bound**, in milliseconds:
`U = native_TTFT − native_total + (completion_tokens − 1) / native_decode_tps`.
It excludes scheduler queueing and prefill, and includes server preparation
before submission. It is not end-to-end TTFT. Native “prefilled” occurs after
first-token selection: neither the microsecond host event nor U measures exact
GPU prefill-finish → first-token computation time. That duration was not recorded.

'''
    text+=table(['Recipe','Weighted decode, tok/s','Median request, tok/s','Peak request, tok/s','Post-prefill bound, ms','Prefill¹, tok/s'],[
        [LABELS[a['arm']],number(a['native_weighted_decode_tps']),number(a['native_decode_tps_median']),number(a['max_request_native_decode_tps']),number(float(a['post_prefill_first_token_upper_bound_seconds_median'])*1000),number(a['native_prefill_tps'])] for a in aggregates])
    text+='''
¹ Computed prompt tokens / native cumulative prefill time. This workload includes
short cached continuations; these are **not cold long-context prefill** rates.
Its request-level prefill medians answer a different question and remain in CSV.

## C1–C12: the direct decode comparison

**Weighted native decode tok/s per request** in every cell. Final column is
computed-prefill throughput pooled across all 12 cells, not an arithmetic mean
of rates. Compare rows at the same C. [All cell medians, request peaks, aggregate
rates, elapsed times and post-prefill latencies](WORKFLOW-CELLS-20261021.md).

'''
    text+=table(['Recipe']+[f'C{c}' for c in range(1,13)]+['Prefill¹'],[
        [LABELS[a['arm']]]+[number(by[a['arm'],c]['native_weighted_decode_tps']) for c in range(1,13)]+[number(a['native_prefill_tps'])] for a in aggregates])
    text+='''
## Comparison contract and limits

- Same pinned DeepSeek-V4.1-Flash-1017 weights, EXL3 routed experts ~3.5101 bpw,
  TP3 on three DGX Sparks, 2,200 MHz GPU clocks, MAX reasoning, temperature 0.6,
  seed 42, effective top-k 0 / top-p 1 / min-p 0, max_tokens 65,536, chunk 2,048.
  The request body explicitly sets temperature/seed/max_tokens; the remaining
  sampler values are engine defaults. [Full settings](BENCHMARK-SETTINGS-20261020.md).
- Original 1.2.2TF: C12/R32, 8M configured KV, 2 GiB snapshots. Bert1: C16/R48,
  7M, 2 GiB. xTF/1.2.3 and new A–G/combinations: C12/R48, 7M, 3 GiB. C is the
  workload concurrency; configured C16 in Bert1 is capacity, not a C16 test.
- Historical original/Bert1/xTF rows retain the earlier pass. The separate
  new 1.2.3 reference uses common timing instrumentation. They are not duplicate
  measurements averaged into a single winner. All 78 initial body hashes are
  frozen and verified by the [public fixture](../benchmarks/workflow/).
- One pass per recipe; no confidence intervals. Initial inputs match, subsequent
  output lengths, tool traces and context growth vary. Do not infer an isolated
  cache, speculation, learning or quality effect from recipe comparisons.
- Native counter coverage matched request/token totals in every scored cell.
  `requests.csv`, `tasks.csv` and `native-counters.csv` permit independent
  aggregation. Original source-cell SHA256 digests bind the export to retained
  evidence; private conversations, addresses and runtime state are excluded.
- The older “~50% fewer errors” observation was 31 → 16 rejected tool attempts
  in Bert1 → historical xTF. Here the new reference has 24 and CDEF 26; all final
  tasks pass. This does not establish any accuracy improvement or degradation.
- D/E/F/G mechanism counters were not captured. Their code is present, but
  how often each path fired is not proven by this archive. CDEF contains E;
  sustained-workload gains beyond this pass remain unmeasured.
- Two preliminary G attempts stopped at the external monitor. Its guard was
  repaired to use the primary local monotonic timeline; the initial trigger's
  root cause remained unproven. Only the complete G pass is scored here.
- Wave time sums exclude startup and between-cell overhead. No full 7M capacity,
  vision throughput, instantaneous aggregate GPU peak or 12-hour stability result
  is claimed. API/gateway behavior is outside this native workflow benchmark.

## Reproduce and inspect

[All recipes and mechanisms](RECIPES.md) · [1.2.4 source/build/install](DS41F-1.2.4.md)
· [online-learning design](ADAPTIVE-SPECULATION.md) · [public benchmark procedure](../benchmarks/workflow/README.md).
Rebuild these Markdown tables with `python3 tools/render_workflow_report.py`.
The preserved [earlier three-way report](WORKFLOW-BENCHMARK-20261020.md) and
[older benchmark index](BENCHMARKS.md) retain their original scopes.
'''
    detail='''# All C1–C12 cell metrics

[Summary and definitions](WORKFLOW-BENCHMARK-20261021.md) · [machine-readable evidence](../results/20261021-workflow/).
Native weighted, median and peak are per request. Aggregate is completion tokens
/ whole wave wall time. Prefill uses computed tokens. Latency is the queue/prefill
excluded upper-bound median, **not exact GPU first-token computation**.

'''
    for c in range(1,13):
        detail+=f'## C{c}\n\n'
        detail+=table(['Recipe','Decode weighted','Decode median','Request peak','Wave aggregate','Prefill','Post-prefill ms','Wave s','Rejected tools'],[
            [LABELS[a['arm']]]+[number(by[a['arm'],c][k]) for k in ['native_weighted_decode_tps','native_decode_tps_median','max_request_native_decode_tps','workflow_completion_tps','native_prefill_tps']]+[number(float(by[a['arm'],c]['post_prefill_first_token_upper_bound_seconds_median'])*1000),number(by[a['arm'],c]['wave_seconds']),by[a['arm'],c]['tool_errors']] for a in aggregates])+'\n'
    return {'docs/WORKFLOW-BENCHMARK-20261021.md':text,'docs/WORKFLOW-CELLS-20261021.md':detail.rstrip()+'\n'}


if __name__=='__main__':
    for name,content in render().items():
        (ROOT/name).write_text(content,encoding='utf-8')
        print(name)
