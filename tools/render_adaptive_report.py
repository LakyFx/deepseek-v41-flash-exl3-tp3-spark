"""Render every retained comparison cell and native counter to public Markdown."""
import argparse
import csv
import json
from pathlib import Path


INTRO = '''# TensorFold workflow benchmark: 1.2.2TF vs Bert1 vs 1.2.2xTF / 1.2.3

**Measured 20 October 2026.** This is the full Markdown companion to the
[34-page English PDF with exact model/quant/sampler addendum](../results/20261020-adaptive/workflow-report-reproducibility.pdf).
**1.2.3 is the release name for the configuration measured as 1.2.2xTF**;
the rename creates no new measurement. Evidence retains its original labels.

We measured 234 completed local agent tasks, 78 per recipe, with 1,031 model
requests. xTF had 8.76% less summed wave time than the original recipe and
16 intermediate rejected tool attempts instead of 30. These are results of
one matched-initial-input workflow pass, not proof of a general semantic
quality improvement or an isolated learner speedup.

## Navigation and evidence

- [Release configuration and installation](DS41F-1.2.3.md)
- [Exact model, quantization and sampling settings](BENCHMARK-SETTINGS-20261020.md)
- [Online cost/acceptance learner, equations and AdaSpark attribution](ADAPTIVE-SPECULATION.md)
- [All 36 comparison cells, full precision CSV](../results/20261020-adaptive/comparison-all-metrics.csv)
- [Native counter deltas](../results/20261020-adaptive/native-counters.csv)
- [234 per-task outcomes](../results/20261020-adaptive/task-results.csv)
- [1,031 per-request timings](../results/20261020-adaptive/request-timings.csv)
- [Aggregate JSON](../results/20261020-adaptive/measurement-summary.json)
- [Frozen source-evidence SHA-256 manifest](../results/20261020-adaptive/evidence-sha256.json)

Numeric exports contain no raw conversations, host addresses, credentials,
deployment configuration or images. The Markdown rounds presentation values;
the CSV files preserve the saved precision. Source digests identify private
frozen evidence without claiming that its conversations are public fixtures.

The report tables can be regenerated offline with
`python3 tools/render_adaptive_report.py --public .`; this reads only the
published numeric exports and performs no inference or network requests.

## Configurations and what changed

| Setting | Original 1.2.2TF | Bert1 | xTF, released as 1.2.3 |
|---|---:|---:|---:|
| Native serving slot cap | 12 | 16 | 12 |
| Shared verification rows, maximum | 32 | 48 | 48 |
| KV pool, tokens | 8,000,000 | 7,000,000 | 7,000,000 |
| Prompt-state snapshot cache | 2 GiB | 2 GiB | 3 GiB |
| Gateway reserved-token ceiling | 5,336,064 | 6,699,008 | 5,336,064 |
| Allocation | Original cost-based | Original cost-based | Online timing + acceptance calibration |

Original and Bert1 already allocated drafts dynamically from cost priors.
The new layer learns runtime corrections and acceptance calibration, with an
average-reward price for verification time. R includes mandatory anchor rows;
it is not a fixed per-stream speculation length. C in this campaign counts
parallel workflows. Actual native streams vary as those workflows use tools.

Held constant: DeepSeek V4.1 Flash 1017, frozen quantized weights and
tokenizer/template, Engram assets, TensorFold native CUDA stack, TP=3 on
three GB10 / Spark-class systems, 2,200 MHz GPU clock, existing graph,
prefetch, RDMA and target sampling paths. Vision remained C2/MEDIUM and
was not exercised by this campaign. Gateway admission ceilings are shown
for configuration transparency; the benchmark used the direct native API.

## Exact model, quantization and sampling

All three arms used **deepseek-ai/DeepSeek-V4.1-Flash-1017** at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, with the same
**LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold** weights at
`8621801c1fc74aece1fa83e67c6ade3fde1e7894`. Routed experts are approximately
**3.5101 bpw EXL3**; the model is mixed precision, with a K6 head, K4 DSpark
draft projections, FP8 Engram and BF16 vision. Runtime KV/index is FP4.

Effective sampling was **temperature 0.6, top_p 1.0, top_k 0 (disabled),
min_p 0.0, seed 42, thinking enabled / MAX**. The benchmark explicitly sent
temperature/seed/effort; top_p/top_k/min_p came from the checked native HTTP
defaults. No repetition/frequency/presence penalty overrides or custom stop
were requested. Each call allowed 65,536 output tokens; server context was
401,408 tokens. These are the measured settings, including temperature 0.6,
rather than the upstream recommendation of 1.0.

The [settings addendum](BENCHMARK-SETTINGS-20261020.md) includes pinned weight
links, tokenizer/template hashes, the exact request-control JSON and default
resolution evidence; [benchmark-settings.json](../results/20261020-adaptive/benchmark-settings.json)
is the machine-readable companion. The PDF adds the same information on pages
33–34, retaining the original 32 pages and all measurements.

## Workload and completion protocol

Each task repairs `merge_events`, executes isolated tests, performs paginated
usage accounting, and finishes with an explanation. Specifically:

1. Choose the newest timestamp per request ID, with later input winning ties;
   sort by request ID, do not mutate inputs, and return copies.
2. Run the actual isolated tests. The safe function evaluator forbids imports.
3. Query two pages of records, deduplicate IDs by latest time/later occurrence,
   retain backend `ds41` with status `done`, and total requests/prompt/output tokens.
4. Submit the correct accounting plus 4–6 Czech sentences, 30–180 words,
   explaining the fix, tests and accounting. This report is English; the
   tested task language was Czech.

The tools are `read_file`, `replace_file`, `run_tests`, `query_records` and
`finish_task`. Only `events.py` is available through `read_file`; attempts
to read hidden test files receive an error. The deterministic grader checks
code, executed tests, accounting and output format, not full prose semantics.

Each recipe ran C1 through C12: `1+2+...+12 = 78` workflows in 12 waves.
Initial request hashes match for every corresponding task across all 36
cells. Subsequent generated tokens, tool actions and prompts can diverge.
Sampling used temperature 0.6, seed 42 and MAX reasoning. Requests allowed
up to 65,536 tokens and the protocol had a 16-turn guard. Tasks ran to natural
completion; there was no arbitrary 120-second successful cutoff. A length
finish would not qualify as task success. Rejected tools and repair turns
remain in elapsed times and error counts.

The test used an owned tunnel to the native API with public ingress closed.
It measures local model/tool workflows, not whole Hermes sessions or public
gateway overhead. Bert1 was an already-running production process; original
TF and xTF used restarted native processes, so warmness was not identical.
There were no repeated trials, randomised profile ordering or confidence
intervals. Small-sample P95 values are descriptive observed quantiles.

## Aggregate results
'''

DISCUSSION = '''
The summed measured wave durations total **66:36.605** across all recipes.
That excludes switching, startup, waiting and an excluded partial attempt;
it is not one continuous campaign wall time. xTF's sum is 8.76% shorter than
TF and 13.33% shorter than Bert1. It generated 6.69% fewer tokens than TF
and used 331 rather than 350 model calls, so the full time reduction cannot
be attributed to native decode speed alone. Generated tokens include reasoning.

## Native decode and task duration

![Native request decode by workflow concurrency](../results/20261020-adaptive/figures/decode-by-concurrency.png)

![xTF native decode relative to original](../results/20261020-adaptive/figures/decode-relative-gain.png)

Median native per-request decode improved by **8.04% at C5, 10.63% at C6,
6.20% at C7 and 9.80% at C8** versus TF. At C12 it was **3.81% lower**.
These rates are not aggregate GPU throughput, and multiplying a request
median by workflow C is not a valid aggregate reconstruction.

![Full workflow completion duration](../results/20261020-adaptive/figures/task-duration-by-concurrency.png)

The recipe did not reduce errors by simply slowing native decode at C6/C7:
the xTF native medians were faster there. Conversely, more TPS did not always
finish the task sooner:

| Counterexample | Original TF | Bert1 | xTF |
|---|---:|---:|---:|
| C3 median task duration, s | 48.69 | 46.00 | 60.31 |
| C6 median task duration, s | 98.38 | 76.97 | 85.62 |
| C12 median task duration, s | 138.68 | 139.70 | 135.16 |

At C3, xTF generated 8,286 tokens versus TF's 5,602, despite slightly faster
decode. At C6, xTF decoded faster than Bert1 but took longer to finish.
At C12, xTF finished sooner despite its lower native median rate.

## Prefill, first token and metric scope

The native timing table below pools requests. Milliseconds are shown for
latency, and tokens/s for the weighted uncached-prefill row.
'''

LATENCY_NOTE = '''
**The xTF 3.72 ms post-prefill number is an upper bound, not its full TTFT.**
The frozen binary does not expose the exact post-prefill event interval.
The derived value removes scheduling wait and prefill but includes preparation
before submission:

```text
upper bound = native TTFT - native request seconds
              + (completion tokens - 1) / native decode TPS
```

The xTF received-to-first-token median was 382.51 ms. Exact post-prefill
latency remains null in the per-request export. Client first-delta latency
also includes client-to-server/delivery effects and may identify a reasoning
or tool delta, not the first visible prose answer token.

Native decode TPS describes the generation stage of one request. Weighted
native decode divides summed generated tokens by summed request decode times;
those times overlap under concurrency. Whole-wave TPS divides all generated
tokens by wave wall time, including prefill/tool/repair gaps. Weighted
uncached prefill uses actually computed prompt tokens over native prefill time,
excluding reused prefix tokens. These denominators are not interchangeable.

## Intermediate tool errors

Every recipe ultimately passed **78/78** tasks. These counts are intermediate
rejected attempts, not final task failures, HTTP crashes or native restarts.
'''

ERROR_NOTE = '''
The reduction from TF's 30 to xTF's 16 is **46.67% fewer counted rejections**;
Bert1's 31 to 16 is 48.39%. It is not a demonstrated reduction of the target
model's general reasoning error probability. Tool attempts were 508 / 508 /
487, and affected tasks were 20 / 20 / 13. Trajectories and denominators differ.

Most of the reduction came from unavailable test-file reads and incorrect
accounting. Import restrictions still caused ten xTF rejections.

![Rejected tools by concurrency](../results/20261020-adaptive/figures/tool-errors-by-concurrency.png)

| Rejection band | Original TF | Bert1 | xTF |
|---|---:|---:|---:|
| C1–C8 | 8 | 7 | 12 |
| C9–C10 | 13 | 15 | 0 |
| C11–C12 | 9 | 9 | 4 |

xTF had more errors in C1–C8, with its advantage concentrated later. One
pass does not establish that concurrency caused errors or that a particular
cache/allocator mechanism prevented them.

## Memory, KV and prompt snapshots
'''

MEMORY_NOTE = '''
The Bert1 retained per-rank monitor files cover the continuation window;
TF/xTF monitoring covers their measured sweeps. The minima therefore are
descriptive, not a controlled full-run memory comparison. Cold historical
swap occupancy is not the same as current swap pressure. xTF sampled full
memory PSI remained zero; native cancellations and preemptions were zero
for all three recipes.

A larger prompt-state snapshot cache can reduce recomputation when retained
state is reused. Eviction does not show that the next supplied prompt lost
its text. xTF's prefix **token** reuse fraction was actually lower here:
81.46% versus 83.16% TF. This run does not establish a cache-driven speedup
or semantic quality gain. Prefix counters are token-denominated, not snapshot
object hit counts. The configured 7M/8M logical KV capacities were not filled
or stress-qualified by this workflow.

## Learner telemetry

The learner changes the shared row budget using online cost and conditional
acceptance feedback. Original TF/Bert already had cost-based allocation.
The [complete design](ADAPTIVE-SPECULATION.md) explains buckets, priors,
calibration, censored labels, average-reward pricing and fallbacks. No target
or draft weights were trained; the target verifier remains authoritative.

The idea was inspired by [AdaSpark, Liu et al.](https://arxiv.org/abs/2610.05774),
adapted to existing TensorFold chain/copy lanes. It is not a complete AdaSpark
or imparo replication. Our pinned/extrapolated priors, rolling median and
bucket calibration are implementation choices, not the paper's full method.
The paper's speedup figures are not our benchmark figures.
'''

LIMITS = '''
Row ranges use 3-second telemetry samples, not every-round extrema. Native
stream count is not necessarily the C label of the enclosing workflow wave.
The timing error is `sum(abs(predicted - actual)) / sum(actual)`, not MAPE,
not a convergence curve and not “95.24% prediction accuracy”. Its mean
absolute error was 4.10 ms over 11,267 qualified samples and 969.52 s summed
measured verification time. The final sample had 1 stream / 6 rows / 4,023
context tokens; predicted/actual verify times were 46.659 / 47.453 ms.

The learner is shared across requests in one native process. A new chat does
not reset it; a process restart returns to priors. It does not persist to
disk or guarantee monotonic improvement. Rare probes are useful normal rounds,
not extra model benchmark calls.

## Interpretation and limitations

The strongest evidence is the recipe-level result: faster middle-load decode,
shorter summed workflow time, higher pooled draft acceptance and fewer repair
attempts in this pass. The changed row cap, KV budget, snapshot cache and
learner are not separate ablations. Different output volume and tool paths
also change wall time. Higher acceptance is an efficiency metric; the target
verifier does not grade the meaning of a tool command.

The frozen sampler keys randomness by seed, absolute position and token ID.
We have no basis to blame RNG-consumption order for trajectory differences.
GPU batch shapes can in general change floating-point results; see the
[PyTorch numerical accuracy background](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
No token-level numerical root-cause diagnosis was performed in this experiment.
General speculative-inference background is
[Leviathan et al. (2023)](https://proceedings.mlr.press/v202/leviathan23a.html).

This campaign does not validate long-session repetition resistance, semantic
quality on other tasks, full 7M/8M pool capacity, image throughput, mixed
text/vision contention, public-gateway latency or hours-long production
reliability. No such measurements are reconstructed from token-rate peaks.

## Run integrity and operational exceptions

An initial Bert1 C10 attempt was stopped by a cold swap-occupancy guard after
roughly 304 MiB of growth, without evidence of real memory pressure. Valid
C1–C9 cells were reused only after strict initial-input hash checks. C10–C12
were run in a shared fresh continuation namespace across all three recipes.
The retained partial attempt is excluded from the displayed cells, not
silently counted as a successful benchmark or a model-quality failure.

The first post-test API reopening did not complete, and the controller rolled
back automatically. A later promotion reopened and verified the selected
xTF API. Those are separate operational events, not the model/tool error
counts above. Switching/startup/deployment times are excluded from summed
wave durations. Pre-benchmark CPU-preparation receipts keep their historical
status labels; the later GPU workflow is the measurement reported here.

## Complete per-concurrency measurements

Each table below shows all C1–C12 cells in Original TF / Bert1 / xTF order.
TF means 1.2.2TF; xTF means the configuration now released as 1.2.3.
No fastest-repeat selection, synthetic missing cells or confidence intervals
are inserted. See the CSV supplements for full precision and individual tasks.
'''


def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
        ['| '+' | '.join(str(v).replace('|','\\|') for v in row)+' |' for row in data])+'\n\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);a=p.parse_args()
    result=a.public/'results/20261020-adaptive'
    s=json.loads((result/'measurement-summary.json').read_text(encoding='utf-8'))
    arms=['original','bert1','xtf'];labels=['Original TF','Bert1','xTF / 1.2.3'];A=s['aggregates']
    text=INTRO+'\n'+table(['Metric']+labels,[
        ['Completed / attempted tasks']+['78 / 78']*3,
        ['Sum of measured wave times, s']+[f"{A[x]['wave_seconds']:.3f}" for x in arms],
        ['Task duration median / P95 / max, s']+[' / '.join(f"{A[x]['all_task_seconds'][k]:.3f}" for k in ['median','p95','maximum']) for x in arms],
        ['Model requests']+[A[x]['requests'] for x in arms],
        ['Generated tokens, including reasoning']+[A[x]['generated_tokens'] for x in arms],
        ['All prompt tokens']+[A[x]['prompt_tokens'] for x in arms],
        ['Actually computed prompt tokens']+[A[x]['computed_prompt_tokens'] for x in arms],
        ['Rejected intermediate tool attempts']+[A[x]['corrected_tool_errors'] for x in arms],
        ['Tasks with a rejection']+[s['error_breakdown'][x]['tasks_with_error'] for x in arms],
        ['Draft acceptance, %']+[f"{A[x]['acceptance']*100:.4f}" for x in arms],
        ['Pooled native decode median / P95 / max, tok/s']+[' / '.join(f"{A[x]['native_decode_tps'][k]:.3f}" for k in ['median','p95','maximum']) for x in arms],
    ])+DISCUSSION
    text+=table(['Native metric']+labels,[
        [title]+[' / '.join(f'{A[x][key][k]*1000:.3f}' for k in ['median','p95','maximum']) for x in arms]
        for title,key in [('Prefill median / P95 / max, ms','prefill_seconds'),
                         ('TTFT median / P95 / max, ms','native_ttft_seconds'),
                         ('Post-prefill upper bound median / P95 / max, ms','post_prefill_first_token_upper_bound_seconds')]
    ]+[['Weighted uncached prefill, tok/s']+[f"{A[x]['weighted_prefill_tps']:.3f}" for x in arms]])
    text+=LATENCY_NOTE
    text+=table(['Exact tool error returned']+labels,[[key]+[s['error_breakdown'][x]['errors'].get(key,0) for x in arms]
        for key in ['unsupported pure-function syntax: Import','unsupported pure-function syntax: ImportFrom',
                    'one merge_events function required','unknown tool or invalid arguments',
                    'incorrect accounting: requests','provide a concise Czech explanation, 30 to180 words',
                    'provide four to six explanatory sentences']])
    text+=ERROR_NOTE
    text+=table(['Memory/cache observation']+labels,[
        ['Minimum sampled MemAvailable, GiB']+[f"{A[x]['minimum_available_gib']:.3f}" for x in arms],
        ['Maximum sampled full-memory PSI avg10, %']+[f"{A[x]['maximum_memory_full_psi_avg10']:.2f}" for x in arms],
        ['Native cancellations / preemptions']+['0 / 0']*3,
        ['Prefix token reuse fraction, %']+[f"{A[x]['prefix_hit_fraction']*100:.4f}" for x in arms],
    ])
    text+=MEMORY_NOTE
    learning=s['learner']
    text+=table(['Learner measurement','Value'],[
        ['Rounds',int(learning['adaptive_rounds_total'])],['Cost updates',int(learning['adaptive_cost_updates_total'])],
        ['Acceptance labels',int(learning['adaptive_acceptance_updates_total'])],['Probes',int(learning['adaptive_probes_total'])],
        ['Rejected timing samples',int(learning['adaptive_rejected_times_total'])],
        ['Qualified prediction samples',int(learning['adaptive_prediction_samples_total'])],
        ['Normalised absolute timing error, %',f"{learning['timing_normalized_absolute_error']*100:.6f}"],
    ])
    text+='![Sampled verification-row ranges](../results/20261020-adaptive/figures/adaptive-row-ranges.png)\n\n'
    text+=table(['Active native streams','Sampled minimum rows','Sampled maximum rows'],[[k,*v] for k,v in learning['observed_rows_range_by_streams'].items()])
    text+=LIMITS
    with (result/'comparison-all-metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows={(r['arm'],int(r['C'])):r for r in csv.DictReader(f)}
    with (result/'native-counters.csv').open(encoding='utf-8-sig',newline='') as f:native={(r['arm'],int(r['C'])):r for r in csv.DictReader(f)}
    def metric(label,key,scale=1,digits=2,source=rows):return label,key,scale,digits,source
    sections=[
        ('A1 / Full workflow durations',[metric('Wave elapsed, s','wave_elapsed_seconds'),metric('Median task, s','task_median_seconds'),metric('P95 task, s','task_p95_seconds')],
         'Wave elapsed ends when the last task finishes; per-task distributions include tools and repairs.'),
        ('A2 / Native decode distributions',[metric('Median, tok/s','native_decode_tps_median'),metric('P95, tok/s','native_decode_tps_p95'),metric('Weighted, tok/s','weighted_decode_tps')],
         'These are native per-request rates, not aggregate GPU throughput. Weighted time is a sum of overlapping request decode times.'),
        ('A3 / Full request and wave rates',[metric('Request median, tok/s','per_request_completion_tps_median'),metric('Whole-wave, tok/s','aggregate_workflow_completion_tps'),metric('Correct tasks/min','correct_tasks_per_minute')],
         'Request wall includes queue/prefill/delivery; whole-wave includes tools and repairs. Reasoning is included.'),
        ('A4 / Prefill timing and weighted speed',[metric('Median, ms','prefill_seconds_median',1000),metric('P95, ms','prefill_seconds_p95',1000),metric('Weighted, tok/s','weighted_prefill_tps')],
         'Speed uses actually computed input, excluding prefix reuse. These short-request rates are not long uncached-prompt microbenchmarks.'),
        ('A5 / Computed prefill work',[metric('Median, tok/s','computed_prefill_tps_median'),metric('P95, tok/s','computed_prefill_tps_p95'),metric('Computed tokens','computed_prefill_tokens',1,0)],
         'Per-request quantiles and total-token / total-time weighted rates use different weighting.'),
        ('A6 / Native TTFT and client first delta',[metric('Native median, ms','native_ttft_seconds_median',1000),metric('Native P95, ms','native_ttft_seconds_p95',1000),metric('Client median, ms','ttft_median_seconds',1000)],
         'Client delta may be reasoning or a tool delta. Native TTFT begins at request receipt.'),
        ('A7 / Post-prefill bound and acceptance',[metric('Bound median, ms','post_prefill_first_token_upper_bound_seconds_median',1000,3),metric('Bound P95, ms','post_prefill_first_token_upper_bound_seconds_p95',1000,3),metric('Acceptance, %','acceptance',100)],
         'Bound is not exact post-prefill latency; acceptance is draft/target agreement, not task correctness.'),
        ('A8 / Requests, output and errors',[metric('Requests','requests',1,0),metric('Generated tokens','completion_tokens',1,0),metric('Rejected tools','task_errors',1,0)],
         'Same initial inputs; later trajectories and output volumes differ.'),
        ('A9 / Native drafts and request rounds',[metric('Drafted','mtp_drafted_total',1,0,native),metric('Accepted','mtp_accepted_total',1,0,native),metric('Request-round sum','decode_rounds_total',1,0,native)],
         'The request-round sum is not a count of unique shared GPU rounds.'),
        ('A10 / Prefix counters and output per request-round',[metric('Token queries','prefix_cache_queries_total',1,0,native),metric('Token hits','prefix_cache_hits_total',1,0,native),metric('Output/req-round','output_tokens_per_request_decode_round',1,3,native)],
         'Prefix queries/hits count tokens, not snapshot objects. Output/req-round is not shared-round device throughput.'),
        ('A11 / Native summed time and reuse',[metric('Prefill seconds','prefill_duration_seconds_total',1,3,native),metric('Decode seconds','request_decode_seconds_sum',1,2,native),metric('Prefix reuse, %','prefix_token_hit_fraction',100,2,native)],
         'Overlapping request times can exceed wave wall time. Prefix reuse equals token hits / token queries.'),
    ]
    for title,metrics,note in sections:
        text+='\n### '+title+'\n\n'+note+'\n\n'
        headers=['C']+[label+' / '+arm for label,*_ in metrics for arm in ['TF','Bert','xTF']]
        data=[]
        for cc in range(1,13):
            data.append([cc]+[f'{float(source[arm,cc][key])*scale:,.{digits}f}'
                for label,key,scale,digits,source in metrics for arm in arms])
        text+=table(headers,data)
    text+='''\n## Coverage, integrity and how to use these numbers

All 36 cells had `correct = total = C`; `runtime_coverage_requests = requests`,
`coverage=True` and `failures_retained=True`. The full-precision CSV retains
these fields and `rate_scope`. Native generated/computed-token and request
counts agree with the matching comparison cells. There were no native
cancellations or preemptions. Printed numeric tables in the PDF were checked
independently against 1,188 source values; the Markdown tables are generated
from those same numeric exports. A release test verifies them against CSV.

The reusable lesson is to compare task duration, output volume, repair
categories and latency definitions beside TPS. xTF was the strongest overall
recipe in this pass, but it did not win every cell and this single pass does
not establish a general model-quality gain. Historical benchmark campaigns
elsewhere in the repository retain their own models, scopes and metric
definitions; do not blend them into this matched-input comparison.
'''
    path=a.public/'docs/WORKFLOW-BENCHMARK-20261020.md'
    path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(markdown=str(path),lines=len(text.splitlines()),cells=36)))


if __name__=='__main__':main()
