# Christopher Owen benchmark: attribution and exact scope

The second suite is **adapted from Christopher Owen's [spark-ds41f GitHub recipe](https://github.com/christopherowen/spark-ds41f)**. It is not an original benchmark invented for this repository. We preserve the original streaming timing and aggregate decode functions from [`bin/spark` at commit dd35633](https://github.com/christopherowen/spark-ds41f/blob/dd35633edeadd8864ec177890025b6e7c5df47ce/bin/spark).

| Input | Pin |
|---|---|
| Upstream commit | `dd35633edeadd8864ec177890025b6e7c5df47ce` |
| `bin/spark` SHA256 | `b130f172ec4e039532e9fb0b7d4d6256ddc5f16918c88d0689841146631ba46a` |
| Frozen common-context fixture SHA256 | `40427145003f73d3db195de5e98810fcc6ed294d68872df8028f9a3181fcb367` |
| Reasoning used here | Production MAX for chat; the author's historical effort is unverified |
| Decode | Code and prose, concurrency 1, 2, 4 and 8, output cap 256, temperature 0, seed 42 |
| Repeats | One discarded warmup and three measured samples for each decode point |
| Cold prefill | Nominal 32K and 256K, two repeats each, one output token, actual API prompt counts recorded |
| Excluded | 500K exceeds our 401,408 context limit |

The fixture freezes the standard-library source text generated on our Linux Python 3.12.3 host. It is public CPython code, not private user content. It is not proven byte-identical to the author's source-text corpus. Actual tokenizer counts can differ from the nominal size and are retained per request. The CPython license is preserved in `licenses/python-stdlib.txt`.

## Metric definitions

The adapter extracts only `stream_request` and `decode_window_tps` by AST from the hash-verified client. It does not import or execute the author's host-management CLI. The decode aggregate uses that function's measured streaming window, rather than the Hermes cohort wall-time denominator. Cold prefill is **actual prompt tokens / TTFT**; it includes the path to the first token and is not the same as engine-computed prefill time.

See the source function for the exact concurrency-window implementation. Tables average the three measured decode samples, not the warmup. Both individual cold-prefill repetitions are retained, and their displayed rate is their arithmetic mean. Requests can hit their 256-token cap in reasoning with no visible answer. This is throughput evidence, not a scored coding or prose quality result.

## Run the same frozen subset

Prepare the upstream source as data, before the exclusive window:

```bash
python3 benchmarks/owen/fetch_source.py
```

The download is kept under `.cache/`, checked against the SHA above, and excluded from commits. This avoids republishing the author's entire cluster-management client. Prepare the runtime with the optional local benchmark control lane as described in [BENCHMARKS.md](BENCHMARKS.md). Close other ingress, confirm all ranks healthy and idle, and use a fresh output directory and an unoccupied lease:

```bash
python3 benchmarks/owen/run.py --config config/cluster.local.json \
  --exclusive --output local-results/c4-owen-01
```

The portable adapter explicitly sends MAX on chat requests, matching the historical server default, and retains the fixed input bodies otherwise. It checks source/fixture hashes, frontend generation, exclusive lease, request usage and counter coverage per wave. It stops on unexpected traffic or incomplete requests. Completions cancellation uses stream disconnect followed by a mandatory idle check. No service restart, weight reload or API reopening is performed by this tool.

Each suite has its own 1,200-second limit. Running Hermes plus Owen was never a promise to finish both, including model loading, within 20 minutes. After a completed run, archive the inactive lease before the next suite. After failure, resolve remaining requests and inspect the failed-closed receipt before reopening ingress.

The public adapter's additional per-wave accounting and loopback transport are portability/ownership adaptations. They do not establish bit-identical client overhead to our isolated production benchmark harness. The frozen payloads and pinned timing functions provide the comparison contract; record your own hardware, graph warmup and cache behavior with new results.

## Published measurements

All 18 measured host64k arms completed this Owen subset. [The complete campaign tables](CAMPAIGN-20261012.md) link to the numerical envelopes containing every warmup, repeat, request token count, finish reason, TTFT and decode value. The experiment did not run the author's NVFP4 deployment alongside ours; it must not be used to claim a controlled EXL3-versus-NVFP4 result.
