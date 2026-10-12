# Reproduce and inspect the tuning variants

The history contains several campaigns, not one giant controlled sweep. Start with [HISTORY.md](HISTORY.md) for A/B/X/X11 follow-ups and [CAMPAIGN-20261012.md](CAMPAIGN-20261012.md) for the latest host64k baseline, isolated changes and combinations.

## What is available

- `results/`: historical numerical evidence, including failed/rejected experiments and incomplete cells.
- `results/20261012-host64k-campaign/`: all 18 final result envelopes, separate Hermes/Owen/curve measurements, qualification evidence and summary generation.
- `experiments/20261012/variants.json`: exact per-arm source map, environment, specification and compilation arguments. Concurrency C1 and candidate named C1 are different concepts.
- `experiments/20261012/baseline.json`: mounted source baseline for B0 on the upgraded host, without private host locations.
- `experiments/20261012/sources/`: content-addressed source files, shared where identical. Hashes are verified when materializing an arm.
- `runtime/rootfs/` and `native/`: selected production C4 source, used by the default build.

The source archive deliberately includes non-selected implementations. You can inspect the real N5/N7/N8 changes rather than infer them from a benchmark name. The N9 plan is retained but its trace found no useful additional graph bucket, so it has no invented performance row and the materializer refuses it.

## Create a separate experimental build

Use a Git clone with tags available. The pre-N4 `gemm.cu` is recovered from the retained `v0.8.9X11c` tag; the native build is necessary when comparing different kernels.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N6 --output /srv/dsv41-build-n6
cd /srv/dsv41-build-n6
RECIPE_IMAGE=ds41f:experiment-n6 bash build/build.sh
cp config/cluster.example.json config/cluster.local.json
# Supply the same actual topology, weights and isolated writable cache paths.
```

Supported measured choices are B0, N1, N2, N3, N4, N5, N6, N7a, N7b, N8a, N8b, C1, C2, C3, C4, C5-indexer, C5-sp512 and C5-sp256. Materialization only writes a new directory outside this checkout. It does not SSH, execute Docker, reload the model or benchmark anything.

The tool combines the complete baseline source map with the selected exact overlay, uses that arm's full recorded environment and compilation configuration, and restores the baseline native kernel when N4 is absent. Source-level changes cannot be undone by guessing environment switches. The source archive records measured hashes separately where the benchmark control peer was changed into an operator input for public use.

This is a **new public build**, not the original campaign's loaded image. Verify the build, all three rank identities, numerical behavior, cache/vision correctness and the 3.3M pool before accepting its measurements. The archived qualification RPCs retain their strict original campaign ownership contract; the public measurement runner does not forge those receipts or automatically execute the private fleet controller. Use the retained numerical gates as a reference for your own qualification.

## Controlled comparison

Keep checkpoint revision, host/kernel/driver, memory packing, TP topology, clocks, KV allocation, context limit, output caps and reasoning fixed. Prepare all images and inputs first. Then close ingress once, load a qualified arm, run Hermes, archive the inactive lease, run Owen, save results, and continue to the next arm. Reset prefixes through the owned control lane; do not import an unrelated KV snapshot. Loading/qualification time is additional to the per-suite cap.

Use the standard Hermes coverage20 suite as the primary workload comparison. The Owen common-context suite and supplemental C1–C8 request-overlap curve answer different questions. Do not replace an unfavorable Hermes result with an Owen number using a different denominator. Preserve partial results, output lengths, cache proof and failed task scores.

The public authored workflow/long-context corpus is included. The historical legacy code/prose screen also used private captured material, which is deliberately absent. Public synthetic replacements make the runner usable, but results with those replacements are new experiments, not an exact replay of that historical subset. Use your own representative private workload locally when selecting a deployment.

## Earlier campaigns

The earlier A/B/X/X11 tables preserve their actual measured scopes, including the October 2 full C1–C4 cold/hot matrix, the X1–X11 coverage20 sweep, independent X11a–e follow-ups, and X11c none/medium/MAX comparisons. They do not all have a separately packaged installable image. `v0.8.9X11c` preserves the previous public recipe. No missing version is assigned a synthetic speed or an untested combination claimed to be qualified.
