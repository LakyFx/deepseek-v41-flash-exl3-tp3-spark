# Variant recipes

[Choose by measured workload](../README.md#choose-your-variant) · [Complete results](CAMPAIGN-20261012.md) · [Build verification](VARIANT-VALIDATION.md)

All 18 measured arms use the same pinned EXL3 checkpoint, TP3 and cache allocation. N9 was a no-op and is not an installable measured arm. C4 is DS41F 1.0.0. Each command below starts in the repository root with tags fetched and creates a new directory outside it. Use a fresh output path if one exists; no automatic cleanup occurs.

After building, copy `config/cluster.example.json` to `config/cluster.local.json`, configure your own topology, weights and independent writable caches, and follow [installation from checkpoint preparation](INSTALL.md#3-download-and-prepare-the-checkpoint) and [operator acceptance](VERIFICATION.md#operator-acceptance). Build once, use the identical image on all three ranks. No recipe here deploys automatically.

Verification labels for every arm: **historical GPU campaign recorded**; **source reconstruction separately checked**; **fresh public image build/ABI and GPU requalification are not implied**. Consult the [status and receipts](VARIANT-VALIDATION.md) for actual evidence.

## B0

**Change:** Fresh X11c on kernel 7.0 / 64K pages.

**Reason to compare / limitation:** Same-host reference, not the old October 9 score copied forward.

**Observed versus B0:** Hermes C1 62.3 tok/s (+0.0%); Hermes C8 159.0 tok/s (+0.0%); 64K prefill 1318.0 tok/s (+0.0%); Owen code C8 192.9 tok/s (+0.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm B0 --output /srv/dsv41-build-b0
cd /srv/dsv41-build-b0
RECIPE_IMAGE=ds41f:experiment-b0 bash build/build.sh
```

## N1

**Change:** Full valid K5 verification for eligible single-request pure decode.

**Reason to compare / limitation:** Avoid over-short verification; isolated C1 declined, so full verification is not a universal win.

**Observed versus B0:** Hermes C1 58.5 tok/s (-6.1%); Hermes C8 156.0 tok/s (-1.9%); 64K prefill 1283.4 tok/s (-2.6%); Owen code C8 205.6 tok/s (+6.6%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N1 --output /srv/dsv41-build-n1
cd /srv/dsv41-build-n1
RECIPE_IMAGE=ds41f:experiment-n1 bash build/build.sh
```

## N2

**Change:** Stock lagged mHC for 1–6 pure-decode rows; fused mixed/prefill retained.

**Reason to compare / limitation:** Reduce small-batch overhead; phase metadata and eager tiny-prefill fallback are required.

**Observed versus B0:** Hermes C1 62.1 tok/s (-0.4%); Hermes C8 158.5 tok/s (-0.3%); 64K prefill 1405.0 tok/s (+6.6%); Owen code C8 196.5 tok/s (+1.8%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N2 --output /srv/dsv41-build-n2
cd /srv/dsv41-build-n2
RECIPE_IMAGE=ds41f:experiment-n2 bash build/build.sh
```

## N3

**Change:** Reusable logits scratch and batched position keys.

**Reason to compare / limitation:** Cheaper same-distribution draft bookkeeping; isolated C3 improved, C1/C8 declined.

**Observed versus B0:** Hermes C1 55.1 tok/s (-11.6%); Hermes C8 151.2 tok/s (-4.9%); 64K prefill 1284.8 tok/s (-2.5%); Owen code C8 208.5 tok/s (+8.1%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N3 --output /srv/dsv41-build-n3
cd /srv/dsv41-build-n3
RECIPE_IMAGE=ds41f:experiment-n3 bash build/build.sh
```

## N4

**Change:** Four-warp specialization of the existing large-row EXL3 kernel.

**Reason to compare / limitation:** Exact TP3 routed shapes, MUL1 K3/K4, FP32 accumulation retained; decode K path unchanged.

**Observed versus B0:** Hermes C1 63.5 tok/s (+1.9%); Hermes C8 150.7 tok/s (-5.2%); 64K prefill 1400.2 tok/s (+6.2%); Owen code C8 204.2 tok/s (+5.9%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N4 --output /srv/dsv41-build-n4
cd /srv/dsv41-build-n4
RECIPE_IMAGE=ds41f:experiment-n4 bash build/build.sh
```

## N5

**Change:** Exact compatible indexer optimization.

**Reason to compare / limitation:** Selection/masking must remain correct; good isolated C1/C3 did not imply the best combined C3.

**Observed versus B0:** Hermes C1 65.4 tok/s (+5.1%); Hermes C8 154.6 tok/s (-2.8%); 64K prefill 1337.4 tok/s (+1.5%); Owen code C8 191.0 tok/s (-1.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N5 --output /srv/dsv41-build-n5
cd /srv/dsv41-build-n5
RECIPE_IMAGE=ds41f:experiment-n5 bash build/build.sh
```

## N6

**Change:** Bounded decoder replay, cache/metadata/vision/SP integration.

**Reason to compare / limitation:** Largest isolated long-prefill gain; lower Hermes C8, full 128K phase completed.

**Observed versus B0:** Hermes C1 59.4 tok/s (-4.6%); Hermes C8 145.7 tok/s (-8.4%); 64K prefill 1770.3 tok/s (+34.3%); Owen code C8 187.8 tok/s (-2.6%).

**Full bounded 128K-C3 phase:** completed.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N6 --output /srv/dsv41-build-n6
cd /srv/dsv41-build-n6
RECIPE_IMAGE=ds41f:experiment-n6 bash build/build.sh
```

## N7a

**Change:** Pure-prefill row-partition threshold 512 instead of 2048.

**Reason to compare / limitation:** More work distributed sooner, more communication opportunities; no isolated prefill improvement.

**Observed versus B0:** Hermes C1 65.3 tok/s (+4.9%); Hermes C8 149.8 tok/s (-5.8%); 64K prefill 1286.6 tok/s (-2.4%); Owen code C8 196.7 tok/s (+2.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N7a --output /srv/dsv41-build-n7a
cd /srv/dsv41-build-n7a
RECIPE_IMAGE=ds41f:experiment-n7a bash build/build.sh
```

## N7b

**Change:** Same threshold 256.

**Reason to compare / limitation:** Even earlier partitioning; not a monotonic improvement over 512 or baseline.

**Observed versus B0:** Hermes C1 63.9 tok/s (+2.6%); Hermes C8 155.2 tok/s (-2.4%); 64K prefill 1285.7 tok/s (-2.5%); Owen code C8 201.2 tok/s (+4.3%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N7b --output /srv/dsv41-build-n7b
cd /srv/dsv41-build-n7b
RECIPE_IMAGE=ds41f:experiment-n7b bash build/build.sh
```

## N8a

**Change:** B12X MXFP8 A16 small rows 1–6, split-K 2.

**Reason to compare / limitation:** Different small dense projection work split; larger rows unchanged, mixed results.

**Observed versus B0:** Hermes C1 65.9 tok/s (+5.8%); Hermes C8 157.0 tok/s (-1.3%); 64K prefill 1280.0 tok/s (-2.9%); Owen code C8 195.1 tok/s (+1.1%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N8a --output /srv/dsv41-build-n8a
cd /srv/dsv41-build-n8a
RECIPE_IMAGE=ds41f:experiment-n8a bash build/build.sh
```

## N8b

**Change:** Same eligible rows with split-K 1.

**Reason to compare / limitation:** Avoid reduction/split overhead; near-baseline C8, weaker C1/C3.

**Observed versus B0:** Hermes C1 60.6 tok/s (-2.7%); Hermes C8 159.6 tok/s (+0.4%); 64K prefill 1351.1 tok/s (+2.5%); Owen code C8 192.8 tok/s (-0.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm N8b --output /srv/dsv41-build-n8b
cd /srv/dsv41-build-n8b
RECIPE_IMAGE=ds41f:experiment-n8b bash build/build.sh
```

## C1

**Change:** N1 + N2.

**Reason to compare / limitation:** Single-stream specialization; C1 did not beat B0.

**Observed versus B0:** Hermes C1 59.2 tok/s (-5.0%); Hermes C8 148.6 tok/s (-6.5%); 64K prefill 1400.0 tok/s (+6.2%); Owen code C8 202.1 tok/s (+4.8%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C1 --output /srv/dsv41-build-c1
cd /srv/dsv41-build-c1
RECIPE_IMAGE=ds41f:experiment-c1 bash build/build.sh
```

## C2

**Change:** N1 + N2 + N3.

**Reason to compare / limitation:** Add draft overhead reduction; gains do not simply add.

**Observed versus B0:** Hermes C1 61.0 tok/s (-2.0%); Hermes C8 155.9 tok/s (-2.0%); 64K prefill 1342.1 tok/s (+1.8%); Owen code C8 195.9 tok/s (+1.6%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C2 --output /srv/dsv41-build-c2
cd /srv/dsv41-build-c2
RECIPE_IMAGE=ds41f:experiment-c2 bash build/build.sh
```

## C3

**Change:** N1 + N2 + N3 + N4.

**Reason to compare / limitation:** Strongest Hermes C1 in this campaign, without the replay prefill gain.

**Observed versus B0:** Hermes C1 68.1 tok/s (+9.3%); Hermes C8 149.1 tok/s (-6.2%); 64K prefill 1279.6 tok/s (-2.9%); Owen code C8 190.9 tok/s (-1.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C3 --output /srv/dsv41-build-c3
cd /srv/dsv41-build-c3
RECIPE_IMAGE=ds41f:experiment-c3 bash build/build.sh
```

## C4

**Change:** N1 + N2 + N3 + N4 + N6.

**Reason to compare / limitation:** Selected long-context compromise: large prefill gain and completed 128K phase; not highest Hermes C8.

**Observed versus B0:** Hermes C1 63.6 tok/s (+2.1%); Hermes C8 146.1 tok/s (-8.1%); 64K prefill 1734.1 tok/s (+31.6%); Owen code C8 194.5 tok/s (+0.8%).

**Full bounded 128K-C3 phase:** completed.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C4 --output /srv/dsv41-build-c4
cd /srv/dsv41-build-c4
RECIPE_IMAGE=ds41f:experiment-c4 bash build/build.sh
```

## C5-indexer

**Change:** C3 + N5.

**Reason to compare / limitation:** Highest Hermes C8 in this campaign; lower C3 and no replay prefill gain.

**Observed versus B0:** Hermes C1 66.7 tok/s (+7.0%); Hermes C8 162.1 tok/s (+2.0%); 64K prefill 1284.9 tok/s (-2.5%); Owen code C8 194.0 tok/s (+0.6%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C5-indexer --output /srv/dsv41-build-c5-indexer
cd /srv/dsv41-build-c5-indexer
RECIPE_IMAGE=ds41f:experiment-c5-indexer bash build/build.sh
```

## C5-sp512

**Change:** C3 + N7a.

**Reason to compare / limitation:** Alternative row partitioning; inspect complete suite rather than one cell.

**Observed versus B0:** Hermes C1 60.7 tok/s (-2.6%); Hermes C8 153.8 tok/s (-3.3%); 64K prefill 1402.8 tok/s (+6.4%); Owen code C8 198.6 tok/s (+3.0%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C5-sp512 --output /srv/dsv41-build-c5-sp512
cd /srv/dsv41-build-c5-sp512
RECIPE_IMAGE=ds41f:experiment-c5-sp512 bash build/build.sh
```

## C5-sp256

**Change:** C3 + N7b.

**Reason to compare / limitation:** Strong C3, reasonable C8; temporary end-of-test deployment, later replaced by selected C4.

**Observed versus B0:** Hermes C1 65.4 tok/s (+5.1%); Hermes C8 158.9 tok/s (-0.1%); 64K prefill 1347.6 tok/s (+2.2%); Owen code C8 201.3 tok/s (+4.4%).

**Full bounded 128K-C3 phase:** incomplete; do not interpret a cold-prime value as a completed phase.

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C5-sp256 --output /srv/dsv41-build-c5-sp256
cd /srv/dsv41-build-c5-sp256
RECIPE_IMAGE=ds41f:experiment-c5-sp256 bash build/build.sh
```
