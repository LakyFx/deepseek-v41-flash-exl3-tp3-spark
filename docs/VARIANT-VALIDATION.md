# Variant verification status

[Choose a recipe](../README.md#choose-your-variant) · [Per-arm commands](VARIANTS.md) · [Original campaign](CAMPAIGN-20261012.md)

## Four independent evidence levels

| Level | What it establishes | Current scope |
|---|---|---|
| Historical GPU campaign | Original loaded variant ran bounded quality checks and benchmarks | All 18 arms; only N6/C4 completed the full bounded 128K-C3 phase |
| Public source reconstruction | Archived source hashes, environment, compilation and generated launch arguments match each arm | All 18 checked by the validator; committed source receipt below |
| Fresh public image build + CPU ABI | Public build finishes; compiled extensions load without a GPU | Not yet established for this 18-arm archive; older X11c receipts do not certify these images |
| Fresh public GPU qualification | New image loads correctly on three nodes and passes numerical/cache/vision tests | Not performed for these rebuilt images; required before accepting them on a new cluster |

The historical C4 deployment is our production DS41F 1.0.0. An experimental image named C4 is still a fresh build, not automatically the same binary. A source check must never be presented as a GPU or performance pass.

The committed [source receipt](../results/variant-validation/source-check.json) records the checked archive revision and individual results. The prepared GitHub Actions template will publish a receipt per tested commit once activated; it is not currently installed. The publication token cannot create workflows. Build and ABI results are `not_run` until actually executed. This page is not an assertion that all public image builds have succeeded.

## Automatic source checks

The [Variant source reconstruction workflow template](../tools/ci/variant-sources.yml) is ready for pushes and pull requests. To activate it, a maintainer with workflow write permission must place it at `.github/workflows/variant-sources.yml` and commit it. Until then, run the automatic checker below locally. It executes repository tests, checks generated README/catalog tables against the stored numerical results, and reconstructs all 18 arms outside the checkout. It verifies source hashes (including pre-N4 native restoration), settings and the generated launch arguments for all three ranks. It does not call Docker, SSH or an inference endpoint.

Run the same check locally, with tags available:

```bash
git fetch origin --tags
python3 -m unittest discover -s tests -v
python3 experiments/render_catalog.py --check
python3 experiments/validate_variants.py --output /srv/dsv41-source-check-001
```

`receipt.json` is updated after every arm. A failure is preserved and produces a nonzero exit code; subsequent arms are still checked. Contexts are retained. An existing output directory is refused rather than overwritten.

## Progressive image builds

Use a dedicated **Linux ARM64 Docker build host**, not an active inference node. Build prerequisites and retained base download are described in [INSTALL](INSTALL.md). The base download is approximately 10.8 GB and requires at least 40 GB additional headroom before accumulating variant images. Check available resources; source-only checks do not reserve build resources for you.

Start with the most useful alternatives:

```bash
python3 experiments/validate_variants.py \
  --arms B0 C4 C3 C5-indexer N3 N6 \
  --build --output /srv/dsv41-build-check-001
```

Then check the remaining arms:

```bash
python3 experiments/validate_variants.py \
  --arms N1 N2 N4 N5 N7a N7b N8a N8b C1 C2 C5-sp512 C5-sp256 \
  --build --output /srv/dsv41-build-check-002
```

Or omit `--arms` to build all 18 sequentially. The validator creates tagged images, performs a CPU-only ABI check with no GPU/model mounts and no container networking, and retains build/ABI logs, image IDs, container IDs and per-arm status. It never deploys, modifies services, loads weights, benchmarks, upgrades drivers or cleans up images/containers. Run batches from the same clean pinned commit. Failure logs may contain local paths; review before sharing.

After a successful image/ABI build, follow [operator acceptance](VERIFICATION.md#operator-acceptance) on your own cluster. Publish fresh qualification and benchmark receipts separately, including host, image ID, checkpoint and measurement scope. Do not overwrite the original campaign numbers with a new workload's scores.

## Updating the tables

```bash
python3 experiments/render_catalog.py
python3 experiments/render_catalog.py --check
```

Both README tables and per-arm cards are generated from the stored full-precision summary and campaign descriptions. Keep the two benchmark definitions separate. The source archive includes 19 plans but only 18 measured arms; N9 is deliberately refused.
