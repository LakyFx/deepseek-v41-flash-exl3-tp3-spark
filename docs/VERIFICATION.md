# Installation verification and historical measurements

The historical measurements were made on the running X11c deployment. The public image rebuild is a separate packaging check. Compiling identical source does not establish numerical equivalence on a new machine or reproduce a throughput measurement by itself.

## Retained base

`config/base-image.json` pins the exact ARM64 base image ID, ordered archive parts, sizes and SHA256 identities. The original upstream nightly tag returned 404 when checked on 2026-10-09. The public release mirror preserves the existing base rather than substituting a newer vLLM distribution. The historical base Dockerfile, source pins and patches are also included.

The archive is one continuous gzip stream split into eleven parts, not eleven independent tar archives. `tools/fetch_base.py` checks each part, streams them in order to `docker load`, and checks the resulting image ID. Downloads can resume through HTTP Range and remain local on failure. This helper does not start a container, load weights or change a running service.

## Source checks

CPU checks cover the TP3 Engram bucket ranges against the included model helper, selected launch flags, reasoning mappings at both request fields, published corpus hashes and authored workflow bodies, semantic fixture scoring, and invalid counter coverage. They do not prove GPU kernel numerical equivalence or measure model quality.

```bash
python3 -m unittest discover -s tests -v
```

Public results distinguish complete, incomplete, rejected and observational measurements. Private captured prompt bodies and raw assistant responses are excluded. Historical body hashes and counts are retained as identifiers, not a claim that the new public legacy surrogate prompts are identical.

## Operator acceptance

Run the completed image on all three nodes only after preparing your own topology and local Engram ranges. Confirm startup version, dependency checks, TP rank assignment, graph capture, AOT manifest, KV capacity and reasoning policy. Then run a short owned request and an exclusive benchmark using [BENCHMARKS.md](BENCHMARKS.md). Retain your old deployment until this acceptance succeeds.

The recipe does not perform a fresh 460 GB model download or a second live three-node weight load during publication. It preserves the measured deployment settings and makes the installation inputs explicit. Throughput, memory availability and graph capture can differ on another host; report them as new measurements.
