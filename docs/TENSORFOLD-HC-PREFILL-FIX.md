# TF351 short-prefill AOT repair, 14 October 2026

The deployed TF351 TP3 kit is now included in this recipe. The fix adds **six `_hc_mix_pf` Triton AOT variants**: row buckets 4, 5 and 6, each for `WORLD=1, POSTED=0` and `WORLD=3, POSTED=1`. The manifest grows from **536 to 542 variants**. Weights, native serving executables, speculation/cache policy, TP3, C8, round rows 16, MAX reasoning, 401,408 request context and **10M shared KV pool** were unchanged in the deployment.

## Evidence and root cause

Production logs showed requests ending with `MissingTritonVariant`, `_hc_mix_pf`, `rows: i32 4`, while other requests and all rank processes continued. Sixteen distinct failing requests were identified; the examined log evidence did **not** show a CUDA OOM, Xid or an inference-process crash explaining them. Row-5/6 omissions were found by manifest/selector checks, not claimed as independently observed production errors.

A cached long prefix can leave a very small uncached continuation. The old preflight did not cover every small prefill shape, so its pass did not protect that path. Reducing the pool would not provide the missing launch variants. This repair addresses that documented error, not every possible HTTP 502 or every aspect of memory pressure.

## Reproduce the repaired kit

The standard [extractor and installation procedure](TENSORFOLD-INSTALL.md) now contain the repaired **complete kernel archive**, not a metadata-only patch. Every previously shipped kernel launch contract is unchanged. New cubin bytes match those deployed and GPU-checked on SM121. The complete AOT manifest SHA256 is:

```text
d2a74f96ead50f86f7fcaab2905dee651bd6e27d3c831ee5c4fef1f07130b684
```

[Patch inventory and six cubin hashes](../tensorfold/hc-prefill/PATCH.json), [compiler cases](../tensorfold/hc-prefill/COMPILE-CASES.json) and [path-redacted compiler provenance](../tensorfold/hc-prefill/COMPILE-PROVENANCE.json) are included. Provenance retains the original compile-only status; the separate GPU receipt records subsequent numerical qualification. The archive also retains all existing license notices.

The native **CPU checker source** is expanded to cover prefills 1–128 and selected larger boundaries, indexer prefills 1–64 and selected boundaries, and existing decode shapes. Only `zig/src/families/dsv41/aot_check_main.zig` changed in the shipped source snapshot relative to the previous published package; runtime numerical source and the production serving binaries were not changed. Its source provenance is `5fa66fc`. [Inputs inventory](../tensorfold/inputs.json) binds the updated source and kernel archives. This is an overlay on the frozen runtime source pin, not a claim that that older Git commit already contained the checker change.

On a new independent build directory:

```sh
python3 tensorfold/extract.py --out /srv/ds41f-tf-fixed
# Follow TENSORFOLD-INSTALL.md to build and prepare the same pool/weights.
# Optional CUDA-only numerical smoke on an idle SM121 device:
python3 tensorfold/hc-prefill/gpu_check.py /srv/ds41f-tf-fixed/kit
```

The GPU smoke compares the new row-4/5/6 kernels with the established RB8 arithmetic; it does not load model weights. The archived expanded CPU selector test additionally covers 12,158 launch selections. Prepare the complete matching 10M RoPE kit and use the same kit on **all three ranks**. Existing deployments require a coordinated stop/start before switching kit paths. An updated environment file must take precedence over inherited service environment files; verify the actual running process environment, not only the unit's `Environment=` display. The public launcher already constructs its selected kit paths directly from operator configuration.

## Recorded deployed acceptance

| Check | Recorded result |
|---|---|
| Expanded CPU selector, actual repaired manifest | 8,699 base + 3,459 position-view checks; no missing variant |
| GPU numerical check | All six new variants produce bitwise-identical XO/PART to RB8 on the same synthetic inputs |
| Short model inputs | 4/5/6-token prefills completed |
| Concurrent smoke | Eight simultaneously submitted short requests completed; not a throughput C8 benchmark |
| Warm-prefix regression | 12,288-token base followed by tails of 4/5/6; all HTTP 200, cached_tokens=12,288; about 0.115 / 0.127 / 0.140 s |
| Gateway SSE | HTTP 200, valid DONE terminator and answer 4 to 2+2 |
| Logs after final startup | No MissingTritonVariant, reason=error or CUDA OOM in the checked new-process logs |

[Actual CPU receipt](../results/20261014-hc-prefill/ACTUAL-AOT-CHECK.json), [GPU receipt](../results/20261014-hc-prefill/GPU-CHECK.json), [warm-tail HTTP usage/timings](../results/20261014-hc-prefill/WARM-TAIL-SMOKE.json). These are targeted correctness checks, not a new speed or broad quality campaign, and they do not guarantee the absence of future failures under untested loads. The same shared kernel package is provided to TF29; the post-fix model/gateway acceptance above was performed on **TF351**, not a new TF29 deployment.

## Rollback and cache

Keep the prior complete kit, operator commands, service environment and weights until acceptance completes. On rollback, stop all three selected ranks, restore the previous kit/environment consistently, restart them, wait for the model to warm and reopen ingress. Restoring an old incomplete kit restores the documented bug; use that only as a diagnostic rollback.

This build has no supported live KV export/import. The actual restart used the operator-approved cold-cache fallback. **Capacity stayed at 10M; cached contents were not transferred.** Switching engine or shared-pool configuration is not a safe cache import. No driver update, host reboot or pool reduction is needed for this AOT repair.
