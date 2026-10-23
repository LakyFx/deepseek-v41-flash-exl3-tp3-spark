# Stock SparkDash reproduction

Original collectors: MiaAI-Lab/sparkDash `9e3f01729a4e63dccc2b6b608e9e67d3168891b6`. Vendored files retain exact
upstream bytes and license notices; PROVENANCE.json binds their SHA-256 values.
The portable wrapper only coordinates the managers and checks measured outcomes.
It does not change prompts, warmups, sampling, formulas or collector behavior.

Requires Node.js 24 (measured with 24.13.0). It submits inference requests.
First close public ingress yourself, let old work finish or explicitly cancel it,
and verify the backend is exclusive. It never stops services or reopens your API.

```sh
cd benchmarks/sparkdash
npm ci --ignore-scripts
SPARKDASH_EXCLUSIVE=1 node run.mjs /path/to/NEW/results
```

Defaults: loopback port 8003, model `deepseek-v4.1-flash`, prose and code,
400 tokens per stream, temperature 0, top_p 1, thinking OFF;
C1/2/3/4/5/6/8/10/12; prefill 1K/2K/4K/8K with three samples each.
`SPARKDASH_PORT` and `SPARKDASH_MODEL` select your local endpoint/model alias.
Optional `SPARKDASH_NATIVE_PID` plus `SPARKDASH_BINARY_SHA256` pins your own
native executable identity; a rebuild need not reproduce the historical binary.
Always use a fresh results directory; an existing CAMPAIGN.json is rejected.
The wrapper verifies all streams and closes the HTTP agent on exit. It does
not change model/default serving settings or pretend to authenticate a gateway.

[Published measurement tables](../../docs/SPARKDASH-20261023.md) ·
[raw measured jobs](../../results/20261023-sparkdash/).
