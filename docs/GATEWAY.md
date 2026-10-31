# Production gateway 1.2.6 and durable vision jobs

**TP4 4.0.0:** use the same package and retained private state with [the TP4 environment example](../tensorfold/tp4/gateway.env.example). Set `DS41F_GATEWAY_PROFILE` to the installed [TP4 JSON profile](../tensorfold/tp4/gateway-profile.json) for C16 and 8M reserved-token admission, routing text/vision to the four-rank head. Native KV is 9M. Plain capacity environment variables do not override the portable entrypoint's JSON. The existing TP3 settings remain unchanged. [Complete TP4 installation](DS41F-4.0.0.md).

## Current 1.2.6 defaults

Install `./gateway` and run `ds41f-gateway`; supply your own authentication secret
and upstream. All 31 runtime modules match production, except the two documented
bridge upstream defaults adapted to loopback. The portable package adds no keys,
private topology, TLS material, user images, reasoning traces or runtime database.

Standalone image jobs use **MAX**, **8,192 thinking tokens**, **32,768 final-answer
tokens** (**40,960 total**). The entrypoint enables these defaults, two concurrent
image jobs, finite deadlines and native termination control. Other agent tasks
with image histories/tools retain their existing routing and requested budgets.

Transport uses `tensorfold_disconnect`: it never sends vLLM-specific abort calls
to TensorFold. Accepted image jobs are not automatically regenerated. Native
termination/instance-replacement receipts govern uncertain slot reconciliation.

Private diagnostics are opt-in: use a private, service-owned directory plus an
expiring exact job-ID allowlist. Under `ProtectSystem=strict`, use the existing
`StateDirectory` writable mount or add only the exact external `ReadWritePaths`.
Never publish these files. Numeric values are retained by the repetition matcher;
the old rare rejection was not reproduced in nine successful new image jobs.

[Current recipe and boot guidance](DS41F-1.2.6.md) · [source hashes](../gateway/source-manifest.json).

## Preserved historical gateway documentation

## 1.2.5 production parity

The package now includes the exact current production modules, including native
vision cancellation/control, finite dispatch deadlines and reconciliation.
The two bridge defaults use loopback instead of private addresses; operator
secrets, topology and runtime SQLite state are never included. Production
module hashes were freshly matched before publication.

`ds41f-gateway` enables `DGX_VISION_CONTROL_ENABLED=1` and requires the configured
native `/internal/vision/control` endpoint. It defaults to two image requests,
queue timeout 1,800 s, job timeout 3,590 s and cancellation grace 60 s. Accepted work
outlives client disconnect; uncertain dispatch is never automatically regenerated.
Only a native termination/instance-replacement receipt releases an uncertain slot.

Standalone canonical MEDIUM image requests receive 3,072 reasoning tokens plus
32,768 final-answer tokens (35,840 total), without increasing reasoning effort.
`DGX_VISION_FINAL_ANSWER_TOKENS=32768` controls that final budget. Main agent
requests with tools/history keep their own requested budgets. The legacy vision
alias keeps its separate cap. Read the older 3072/6144 instructions below as
historical behavior, not the new portable entrypoint defaults.

The `laguna_gateway.app` module remains a compatibility/legacy entrypoint;
use **`ds41f-gateway`** for the current production defaults. [Current recipe](DS41F-1.2.7.md).

## Preserved earlier gateway documentation



The current integrated release is [1.2.3](DS41F-1.2.3.md), with the native
adaptive configuration measured as 1.2.2xTF. Gateway policy remains C12 and
two concurrent queued image jobs, text MAX / vision MEDIUM. This package also
includes the already deployed repetitive-output drain fix: the owned backend
stream is drained before validation failure, avoiding a false uncertain-dispatch
block after a repetitive but fully received response. Unknown dispatches still
require explicit reconciliation and are never automatically replayed.

This optional gateway publishes the serving behavior used with the native
TensorFold deployment. Install it in front of the head API. It does not manage
the model, agent, weights or other services. Existing engine recipes and measured
benchmark results retain their original scope.

The package includes the actual production Chat Completions/Responses engine,
context handling, token admission, SSE validation, connection recovery, media
admission and durable vision jobs. Of 29 runtime Python modules, 27 are byte
identical to the reviewed production revision. The two optional ingress bridges
only replace their default private upstream with loopback. The source manifest
records both sets of hashes. The public entry point adds required credentials,
portable configuration and private local state. No operator credentials, stored
conversations, images, runtime databases or TLS material are included.

## Install and start

Use Linux and Python 3.11 or later. From the recipe root:

```sh
python3 -m venv /srv/ds41f-gateway/.venv
/srv/ds41f-gateway/.venv/bin/python -m pip install './gateway[test]'

# Generate your own API secret. Keep it in your private process environment.
export LAGUNA_GATEWAY_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export LAGUNA_GATEWAY_UPSTREAM=http://127.0.0.1:8003
export DS41F_GATEWAY_STATE_DIR=/srv/ds41f-gateway/state
/srv/ds41f-gateway/.venv/bin/ds41f-gateway
```

The backend must already be ready on the operator-selected URL. The listener
defaults to 127.0.0.1:18002. Set `DS41F_GATEWAY_BIND`/`DS41F_GATEWAY_PORT` for
your topology. For a model on another host, supply its actual private URL.
Vision defaults to that same backend, as in production; an optional
`LAGUNA_GATEWAY_VISION_UPSTREAM` selects another compatible backend.

The installed `ds41f-gateway` entry point requires a nonempty API key. The raw
`laguna_gateway.app:app` module retains legacy rollback defaults; use the public
entry point, or `uvicorn ds41f_gateway:create_app --factory --workers 1`, to
load this production profile. Keep one worker: its database lease prevents
multiple vision workers from dispatching the same queue.

`tools/gateway.py` is a compatibility launcher for this package. Its old
1200-second proxy configuration is replaced by the profile below. The retained
`policy()` helper is only for historical recipe tests, not the active policy.

## Same production policy, operator-owned topology

All 72 nonsecret settings are published in
[`production-settings.json`](../gateway/laguna_gateway/production-settings.json).
Tests compare the loaded profile against every field, rather than a few selected
defaults. Upstream URLs, API key and local tokenizer path are separate operator
settings; the production gateway uses backend tokenization, with no local
tokenizer configured.

| Setting | Default |
|---|---|
| API model | `deepseek-v4.1-flash` |
| Main reasoning | MAX by default; explicit supported client overrides retained |
| Physical context |401408 tokens |
| Concurrent requests / waiting queue |12 /64 |
| Standard output / maximum output |32768 /131072 tokens |
| Admission token budget |5336064 |
| Sampling |temperature0.6, top_p0.95, top_k20, presence_penalty0 |
| Chat and Responses checkpoint |enforce, trigger262144, target258048 |
| Image/video token reserve |16384 /32768 |
| Images per prompt |32; overflow mode off |
| Image admission |two image-bearing requests at a time, including agent routes; further accepted jobs queue |
| SSE heartbeat |15 seconds |
| Connection / read / write / pool timeout |10 /3600 /120 /30 seconds |
| Upstream connection recovery window |900 seconds |
| Request body |64MiB |
| Czech output enforcement |on, matching production |

For another output language set `LAGUNA_GATEWAY_ENFORCE_CZECH=false`.
`DGX_MEDIA_MAX_CONCURRENCY=2` is the public entry point and systemd default,
matching production. Set it to `1` for serial processing. Keep
`DGX_MEDIA_SERIAL_REQUESTS=1` to enable this admission gate. These are concurrent
request slots, not a two-image payload limit. Vision and text share C12; two
vision requests leave up to ten slots for text. Native image encoding still
uses its serialized workspace; this release makes no throughput or RAM claim.
For deliberate policy changes copy the JSON to your own untracked file and set
`DS41F_GATEWAY_PROFILE` to its path. Keep all fields; credentials and operator
paths are rejected in that file. Changes are your own profile, rather than the
published production defaults. The historical compatibility field names
`qwen38_vllm`, `xhigh` and `LAGUNA_GATEWAY_*` remain; DeepSeek-specific handling
maps them to the actual native protocol. `xhigh` in the profile means MAX here.

## Vision: submit once, wait for the same job

Use the canonical model with explicit `reasoning_effort: "medium"`,
`thinking_budget:3072`, `max_tokens:6144` and `priority:0` for the qualified
standalone-image request. Supply one user message with a text question and its
original `image_url` data. The public MEDIUM level maps to native `high`.
Image bytes and requested budgets are preserved by the job layer.

The legacy `deepseek-v4.1-flash:vision` alias still has its historical3072 TOTAL
output cap. Increasing a request budget while using that alias does not bypass
its cap. Prefer the canonical model and explicit parameters above.

Ordinary `/v1/chat/completions` clients remain supported. A single-user image
question without assistant history/tools becomes a retained job. Streaming
returns an immediate accepted/waiting SSE comment and regular heartbeats, then
only the validated final answer. Nonstreaming waits for that same final answer.
An SSE comment keeps the transport alive; it does not become an assistant
message or teach an unchanged agent a new waiting protocol. Agent/tool/history
requests retain their existing route.

Explicit job clients can instead:

1. `POST /v1/vision/jobs` with the same Chat Completions body and bearer key.
   HTTP202 is returned only after the SQLite transaction commits.
2. Retain `job_id` and `status_url`. Query that URL with the same credential.
   States are `queued`, `running`, `succeeded`, `failed`, `unknown`.
3. On success consume `result.data`, the OpenAI completion. On timeout or
   disconnect check the same job; do not resize/change the question to bypass it.

`Idempotency-Key` or `X-Client-Request-Id` binds an explicit ID to its original
input. Otherwise credential+canonical-body matching reuses the retained job
for up to 30 minutes. Stream options do not create another inference. Changing
the key, body or credential is not a safe retry strategy after an unknown result.

Only a nonempty final answer with accepted finish status and completed SSE is
success. Reasoning-only, empty, truncated and repetitive outputs never become
an image analysis. No secondary repair generation is launched for these jobs.
Detected repeated incoming assistant `reasoning`/`reasoning_content` produces
HTTP422 before dispatch. This cannot repair an agent's stored conversation or
guarantee that every possible loop is detected.

## Timeouts, disconnect and restart

The total image-job deadline is 900 seconds including queue/preparation. A queued
job can expire before inference under heavy load. For compatibility clients we
recommend a separate 1020-second total deadline and nonstream read timeout. SSE
clients may use 60–90seconds of network inactivity timeout with the 15-second
heartbeat, plus the total deadline. The gateway's3600-second transport read
setting is not the job deadline. Disable blind automatic generation retries;
retain the same job identity instead.

Accepted work survives client disconnection. Queued jobs survive a gateway
restart. A running or ambiguously interrupted dispatch becomes `unknown` and is
never automatically replayed. UNKNOWN blocks new standalone image dispatch
until an operator verifies backend termination. Ordinary text/agent routes
remain available. An already running peer image job may finish, while UNKNOWN
blocks new dispatch into either slot. An HTTP disconnect alone is not termination evidence.

For controlled maintenance wait for active work to finish before restarting the
API. Existing HTTP connections cannot be migrated into a queue across process
restart. Keep the same API credential, database and revision for retained jobs.
Changing the revision invalidates implicit dedup for new submissions.
Release1.2.2TF preserves the previous default job identity revision so upgrading
this compatible scheduler does not invalidate retained implicit jobs. Runtime
source provenance is recorded separately in `source-manifest.json`.

The local operator tool is read-only without a receipt:

```sh
python -m laguna_gateway.vision_jobs_operator \
  --db /srv/ds41f-gateway/state/jobs.sqlite3 --job-id vision_JOB_ID
```

For reconciliation supply `--verified-termination-receipt /path/to/receipt.json`.
It must contain the exact `job_id`, `backend_request_id`,
`termination_confirmed:true` and concrete `evidence` of at least 32 characters.
Reconciliation records an audit row and marks the uncertain job failed; it does
not replay it. This operation is deliberately not a public/model API endpoint.

## State, authentication and optional ingress

The process uses umask 0077 and state directory 0700. The ledger retains original
requests/images and final results. Limits are 64pending jobs, 4096records and
512MiB of payload storage, with 4 MiB per collected result. There is no automatic
deletion. Review retention before the cap is reached; keep UNKNOWN and rollback
evidence. `DGX_VISION_JOBS_DB` overrides the database location.

Authenticated APIs require `Authorization: Bearer <your-secret>`. `/health`,
`/ready` and `/metrics` match production and are intended for the private listener.
Users sharing one gateway credential share its job ownership; use separate
instances if independent client isolation is required. Put a trusted TLS proxy
in front when exposing the listener beyond loopback.

Optional production bridge modules are included. `create_lan_bridge_app` in
`laguna_gateway.lan_bridge` accepts the operator's
`DGX_LAN_CLIENT_KEY_SHA256` allowlist and internal `LAGUNA_GATEWAY_API_KEY`.
`create_bridge_app` in `laguna_gateway.tailscale_bridge` requires a nonempty
`DGX_CODEX_ALLOWED_TAILSCALE_USERS` allowlist and authenticated Tailscale Serve
identity headers. Keep that bridge on loopback behind the trusted identity
proxy; never trust identity headers from arbitrary clients. Both forward only
allowlisted routes, including the exact job/status paths. Configure their
upstream with `DGX_LAN_GATEWAY_UPSTREAM`/`DGX_CODEX_GATEWAY_UPSTREAM`; public
defaults use127.0.0.1:18002. TLS certificates, identities and secrets belong to
the operator, not this repository.

The optional [`systemd template`](../gateway/ds41f-gateway.service) runs the
installed executable under a dynamic user with a private persistent state
directory. Set the API key and backend URL in root-owned 0600
`/etc/ds41f-gateway.env`. Adapt the executable/working directory to your
installation. Installing/enabling the template is an explicit operator step.

## Verification scope

```sh
cd gateway
python -m pytest tests -q
```

The tests use fake upstreams and private temporary databases. They exercise the
actual engine and public entry point, auth, queue/dedup, disconnect, bounded
cancellation/restart recovery, UNKNOWN reconciliation, final-only SSE, alias/main
policy and bridge routes. Source hashes record exact production parity.

Production accepted a real image, completed after the waiting client disconnected
and returned the same cached result through streaming and nonstreaming Chat.
A separate canary retained its completed result through API restart. These are
functional checks, not a new model benchmark or a promise of image latency under
load. The public installation still needs acceptance with the operator's own
backend, network and dependencies; no production services are restarted by this
package or its tests.
