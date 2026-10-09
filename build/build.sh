#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'Build on an ARM64 DGX Spark with the CUDA 13 toolchain in the pinned image.' >&2
  exit 1
fi
# The original upstream nightly tag is no longer downloadable. Use the audited
# exact base archive instead of silently substituting a different nightly.
# Keep at least 40 GB of additional free disk for the archive and image layers.
python3 tools/fetch_base.py --load
docker build -f build/Dockerfile -t dsv41-x11c:recipe-20261009 .
docker image inspect dsv41-x11c:recipe-20261009 --format '{{.Id}}'
