#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'Build on an ARM64 DGX Spark with the CUDA 13 toolchain in the pinned image.' >&2
  exit 1
fi
# Source builds are preparation work. Build on an idle node before deployment.
docker build -f build/base/Dockerfile -t dsv41-x11c-base:local build/base
docker build -f build/Dockerfile -t dsv41-x11c:recipe-20261009 .
docker image inspect dsv41-x11c:recipe-20261009 --format '{{.Id}}'
