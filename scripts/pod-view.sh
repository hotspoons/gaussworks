#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve a trained splat in gsplat's viser viewer. Usage:
#   scripts/pod-view.sh <ckpt.pt> [port]
# Then from your workstation:
#   kubectl port-forward -n <ns> <pod> 8080:8080   ->   http://localhost:8080
set -euo pipefail

CKPT=${1:?usage: pod-view.sh <ckpt.pt> [port]}
PORT=${2:-8080}
GSPLAT_EXAMPLES=${GSPLAT_EXAMPLES:-/opt/gsplat/examples}
[ -d "$GSPLAT_EXAMPLES" ] || GSPLAT_EXAMPLES=/workspace/opt/gsplat/examples

# shellcheck disable=SC1091
source "$(dirname "$0")/gsplat-env.sh"   # same build env as pod-train.sh: no cache rebuild

exec python "$GSPLAT_EXAMPLES/simple_viewer.py" --ckpt "$CKPT" --port "$PORT"
