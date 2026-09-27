#!/usr/bin/env bash
# Refuse to run unless the PVC's gaussworks is the commit this job expects.
#
#   assert-code.sh <sha>        # exact, or a prefix
#
# A job that silently runs the wrong code does not fail -- it SUCCEEDS, and
# hands you a number you cannot tell apart from a real one. That is strictly
# worse than crashing, so this exits non-zero rather than warning.
set -euo pipefail
want=${1:?usage: assert-code.sh <sha>}
have=$(cat /workspace/gaussworks/VERSION 2>/dev/null || echo "(no VERSION file)")
case "$have" in
    "$want"*) echo "[assert-code] ok: $have" ;;
    *) echo "[assert-code] FATAL: PVC holds '$have', this job needs '$want'." >&2
       echo "[assert-code] run scripts/push-code.sh from a checkout, then retry." >&2
       exit 1 ;;
esac
