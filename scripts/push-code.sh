#!/usr/bin/env bash
# Put THIS checkout's gaussworks onto the cluster PVC, and record which commit
# it is, so a pod can refuse to run stale code.
#
#   scripts/push-code.sh [pod]     # default pod: splats-head
#
# Why this exists. /workspace/gaussworks on the PVC is a COPY, not a clone --
# there is no .git, and the CUDA images do not ship git anyway, so `git pull`
# in a pod fails and the pod then runs whatever the last person left there.
# That is not hypothetical: a control bake for the cell_m experiment ran
# against the wrong config until a guard caught it, and the guard only existed
# because the config was new enough to grep for. A config EDIT would have
# sailed straight through and produced a plausible, meaningless number.
#
# The real fix is the container image (.github/workflows/image.yml), which
# carries the code and makes the PVC data-only. Until every job runs from that
# image, this keeps the copy honest: it writes /workspace/gaussworks/VERSION
# holding the commit SHA, and `assert_code` below lets a job demand one.
set -euo pipefail
POD=${1:-splats-head}
cd "$(dirname "$0")/.."

sha=$(git rev-parse HEAD)
if ! git diff --quiet || ! git diff --cached --quiet; then
    # Pushing a dirty tree is fine -- it is how you test before committing --
    # but the SHA would then be a lie, so say so in the VERSION file itself.
    sha="$sha-dirty"
    echo "[push-code] WARNING: working tree is dirty; VERSION will say $sha"
fi

echo "[push-code] $sha -> $POD:/workspace/gaussworks"
# git ls-files, not a recursive copy: .gitignore'd build output and __pycache__
# from a different arch have both been shipped to the cluster by accident.
files=$(git ls-files splatpipe configs scripts deploy)
for f in $files; do
    kubectl exec "$POD" -- mkdir -p "/workspace/gaussworks/$(dirname "$f")"
    kubectl cp "$f" "$POD:/workspace/gaussworks/$f"
done
printf '%s\n' "$sha" | kubectl exec -i "$POD" -- tee /workspace/gaussworks/VERSION >/dev/null
echo "[push-code] $(printf '%s\n' "$files" | wc -l) files; VERSION=$sha"
