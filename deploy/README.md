<!-- SPDX-License-Identifier: Apache-2.0 -->
# Running a whole capture on a multi-GPU cluster

Written 2026-09-26 from the Arrowhead Farms run on `gh200-1` (8 × GH200 480GB,
**aarch64**). Every number here was measured on that run.

Two manifests:

| file | what |
| --- | --- |
| `cluster-batch.yaml` | the RWX volume + the head pod (serial, decode-bound stages) |
| `workers-job.yaml` | one Job per GPU stage, N identical pods pulling a shared queue |

`devpod.yaml` / `zipspace.yaml` remain the interactive path. This is the batch
path, and it is a plain Pod/Job on purpose: no operator, and `kubectl exec` is
the whole interface.

---

## 0. Count the free GPUs first. This is a shared cluster.

**The mistake to avoid:** grepping the pod list for your own names, seeing
nothing, and concluding the cluster is idle. On `gh200-1` three of the eight
GPUs were held by unrelated services (`recon`, `high-brine-flux2-dev`,
`zt-qwen-qwen38-27b`) whose names contain nothing about splats. A Job that
asks for more GPUs than are free does not fail — the extra pods sit `Pending`
indefinitely, which is indistinguishable from slow progress.

Ask the scheduler, not the names:

```bash
kubectl get pods -A -o json | python3 -c '
import json,sys
from collections import Counter
d=json.load(sys.stdin); c=Counter()
for p in d["items"]:
    if p["status"].get("phase") not in ("Running","Pending"): continue
    for k in p["spec"].get("containers",[]):
        r=k.get("resources",{})
        if r.get("limits",{}).get("nvidia.com/gpu") or r.get("requests",{}).get("nvidia.com/gpu"):
            c[p["spec"].get("nodeName")] += 1
print("GPUs claimed per node:", dict(c))
print("nodes busy:", len(c))'
```

Set `parallelism` **and** `completions` in `workers-job.yaml` to what is
actually free. Remember the head pod holds one while it has a GPU limit.

## 1. Volume and head

```bash
kubectl apply -f deploy/cluster-batch.yaml
```

The volume is **ReadWriteMany** on `ceph-filesystem`. `devpod.yaml` asks for
RWO because it was written for one interactive pod; `poses` and `train` claim
work from a queue *on this volume*, so every worker mounts it at once. RWO
silently caps you at one worker.

## 2. Build the stack onto the volume (first time only, ~35 min)

The head image is bare. Give it what `pod-build-stack.sh` assumes, then run it:

```bash
kubectl exec splats-head -- bash -lc '
  apt-get update -qq && apt-get install -y -qq sudo curl git python3 python3-venv ca-certificates xz-utils
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh'

# copy the repo up (it is gitignored inside apex-conduit; no credentials on the pod)
tar czf - --exclude=.venv --exclude=.git gaussworks | kubectl exec -i splats-head -- tar xzf - -C /workspace

kubectl exec splats-head -- bash -lc \
  'cd /workspace && CXX_JOBS=32 MAX_JOBS=4 nohup setsid bash gaussworks/scripts/pod-build-stack.sh \
     > /workspace/logs/build-stack.log 2>&1 < /dev/null &'
```

**Pass `CXX_JOBS` explicitly.** The script sizes it from `MemAvailable`, which
inside a container reports the **node's** memory (600 GB here), not the cgroup
limit. Left alone it picks ~72 parallel C++ jobs against a 128 GiB ceiling and
the container OOMs. `CXX_JOBS=32 MAX_JOBS=4` was comfortable.

Then the gates in `docs/HANDOFF.md` §1.4. Do not skip them.

## 3. Upload footage

```bash
kubectl exec -i splats-head -- bash -c 'cat > /workspace/data/raw/GS010002.360' < GS010002.360
```

14–23 MB/s measured, so ~25 min for 31 GB. **Verify size and md5 per file.** A
truncated 11 GB chapter surfaces an hour later as "fewer GPS samples than
expected", not as an error.

## 4. Serial stages, on the head

```
ingest → mask → chunk
```

`ingest` is the one stage that does not distribute: it is ffmpeg-decode bound,
not GPU bound, and it wants the head's NVDEC. Within the pod it uses every
core. Across pods it does not — for a much larger capture, run chapters
concurrently rather than threading harder.

## 5. Release the head's GPU, then fan out

The head's GPU is only for NVDEC during ingest. While it holds one, the
workers are one short. Recreate it without the `nvidia.com/gpu` limit (it
keeps the volume, which is where everything lives), then:

```bash
kubectl apply -f deploy/workers-job.yaml   # poses first
kubectl exec splats-head -- bash -lc 'source /workspace/env.sh; splatpipe status --chunks <chunks>'
```

**Both GPU stages fan out**, which is the thing worth verifying rather than
assuming — if only `train` did, COLMAP would be the long pole running one
chunk at a time and the cluster would idle through most of the wall clock:

- `poses` → `WorkQueue(chunks_dir, "poses")` — poses.py
- `train` → `WorkQueue(chunks_dir, "train")` — train.py

Same claim-based queue, `os.mkdir` locks, stale claims reclaimed.

**Chunk count must exceed worker count**, or the extra workers have nothing to
claim. `chunk` prints cells and frames per cell; `cell_m` is the dial. The
Arrowhead capture (11.95 km of road over 1.7 × 2.0 km) gave 30 chunks at
`cell_m: 200`, and 42 at 150.

## 6. Merge, back on the head

```
merge --prune-corridor
```

---

## What a worker actually needs

Far less than the head: no ffmpeg, no build tools. COLMAP and GLOMAP carry
their own shared libs in `opt/sfm/lib/bundled`, and the venv is on the volume.
It does need `python3` in the container, because the venv was built on the
**system** interpreter and its `base_prefix` is `/usr`.

Measured cold start from the bare image to a running `splatpipe`, including
apt: **22 s**.

## aarch64

`gh200-1` is Grace+Hopper, so every image must be multi-arch and three build
assumptions needed fixing (CUDA runfile name, pycolmap wheels, Python
headers). See `docs/WORKSPACE-RESTORE.md` §3a.
