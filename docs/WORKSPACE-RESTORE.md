<!-- SPDX-License-Identifier: Apache-2.0 -->
# Restoring the gaussworks workspace after a platform reinstall

Everything that matters lives on one CephFS volume mounted at `/workspace`.
The pod, its home directory, apt state and `~/.bashrc` are disposable. This
page is the procedure for getting a new pod back onto that volume — and the
fallback if the volume is ever lost.

Written 2026-08-26, the day the platform was slated for a breaking reinstall.

## 1. The volume

| | |
| --- | --- |
| PersistentVolume | `pvc-14a11d18-e7b2-41c4-8dc9-07cc6532e80f` |
| Reclaim policy | **Retain** (was `Delete` — a platform teardown would have GC'd the PVC and let the CSI driver delete the subvolume) |
| Labels | `patapsco.ai/keep=true`, `patapsco.ai/original-zipspace=zt-ws-hermes-ws-tkcylf` |
| Annotations | original claim and pod recorded |
| CephFS subvolume | `csi-vol-dfe0f3db-a1f9-4df0-a28e-24b27d98dce4` in `test_cephfs` (for a hand-built static PV if the object is lost) |
| Original PVC name | `default/paio-camberfs-default-zt-ws-hermes-ws-tkcylf` (operator-generated from the zipspace name) |
| Saved PV manifest | `/workspaces/pv-hermes-workspace-backup.yaml` on the admin machine, **not** in this repo or on the pod |
| Size / usage | 1000 Gi, ~65 GB used |

What is on it (2026-08-26):

| Path | Size | What |
| --- | --- | --- |
| `opt/{cuda,sfm,exiftool,gsplat,colmap-vocab}` | 8.1 GB | CUDA 13.0.2 toolkit, COLMAP 3.11.1 + GLOMAP 1.0.0 (+ bundled `.so`s), exiftool 13.44, gsplat 1.6.0 source, vocab tree |
| `venv/` | 5.5 GB | Python 3.12 venv with torch 2.9.1+cu130, gsplat, splatpipe (editable) |
| `.torch_extensions/`, `.torch-home/`, `.uv-cache/` | | gsplat JIT kernels (sm_80), LPIPS weights, wheel cache |
| `apt-cache/archives/*.deb` | 33 MB | every apt package the pipeline needs, for offline bootstrap |
| `downloads/` | 4.1 GB | CUDA runfile, exiftool tarball |
| `data/raw/` → `data/arrowhead_farms_and_patuxent_and_gosheff_and_more/` | 31 GB | the three `.360` chapters — **the only thing that cannot be regenerated** |
| `data/street{A,B,B13,B13c,B13v}` | ~16 GB | ingests, chunks, `colmap.db`s, models, checkpoints, drive videos |
| `gaussworks/`, `trailworks/` | | repos (both fully pushed) |
| `env.sh`, `.gitconfig`, `.git-credentials`, `bin/claude-sync.sh`, `.claude-backup/` | | shell env, git identity, Claude Code state (synced every 5 min) |
| `logs/` | | every job log and `*.done` marker from the build-out |

## 2. Reattach after the reinstall (the normal path)

After a teardown the PV goes to `Released` with a stale `claimRef`.

**Option A — pre-bind now, zero steps later.** Set the PV's `claimRef` to the
name the operator will generate, with no uid:

```yaml
spec:
  claimRef:
    namespace: default
    name: paio-camberfs-default-zt-ws-hermes-ws-tkcylf
```

Recreate the workspace **under the same zipspace name** and the operator's new
PVC binds to this PV instead of provisioning an empty one. This is the
recommended route: it removes the manual step and the window in which a new
empty volume could be mistaken for the real one.

**Option B — clear the binding at restore time.**

```bash
kubectl get pv -l patapsco.ai/keep=true
kubectl patch pv pvc-14a11d18-e7b2-41c4-8dc9-07cc6532e80f -p '{"spec":{"claimRef":null}}'
# then create the zipspace; its PVC must match size/class/access mode (1000Gi, cephfs, RWO)
```

Either way, once the pod is up:

```bash
# inside the new pod
ls /workspace/gaussworks /workspace/opt/cuda/bin/nvcc /workspace/data/raw   # sanity: it is the right volume
bash /workspace/gaussworks/scripts/pod-bootstrap.sh                          # ~7 s, offline-capable
source /workspace/env.sh
splatpipe profiles                                                          # gate 4
splatpipe verify /workspace/data/raw/GS010002.360 --out /tmp/v --at 90 --hwaccel cuda   # gate 6
```

`pod-bootstrap.sh` installs ffmpeg and the build tools from the PVC deb cache,
re-links the editable install, hooks `/workspace/env.sh` into `~/.bashrc`, and
starts `claude-sync.sh` (restores `~/.claude` from `/workspace/.claude-backup`
when the home copy is empty). It was run end-to-end on the live pod on
2026-08-26.

### Pod spec: carry these into the new manifest

`deploy/devpod.yaml` already has the first two; the current pod had neither.

- `/dev/shm` as a `Memory` emptyDir (the pod had 64 MB — gsplat's loader dies; see HANDOFF trap 12)
- a container memory limit (27 Gi on a 32 GB node; this pod had none on a 62 GB node)
- init: `bash /workspace/bin/claude-sync.sh start` after the volume is mounted
- env: `TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions`, `TORCH_HOME=/workspace/.torch-home`, `PIP_CACHE_DIR=/workspace/.pip-cache` (all also set by `env.sh`)

### Jobs that were running at teardown time

Anything in flight dies and is restartable; nothing is lost that was expensive:

- `poses`: every chunk keeps a complete `colmap.db`, so a rerun skips feature
  extraction and matching and goes straight to mapping (`splatpipe poses …`
  prints "reusing features and matches").
- `train`: no resume — rerun the same command; checkpoints at 7k/30k survive.
  Clear a stale `.queue/train/<chunk>.failed` or `.done` if a run was killed
  mid-way (HANDOFF §1.6 has the per-run scripts under `/workspace/logs/*.sh`).

## 3. If the volume is gone (fallback)

1. New pod with an empty PVC. Base image needs only `python3`, `uv`, `gcc`, `sudo`, network.
2. `git clone https://github.com/hotspoons/gaussworks /workspace/gaussworks`
   (token needed — it is not in the repo).
3. `bash /workspace/gaussworks/scripts/pod-build-stack.sh` — 60–90 min:
   CUDA runfile (4 GB), venv + torch, exiftool, COLMAP, GLOMAP, gsplat +
   JIT kernels, deb cache, bundled libs. **Assembled from exactly what was
   run on 2026-08-26 but never yet executed end-to-end on an empty volume**;
   expect to babysit the first run. `CUDA_ARCHS` is read from `nvidia-smi`.
4. Re-upload the three `.360` files (31 GB) to `/workspace/data/raw/`
   (HANDOFF §2.2 has the per-file streaming recipe; `.LRV`/`.THM` not needed).
5. Re-derive the street: HANDOFF §2.4 Tier 1 — ingest ~5 min, poses ~30 min,
   train ~80 min on an A100 to get back to the 21.2 dB driveway render.
6. Recreate `/workspace/.gitconfig` and `/workspace/.git-credentials` (HANDOFF §2.1).

## 3a. Walked for real, on a different cluster (2026-09-26)

The fallback above stopped being hypothetical. The volume is **not on
`gh200-1`** — no PV with `patapsco.ai/keep=true`, and
`pvc-14a11d18-e7b2-41c4-8dc9-07cc6532e80f` does not exist there — so the
Arrowhead Farms capture was rebuilt from step 1. What the first real run of
§3 cost, and what it needed that the script did not have:

**The fleet is aarch64.** `gh200-1` is 8 × GH200 480GB (Grace+Hopper, sm_90,
72 cores and ~600 GB of RAM per node). Every measurement in this repo before
now came from an x86_64 A100 pod, and three things assumed that:

| assumed | reality on Grace | fixed by |
| --- | --- | --- |
| CUDA runfile `..._linux.run` | ARM build is `..._linux_sbsa.run`; the x86 name 404s | runfile name from `uname -m`; and if the image already ships a matching toolkit, adopt it (`nvidia/cuda:13.0.2-devel-ubuntu24.04` is multi-arch) |
| `pycolmap` from PyPI | **no aarch64 wheels at any version** | build from `$S/colmap/pycolmap`, which also fixes reader/writer version skew on x86 |
| Python headers present | `python3-dev` absent; CMake reports it as "could not find Python" | added to `PKGS` |

**The build used to kill itself on line one of a fresh volume.** `env.sh`
ended with `[ -f venv/bin/activate ] && source ...`; with no venv that returns
1, so *sourcing* `env.sh` returned 1, and `pod-build-stack.sh` runs under
`set -e`. The log simply stopped after the last line that printed, which looks
exactly like a job still running. Fixed in `scripts/env.sh`; it now always
returns 0.

**Shape of the pod.** For batch work a plain `Pod` mounting the PVC is easier
to drive than a `ZipspaceDeployment` — no operator involved, and `kubectl
exec` is the whole interface. The pieces from §2 that still matter are
`/dev/shm` as a Memory emptyDir and a container memory limit.

One trap the limit introduces: `pod-build-stack.sh` sizes `CXX_JOBS` from
`MemAvailable`, which inside a container reports the **node's** memory (600 GB
here), not the cgroup limit. Unset, it picks ~72 parallel C++ jobs against a
128 GiB ceiling and the container OOMs. Pass `CXX_JOBS` explicitly
(`CXX_JOBS=32 MAX_JOBS=4` was comfortable).

**RWX, not RWO.** `deploy/devpod.yaml` asks for ReadWriteOnce because it was
written for one interactive pod. Stages 3 and 4 pull from a claim-based queue
on shared storage, so putting all eight GPUs on one world needs
ReadWriteMany — `ceph-filesystem` provides it:

```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: splats-work, namespace: default, labels: {patapsco.ai/keep: "true"}}
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ceph-filesystem
  resources: {requests: {storage: 1Ti}}
YAML
```

**Re-uploading 31 GB** (§3 step 4) ran at 14–23 MB/s through
`kubectl exec -i <pod> -- bash -c 'cat > dst' < src`, so ~25 min for the three
chapters. Verify by size **and** md5 per file; a truncated 11 GB chapter
surfaces an hour later as "fewer GPS samples than expected".

## 4. Not this repo's call

The platform operator's crashloop and whether to delete its `OperatorConfig`
are unrelated to this volume; nothing here depends on the operator beyond it
creating the PVC that binds to the PV above.
