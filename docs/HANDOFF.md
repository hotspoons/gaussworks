<!-- SPDX-License-Identifier: Apache-2.0 -->
# Handoff: standing up gaussworks in a fresh dev pod

Written for **a coding agent starting from a raw `/workspace`**, plus a section
for the human on what to copy up. Everything here has been executed on a real
pod; where a number appears, it was measured, not estimated.

Read [LANDSCAPE.md](LANDSCAPE.md) before changing anything. It is the list of
things that have already cost hours, and most of them are not guessable.

---

## Part 1 — For the agent

### 1.1 What this is

`gaussworks` turns 360 video into a chunked, geo-aligned gaussian splat world.
**Two targets, one pipeline:**

- **drivable racing-game stages** from car-mounted capture of Maryland back
  roads — every measurement in this repo comes from this path;
- **trail previews for trailworks** — the same pipeline walked or ridden
  rather than driven, ending in a railed browser flythrough of a trail. Hiking
  and mountain biking are one product with two capture configs. Not yet
  attempted. Capture deltas, the canopy-GPS risk most likely to sink a first
  attempt, renderer options and the open UX question are in
  [TRAILVIEW.md](TRAILVIEW.md); `configs/trail-hike.yaml` and
  `configs/trail-bike.yaml` are the starting points.

Five stages, each a plain CLI over files on disk:

```
ingest → chunk → poses → train → merge     (then: mesh / export / drive / eval)
```

Stages 3 and 4 pull from a claim-based queue on shared storage, so N workers on
N nodes can be pointed at one chunk directory. Architecture and the reasoning
behind each stage are in [../README.md](../README.md); the camera/lens design is
in [SEAM.md](SEAM.md).

### 1.2 Preconditions

- A GPU dev pod, `/workspace` backed by a persistent volume (see
  [../deploy/devpod.yaml](../deploy/devpod.yaml); the manifest carries a
  **27 Gi memory limit** — do not remove it, see §1.7).
- Any Ubuntu 24.04 image with `python3`, `uv`, `gcc` and `sudo`. **Nothing else
  is assumed any more** — the current pod (2026-08-26) shipped with no torch,
  no CUDA toolkit and no ffmpeg, so the build script installs all three onto
  the PVC. 16 cores / 62 GB / A100-PCIE-40GB / no cgroup memory limit.

### 1.3 Provision

```bash
git clone https://github.com/hotspoons/gaussworks /workspace/gaussworks
cd /workspace/gaussworks
bash scripts/pod-build-stack.sh          # ~60-90 min, idempotent; arch from nvidia-smi
```

`CUDA_ARCHS` defaults to the installed GPU's compute capability (`80` A100,
`89` L40S, `90` GH200/H200, `120` RTX Pro Blackwell). **Build only the one you
have**; see §1.7. The CUDA toolkit major (`CUDA_RUNFILE`, 13.0.x) must match
the torch wheel index (`TORCH_INDEX`, cu130).

That script installs, all onto the PVC so they survive container restarts:

| Component | Version | Where | Why this version |
| --- | --- | --- | --- |
| CUDA toolkit | 13.0.2 (runfile, `--toolkitpath`) | `/workspace/opt/cuda` | nvcc for COLMAP + gsplat; the image has none |
| torch | 2.9.1+cu130 | `/workspace/venv` | uv-made venv, no system torch to inherit |
| COLMAP | 3.11.1, CUDA | `/workspace/opt/sfm` | GPU SIFT |
| GLOMAP | **1.0.0**, CUDA | `/workspace/opt/sfm` | global SfM; 1.2.0 is broken here — trap 7 |
| bundled `.so`s | from `ldd` | `/workspace/opt/sfm/lib/bundled` | COLMAP/GLOMAP run after a restart with **no apt** |
| ExifTool | 13.44 (GitHub tag mirror) | `/workspace/opt/exiftool` | **GPS9 support, mandatory** |
| gsplat | 1.6.0 (+examples) | `/workspace/opt/gsplat` | Apache-2.0 trainer |
| `.deb` cache | everything apt installed | `/workspace/apt-cache` | offline bootstrap |
| splatpipe | editable | `/workspace/gaussworks` | this repo |

After a **container restart** (not a fresh pod), the PVC survives; the home
dir, apt state and `~/.bashrc` do not. Recovery is
`bash scripts/pod-bootstrap.sh` — seconds, offline-capable: it installs ffmpeg
and the build tools from the `.deb` cache, re-links the editable install, hooks
`/workspace/env.sh` into `~/.bashrc`, and starts the Claude Code state sync
(§1.5).

### 1.4 Verification gates

Do not proceed past a failing gate. Each one corresponds to a bug that was
silent and expensive.

```bash
source /workspace/venv/bin/activate
export PATH=/workspace/opt/sfm/bin:/workspace/opt/exiftool:$PATH
export EXIFTOOL=/workspace/opt/exiftool/exiftool

# 1. ExifTool understands GoPro GPS9. MUST be > 0.
exiftool -listx | grep -c GPS9

# 2. GLOMAP present AND able to finish. `glomap -h` is NOT sufficient --
#    see trap 8: it can complete an entire reconstruction and abort writing it.
#    The only honest gate is a real mapping on the smallest chunk you have.
which glomap && glomap -h | head -3        # expect "compiled with CUDA!"

# 3. COLMAP has CUDA.
colmap -h | sed -n 2p                      # expect "... with CUDA)"

# 4. Camera profiles load.
splatpipe profiles                         # expect 7: gopro-max, gopro-max2,
                                           # gopro-360-generic, equirect-360,
                                           # insta360-x3, insta360-x4, pinhole

# 5. gsplat imports and its CUDA extension is cached, not rebuilt.
#    RUN THIS ONLY WHEN THE POD IS OTHERWISE IDLE -- see trap 6.
python -c "import gsplat; print(gsplat.__version__)"

# 6. End-to-end on real footage — the single most useful check.
splatpipe verify /workspace/data/raw/GS010002.360 --out /tmp/v --at 90 --hwaccel cuda
```

Gate 6 should print `gopro-max2 (gopro_eac, validated)`, `each lens sees
94.74 deg from its axis`, `GPS: 4477 samples`, and a six-view plan at
`yaw -50.0, 0, 50.0 / 130.0, 180, 230.0`. **Look at the JPEGs it writes.** Each
`cam*.jpg` comes from a single lens, so a visible seam inside one means the
profile geometry is wrong.

### 1.5 The environment contract

**`/workspace/env.sh`** (source: `scripts/env.sh`) is the one file every shell
and every job sources. It sets `CUDA_HOME`, `PATH` (cuda, sfm, exiftool),
`LD_LIBRARY_PATH` (cuda + bundled libs), `EXIFTOOL`, `GSPLAT_EXAMPLES`, the
uv/pip/torch-extension caches, `GIT_CONFIG_GLOBAL=/workspace/.gitconfig`, then
sources `gsplat-env.sh` and activates the venv. `~/.bashrc` gets one line that
sources it. **Non-interactive shells (`bash -lc`, `nohup`, cron) skip
`.bashrc`** — start every script with `source /workspace/env.sh`.

Two more things live on the PVC because the home dir does not:

- `/workspace/.gitconfig` + `/workspace/.git-credentials` (identity, PAT store).
- `~/.claude` — Claude Code's memory, sessions and settings. `scripts/claude-sync.sh
  start` (called by bootstrap; add it to the pod init script too) restores it
  from `/workspace/.claude-backup` when the home copy is empty, then rsyncs it
  back every 5 min. It refuses to sync an empty home over a full backup.

`gsplat-env.sh` remains the **single source of truth** for the JIT build
environment, and every entry point must source it. torch hashes the build
config into the cached extension, so launching the trainer and the viewer with
different flags rebuilds every fused rasterizer kernel from scratch (~30 min
each).

Two things that must be on PATH before any pipeline run:

- **`glomap`** — `poses.py` calls `shutil.which("glomap")` per chunk and falls
  back to COLMAP's incremental mapper without it. Measured on one 3,870-image
  chunk: incremental mapper **4h53m** and CPU-only, against **73 min** for GPU
  feature extraction plus matching combined. It logs the fallback, so read the
  log rather than assuming.
- **a GPS9-capable `exiftool`** — see §1.7.

### 1.6 Current state of the work

Everything below is committed and pushed to `origin/main`
(github.com/hotspoons/gaussworks).

**Settled.** One lens per training image. Views are planned inside each
physical lens' cone (`splatpipe/viewplan.py`) instead of on a fixed yaw ring,
so no image spans two optical centres. Full reasoning, prior art, and what
GoPro's D.WARP actually does: [SEAM.md](SEAM.md). Evidence rendered from real
footage — do not re-litigate this without reading it first.

**Settled.** Hardware lives in data, not code: `splatpipe/data/profiles/*.yaml`
(match rules, lens axes, telemetry, defaults) plus `splatpipe/drivers/` (one
class per projection). New camera with a known projection is a YAML file.
`SPLATPIPE_PROFILES=/path` adds a directory without forking.

**Measured** — one 150 m chunk, 942 images, 30k steps, `splatpipe eval` on
visible pixels only:

| Config | PSNR (visible) | Gaussians |
| --- | --- | --- |
| no mask, no AA, no regularisers | 22.80 dB | 278,025 |
| mask + AA + opacity/scale reg | 21.87 dB | 171,280 |
| **mask + AA, no regularisers** | **23.07 dB** | **293,014** |

23.07 dB is **the number to beat**. Never compare runs using the trainer's own
PSNR when masks differ — gsplat zeroes masked pixels in the render but not the
ground truth, worth ~4.5 dB here.

**In flight at handoff** (2026-08-25 16:40 UTC). `/workspace/data/street2`, the
first full run through the per-lens pipeline: 916 frames × 6 views = 5,496
images, 4 chunks, both drive directions present in every cell.

The COLMAP incremental mapper reached **3,868 of 3,870 images registered
(99.95%)** on `chunk_x0_y-1` — proof that the spatial-matching fix in trap 3
works — and then the container restarted (trap 6) before it wrote its model,
losing that mapping. Its `colmap.db` survived with 63,400 verified pairs, so
`poses` was relaunched and will reuse it. All four chunks now map with GLOMAP.

**Training has not started, so there is no quality number for the per-lens
change yet.** That is the first thing to finish: `splatpipe status --chunks
/workspace/data/street2/chunks`, then `splatpipe eval` against 23.07 dB, then
`splatpipe drive` for a flythrough.

**Next, and worth doing properly: declare the rig.** The GLOMAP failure above
points at something real. Our six virtual cameras are not six independent
cameras — they are one rigid rig whose relative orientations we know
*exactly*, because we synthesised them from the view plan (yaw/pitch per
camera, one shared optical centre per lens, and a known ~3 cm offset between
the two lenses). COLMAP 3.12+ models this natively via `rigs` / `rig_sensors`
with a `sensor_from_rig` transform per camera. Populating it would:

- satisfy the check GLOMAP aborts on, and
- collapse 6 poses per capture position into **1 pose plus 6 fixed offsets**,
  which is a large reduction in free parameters and removes the whole class of
  "the six views drifted relative to each other" error.

`splatpipe/writer.py` already records each camera's `yaw`, `pitch` and `lens`
in `cameras.json`, so the transforms are a short computation away. This needs
COLMAP 3.12+ across the toolchain.

**Open — the trail target.** Nothing has been walked yet. The highest-value
unknown is whether canopy GPS (5–20 m under tree cover, against sub-metre in
the open) breaks locality chunking badly enough to require chunking by
along-track distance instead of by grid. One 200–300 m wooded out-and-back at
Tier 1 scale answers it. See [TRAILVIEW.md](TRAILVIEW.md).

**Open.** Whether per-lens planning beats 23.07 dB. Road crown needs a second,
lower physical camera — no software fix exists, every lens is at one roof
height. Mesh stage is implemented but unvalidated on driving data. The shipped
`insta360-x3` / `insta360-x4` profiles are marked `untested`: their
`geometry.circles` came from published specs, and `splatpipe verify` against a
real `.insv` is how they get fixed.

### 1.7 Traps

The full list with measurements is in [LANDSCAPE.md](LANDSCAPE.md). The five
that will bite a fresh pod:

1. **`TORCH_CUDA_ARCH_LIST`.** NVIDIA containers export every architecture
   (`7.5 8.0 8.6 9.0 10.0 12.0+PTX`). gsplat's JIT then builds seven variants
   of every kernel; `cicc` peaks 9–20 GB, and this OOM-killed the pod three
   times and **took the node down** (required a Proxmox VM reset).
   `gsplat-env.sh` overrides it from `nvidia-smi`. Keep the container memory
   limit so the kubelet survives a bad burst.
2. **ExifTool version.** GoPro moved GPS telemetry from GPS5 to GPS9 with the
   HERO11 generation; the MAX 2 writes GPS9. Ubuntu 24.04 ships 12.76, which
   parses the file, reports all 21,770 lines of every *other* stream — gyro,
   accel, magnetometer, per-lens `Geometry Calibrations` — and returns **zero
   GPS**. Downstream that is indistinguishable from a camera with GPS off:
   geo alignment and locality chunking silently disable, and `--spacing-m`
   degrades to "keep every frame". Same files: 12.76 → 0 samples, 13.44 →
   **4,477 at 10 Hz**. `ingest` now refuses to run rather than continue.
3. **Spatial matching must be sized to the rig.** Every virtual view of one
   capture position carries that position's GPS, and every pass revisits it, so
   `n_cams × n_passes` images sit at one coordinate (18 here). COLMAP's default
   32 neighbours is then ±1 position of road: the graph becomes a chain and the
   mapper builds one local component and stops — **600/1986 and 1843/4320
   registered**. `poses.py` now scales it. Diagnostic worth memorising: all
   cameras registering at the *same* rate means it is **not** a lens/geometry
   problem; contiguous registered positions mean connectivity; scattered means
   image quality.
4. **`pkill -f <pattern>` inside `kubectl exec` kills your own shell** — the
   exec's command line contains the pattern. Looks like a mystery exit 137/143;
   hit four times. Same trap for `pgrep -f -c` as a liveness probe: it counts
   the probe. Use exact-name matching (`pkill -x colmap`, `pgrep -c -x colmap`)
   or signal a PID; for completion, have the job `touch` a marker file.
5. **Do not run a torch import while a heavy job is going.** The container has
   a hard 27 Gi limit (deliberately — see trap 1). COLMAP's mapper sits at
   ~7.8 GB, and `import gsplat` pulls in torch and initialises CUDA for
   several more. Doing both at once **restarts the container**: `kubectl exec`
   returns exit 137, the pod's `restartCount` increments, and every running job
   dies. This happened while verifying gate 5 against a mapper at 99.95%
   registration, and cost five hours of mapping. Check `pgrep -x colmap` and
   `free -g` first, or run verification on an idle pod. What made the recovery
   cheap rather than catastrophic was database reuse (`poses.py` keeps a
   complete `colmap.db`), so only the mapping was lost, not the 73 min of GPU
   feature work.
6. **A container restart takes `~/.git-credentials` with it** — the default
   `credential.helper store` writes to the container filesystem. Point it at
   the PVC, as in §2.1, or the first `git pull` after a restart fails with
   "could not read Username".
7. **GLOMAP 1.2.0 aborts writing the model when the database came from COLMAP
   3.11.** GLOMAP 1.2.0 vendors a COLMAP from Oct 2025 that models a
   multi-camera setup as a *rig*. Opening a 3.11.1 database migrates in empty
   `rigs` / `rig_sensors` / `frames` tables, and GLOMAP then dies at the very
   last step — after a complete, successful reconstruction (1,129,132 tracks,
   global positioning, BA, track filtering all fine):

   ```
   Check failed: existing_rig.RefSensorId() == rig.RefSensorId()
   terminate called after throwing an instance of 'std::invalid_argument'
   ```

   All four street chunks, ~2 h of global SfM each, discarded at the write.
   Keep the COLMAP generation consistent across the toolchain, or use GLOMAP
   1.0.0 (predates the rig model) — now the default in
   `pod-build-stack.sh`, which also passes `-DGUI_ENABLED=OFF`, because 1.0.0's
   vendored COLMAP demands Qt5 otherwise. `--mapper auto|glomap|colmap` makes
   the choice explicit rather than "whatever is on PATH". COLMAP 3.11.1 maps a
   GLOMAP-migrated database without complaint, so the fallback is safe:
   verified at **954/954 images, 0.698 px** on `chunk_x0_y0`.
8. **After a container restart, build dependencies are gone too.**
   `pod-bootstrap.sh` restores the *runtime* libraries COLMAP links against,
   not the `-dev` packages. The next `cmake` fails on a missing Eigen3 or
   Ceres config. Re-run the apt block from `pod-build-stack.sh` first.
9. **`exiftool.org` only serves the newest release.** The 13.44 tarball the
   scripts pinned returned 404 within weeks. Fetch tags from
   `github.com/exiftool/exiftool/archive/refs/tags/<ver>.tar.gz` instead; any
   12.90+ works.
10. **EAC face edges drew a black line** until 2026-08-26. `eac_maps` produced
   pixel-edge coordinates and `cv2.remap` wants pixel centres, so the last
   half-row of every cube face was bilinearly mixed with the black border —
   a dashed 1 px dark line at yaw 135°/225° in every rear view (cam3/cam5), a
   fixed-column feature SIFT matches across frames. `splatpipe verify` output
   is how it was caught: measure column gradients at the predicted face edge
   (`960 + f·tan(Δyaw)`), not just eyeball. Fixed by a half-pixel shift plus
   clamping inside the face; anything ingested before commit `37f6708`
   carries the line.
11. **Trap 4 again, from the other side.** A waiter loop written as
   `until pgrep -f gps-fence.py …` never exits, because the loop's own command
   line contains the pattern. Wait on marker files or output content.
12. **`/dev/shm` is 64 MB on this pod, and read-only to remount.** gsplat's
   trainer hands 33 MB float images between 4 loader workers through shared
   memory and died at step 0 ("DataLoader worker exited unexpectedly"), twice,
   and the queue marked the chunk failed. `splatpipe/gsplat_masked.py` now
   drops to in-process loading when shm is under 1 GB (~10-15% slower).
   The right fix is in the pod spec: a `Memory` emptyDir on `/dev/shm`, as
   `deploy/devpod.yaml` already declares. Check with `df -h /dev/shm` before
   the first training run on any new pod.
13. **GPS needs a minute.** The street recording started in a driveway under
   canopy with an unconverged fix: 15-19 m horizontal and 10 m vertical error
   for the first ~80 positions, ~1 m after that, while SfM stayed at 0.75 px.
   The alignment residual (9.7 m mean) was entirely this. `poses` now rebuilds
   `corridor.json` from the solved cameras, so drive/merge/export use the
   truth; but chunking and spatial-matching priors still come from GPS, so
   power the camera on and let it settle in the open before recording.
14. **Python buffers stdout under `nohup`.** A log that is 0 bytes for 30 minutes
   is usually buffering, and COLMAP writes straight to the fd — so unflushed
   Python lines land long after the subprocess output they label. This made a
   correctly-applied fix look like it had never run. `flush=True` everywhere.

### 1.8 Working style that has held up

- Run pipelines and jobs **on the pod, never on a laptop**. Edit locally or in
  the remote VS Code session, push, pull on the pod, run there.
- `setsid nohup … &` with output to a log, plus a `touch`ed marker file on
  success. Poll the marker, not the process table.
- Before diagnosing a regression, check whether the thing you changed is even
  implicated. Twice here the obvious suspect was innocent.
- Measure one change at a time. A three-change run that lost 1.2 dB took a
  second experiment to attribute.

---

## Part 2 — For the human: what to copy up

### 2.1 Credentials

**GitHub.** A fine-grained PAT scoped to `gaussworks` (and `trailworks` if you
want both). On the pod:

```bash
printf '%s' 'ghp_YOUR_TOKEN' > /workspace/.gh-token
chmod 600 /workspace/.gh-token            # the old pod had this at 644

git config --global user.name  "Rich Siomporas"
git config --global user.email "richard.siomporas@patapsco.ai"
# keep the credential store on the PVC: the default ~/.git-credentials is on
# the container filesystem and vanishes on every restart
git config --global credential.helper "store --file=/workspace/.git-credentials"
printf 'https://hotspoons:%s@github.com\n' "$(cat /workspace/.gh-token)" \
  > /workspace/.git-credentials
chmod 600 /workspace/.git-credentials
```

Nothing above belongs in the repo. `/workspace` is outside the checkout, which
is deliberate.

**Harbor.** Only needed for `just image-push`. Mint a robot account yourself —
`docker login harbor.tools.basedweights.com`. No agent has ever held these.

**GitLab.** There may be an orphaned `devpod-pull` project access token on the
GitLab mirror of gaussworks (read_repository, 7-day expiry, created
2026-08-23). Delete it if it still exists.

### 2.2 Data

Total on the old pod is 48 GB, but **only `data/raw` is worth moving** — 30 GB
of `.360` captures. Everything else is derived and reproducible:

| Path | Size | Copy up? |
| --- | --- | --- |
| `data/raw/GS0{1,2,3}0002.360` | 30 GB | **yes** — the source captures |
| `data/street2` | 9.8 GB | no — re-derive (ingest is ~2 h) |
| `data/hoodhq`, `data/hood` | 8.7 GB | optional; `hoodhq` holds the 23.07 dB baseline checkpoints |
| `data/verify`, `data/seamdemo` | 18 MB | no — regenerate in a minute |

If you want to keep the **23.07 dB baseline** for comparison without re-running
it, copy `data/hoodhq/chunks/*/ckpts` and `data/hoodhq/chunks/*/sparse` — a few
hundred MB rather than 6.4 GB.

**How to move 30 GB.** `kubectl cp` is unreliable at this size (no resume, one
broken pipe loses the file). Stream per file instead, so a failure costs one
file:

```bash
POD=<new-pod-name>            # kubectl get pods -o name | grep gaussworks
kubectl exec "$POD" -c dev -- mkdir -p /workspace/data/raw

for f in GS010002.360 GS020002.360 GS030002.360; do
  echo "== $f"
  kubectl exec -i "$POD" -c dev -- \
    bash -c "cat > /workspace/data/raw/$f" < "/path/on/mac/$f"
done

# verify by size, then by checksum
kubectl exec "$POD" -c dev -- ls -l /workspace/data/raw/
kubectl exec "$POD" -c dev -- bash -c 'cd /workspace/data/raw && md5sum *.360'
md5sum /path/on/mac/*.360
```

Pulling them straight off the old pod is faster if it is still up — cephfs to
cephfs beats a round trip through your laptop:

```bash
OLD=gaussworks-devpod-zipspace-c58f8978d-v992v
for f in GS010002.360 GS020002.360 GS030002.360; do
  kubectl exec "$OLD" -c dev -- cat "/workspace/data/raw/$f" \
    | kubectl exec -i "$POD" -c dev -- bash -c "cat > /workspace/data/raw/$f"
done
```

The `.LRV` and `.THM` sidecars the camera writes are not needed — `.LRV` is a
low-res proxy, `.THM` a thumbnail. Nothing in the pipeline reads them.

### 2.3 VS Code

`deploy/devpod.yaml` already carries the remote-dev annotations and the
`zip-friends-init` container that provides the connect binaries. Apply it,
wait for the pod, then connect with the platform's VS Code plugin. Two notes:

- The CRD group is user-scoped for multitenancy —
  `richard-siomporas.patapsco.ai/v1alpha1` on your cluster, not
  `ai.patapsco.ai`. `deploy/zipspace.yaml` is the main-platform variant.
- Check the manifest applies before trusting it: `kubectl apply -f
  deploy/devpod.yaml --dry-run=server`. That caught an invented `memory:` field
  once (the real one is `resources.limits.memory`).

### 2.4 First run in the new pod — start small

**Scale is the biggest cost lever in this pipeline, and it is easy to get
wrong.** COLMAP's incremental mapper is superlinear in image count. Measured on
the street capture, same settings, same GPU:

| Chunk | Images | Incremental mapping |
| --- | --- | --- |
| `chunk_x0_y0` | 954 | **22 min** |
| `chunk_x0_y-1` | 3,870 | **4h53m** (and it died before writing the model) |

Four times the images, thirteen times the time. A four-chunk run over 300 m of
road is a 10–14 hour job; a one-chunk run over 150 m is under an hour
end to end. **Almost every question worth asking is answerable at one chunk.**
The 23.07 dB baseline was measured on 942 images in a single chunk.

So work in tiers, and only go up a tier when the tier below has told you
something:

**Tier 0 — is the rig decoded correctly? (~1 min)**
```bash
splatpipe verify /workspace/data/raw/GS010002.360 --out /tmp/v --at 90 --hwaccel cuda
```

**Tier 1 — the quality loop (~45 min end to end).** One chunk, ~150 capture
positions, ~950 images. This is where you answer "did that change help?".
```bash
splatpipe --config configs/hq.yaml ingest /workspace/data/raw/GS010002.360 \
  --out /workspace/data/t1 --near 38.983984,-76.695795 --radius-m 60 \
  --spacing-m 1.25 --segment-s 45 --hwaccel cuda
splatpipe --config configs/hq.yaml mask  --frames /workspace/data/t1
splatpipe --config configs/hq.yaml chunk --frames /workspace/data/t1 --cell-m 0   # 0 = one chunk
splatpipe poses --chunks /workspace/data/t1/chunks --matcher spatial --mapper colmap
splatpipe train --chunks /workspace/data/t1/chunks --steps 30000
splatpipe eval  --chunk /workspace/data/t1/chunks/chunk_x0_y0 \
  /workspace/data/t1/chunks/chunk_x0_y0/ckpts/*.pt      # compare against 23.07 dB
```
Note `--cell-m 0` (single chunk, no halo duplication), one chapter, and
`--radius-m 60`. Those three choices are the difference between 45 minutes and
most of a day.

**Tier 2 — the street (hours).** Three chapters, `--radius-m 150`,
`--spacing-m 1.0`, `cell_m: 150` → 916 positions, 4 chunks, 11,130
image-memberships. Only worth it once Tier 1 says the settings are right. This
is what `/workspace/data/street2` on the old pod is, and it is where the
10–14 hour figure comes from.

**Tier 3 — the neighbourhood.** All 30 GB, no `--near` filter. Overnight on one
GPU at minimum; this is what the multi-node work queue exists for (point N
workers at one chunk directory — see §1.1).

Progress at any tier: `splatpipe status --chunks <dir>`. A flythrough of a
trained chunk: `splatpipe drive --chunk <chunk>`.

### 2.5 State left on the old pod

`/workspace/data/street2` — Tier 2, stopped part-way, deliberately:

| Chunk | Images | State |
| --- | --- | --- |
| `chunk_x0_y0` | 954 | **solved: 954/954 images, 188,389 points, 0.698 px** |
| `chunk_x-1_y0` | 1,986 | database complete (28,121 verified pairs), unmapped |
| `chunk_x0_y-1` | 3,870 | database complete (63,400 verified pairs), unmapped |
| `chunk_x-1_y-1` | 4,320 | database complete (65,766 verified pairs), unmapped |

**No training has run, so the per-lens change still has no PSNR number.**
`chunk_x0_y0` is solved and correctly sized to produce one — `splatpipe train
--chunks .../street2/chunks --only chunk_x0_y0 --steps 30000` then `splatpipe
eval` is roughly 30 minutes if you want it off the old pod before
decommissioning. Otherwise Tier 1 above reproduces it from scratch in the new
pod in about 45 minutes, which is the cleaner option.

Worth copying up if you want the baseline for comparison:
`data/hoodhq/chunks/*/ckpts` and `.../sparse` — that is the **23.07 dB**
reference, a few hundred MB rather than the full 6.4 GB.
