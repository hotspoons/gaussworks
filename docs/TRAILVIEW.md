<!-- SPDX-License-Identifier: Apache-2.0 -->
# Trail previews: the second target

gaussworks has two consumers, and only one of them has been exercised.

1. **Drivable stages** from car-mounted capture — Assetto Corsa and friends.
   This is what every measurement in the repo so far comes from.
2. **Trail previews for trailworks** — walk a trail with a 360 camera on a
   backpack pole or chest mount, and let someone fly the trail in the browser
   before they drive to the trailhead. Not yet attempted.

The pipeline is the same pipeline. What differs is capture geometry, sampling
density, and where the GPS goes wrong — plus a viewer that does not exist yet.
This note is the design work done up front so the first walking capture is not
wasted.

## Why the existing pieces already fit

The corridor machinery was built for the driving case and is not specific to
it. `corridor.py` emits the polyline the camera actually travelled, plus a
radius and height band; `guardrail.py` clamps any requested camera position
onto it. Its own docstring already says *"a web viewer, Unity, or a Python
notebook can all use it"*.

For a trail that is an unusually clean fit, because **the rail is the trail**.
trailworks already holds the trail geometry; the capture corridor is the same
line, recovered independently from the capture itself. A railed flythrough is
`corridor.json` plus a lookahead — not a new abstraction.

## What changes for walking capture

### Mount

A backpack pole beats a chest mount, for the same reason the roof mount works
on the car: it puts the camera **above** the operator, so the rigid occluder is
a narrow cone straight down instead of a torso filling the lower third of every
frame. `mask.py` finds the occluder by temporal median and fills below a
per-column boundary — a head and shoulders directly beneath the camera is well
inside what that handles; a chest mount looking past your own body is not.

The lens-seam work carries over unchanged: same camera, same profile, same
per-lens view planning (see [SEAM.md](SEAM.md)). Nothing about it is
vehicle-specific.

**Stabilization must be OFF.** HyperSmooth and equivalents rotate the image
relative to the sensor, which breaks the fixed relationship between the lens
boundary and the frame that the whole per-lens view plan depends on. This
matters more on foot than in a car, because walking bounce is exactly what
makes people want to turn stabilization on.

### Sampling density scales with scene distance, not speed

The instinct is to sample less often because walking is slower. That is
backwards.

Frame spacing has to keep enough overlap between consecutive views for feature
matching, and how much a view changes per metre travelled depends on **how far
away the scene is**. A road capture looks at surfaces 5–30 m out; a trail looks
at foliage 1–3 m out. The same metre of travel moves the trail scene several
times as much.

So `spacing_m` goes *down* for trails (0.5–0.75 m against 1.25 m on road), even
though the camera is moving at a seventh of the speed. `extract_fps` can come
down instead, since at 1.4 m/s even 8 fps puts candidates 0.18 m apart — still
several to pick the sharpest from per window. See `configs/trail.yaml`.

### GPS under canopy is the real risk

This is the part most likely to sink a first attempt. Tree cover degrades a
consumer GPS fix from sub-metre to 5–20 m, and three things in this pipeline
lean on position:

- **spatial matching** uses GPS priors to choose which image pairs to compare
- **locality chunking** puts frames in grid cells by position
- **`model_aligner`** fits the reconstruction to the GPS track

All three degrade together, and the failure looks like the connectivity
collapse in [LANDSCAPE.md](LANDSCAPE.md) — a chain graph, one local component,
a fraction of the images registered.

Mitigations, in order of confidence:

1. **Sequential matching already carries this.** `poses.py` runs spatial *and*
   sequential; image names are `camK/NNNNNN.jpg`, so name order is capture
   order per camera. On a linear trail that is a strong graph on its own, and
   it needs no GPS at all.
2. **Chunk along-track, not by grid.** A trail is one-dimensional. Grid cells
   with bad positions scatter a contiguous walk across cells; splitting the
   frame sequence by along-track distance does not care about absolute
   accuracy. `chunks.py` currently only does the grid — this is the change to
   make.
3. **Raise `--alignment_max_error`** and expect metres, not centimetres. The
   *shape* of the track survives canopy far better than absolute position, so
   aligning on shape rather than per-frame position is the principled fix.
4. GPS9 carries per-sample fix quality. Using it to weight or drop samples is
   available and unexplored.

### Corridor dimensions

A trail corridor is a few metres wide, not a road's 25. `configs/trail.yaml`
sets `radius_m: 6`. Too generous and floater pruning stops working; too tight
and the rail clips scenery the walker actually saw.

## The viewer

Does not exist. The pieces it needs — `merge.py` tiles plus `world.json`,
`corridor.json`, `route.json` — do.

**Renderer candidates**, all MIT and therefore fine to bundle in an Apache-2.0
project (confirm before shipping):

| Library | Notes |
| --- | --- |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | The established three.js implementation. Octree culling, WASM SIMD sort, partial GPU sort. Introduced `.ksplat`, a streaming-friendly packing. |
| [Spark](https://github.com/sparkjsdev/spark) | Newer WebGL2 renderer for three.js, built around Niantic's compressed `.spz`. Reads `.ply`, `.splat`, `.ksplat`, `.spz`, `.sog`. Mobile-first. |
| [PlayCanvas / SuperSplat](https://developer.playcanvas.com/user-manual/gaussian-splatting/) | Engine plus an editor that doubles as a strong viewer; `.sog` compression. Not three.js. |

For a trail app the delivery constraint is **phones on cellular at a
trailhead**, which points at whichever compressed format wins on size —
`.spz`/`.sog` rather than raw `.ply`. `merge.py` writes `.ply` tiles today, so
a conversion step is needed either way; picking the format is really picking
the renderer.

**The rail.** `guardrail.clamp` is already the primitive: project the requested
position onto the corridor, pull it back past a threshold. A trail preview
probably wants something stricter than the driving case — closer to an
on-rails dolly with limited free look, since the point is *"what does this
trail look like"*, not *"explore"*. The user's own note from the driving
viewer applies double here: unleashed free-flight in corridor-shaped capture is
disorienting and reads as broken.

## The open UX question

How does someone discover that a trail has a flythrough, and choose between
several?

Several is the normal case, not the edge case. A trail is worth capturing more
than once, and the axes that matter are real:

- **season** — leaf-on vs leaf-off is a different trail, and trailworks already
  models leaf-off as a separate dataset
- **direction** — an out-and-back looks different each way, and the pipeline
  already keeps both passes (`meta.json` records `passes`)
- **date** — blowdowns, reroutes, washouts

So a rendering is keyed by *(trail, date, direction, season)*, and the picker
is closer to a version list than a single "3D" button. The explorer already has
per-trail pages with attachments and revisions
(`pipeline/serving/explorer_api.py`), which is the natural place to hang both
the badge and the list.

Unresolved, and worth deciding before building anything: whether a rendering is
an attachment on an explorer page, or a first-class entity that pages link to.
The second is probably right if renderings are ever shared between pages (a
trail that appears in several areas), but the first is much less work.

## Suggested first walking capture

Small and boring on purpose, to shake out mount, masking and canopy GPS before
committing to a real trail:

- 200–300 m of a wooded path, out and back, backpack pole, stabilization off,
  shutter locked fast
- `configs/trail.yaml`, Tier 1 scale from [HANDOFF.md](HANDOFF.md) §2.4
- check first: does `splatpipe verify` find GPS at all under canopy, and what
  fraction of frames register

The registration rate on that single chunk answers the only question that
matters — whether the walking case needs the along-track chunking change before
it can work at all.
