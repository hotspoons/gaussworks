<!-- SPDX-License-Identifier: Apache-2.0 -->
# Trail previews: the second target

gaussworks has two consumers, and only one of them has been exercised.

1. **Drivable stages** from car-mounted capture — Assetto Corsa and friends.
   This is what every measurement in the repo so far comes from.
2. **Trail previews for trailworks** — cover a trail with a 360 camera, on
   foot or on a bike, and let someone fly it in the browser before they drive
   to the trailhead. Not yet attempted.

Hiking and mountain biking are **one product, two capture configs**. Same
trails, same viewer, same rail, same corridor, same explorer hook, and the same
close-scene regime that drives everything about sampling. A hiker's and a
rider's capture of one trail are two *renderings of the same trail*, which is
already the shape of the picker (see the UX section). Practically, a bike
covers three to five times the ground per session, so it is how coverage
actually gets built — central Maryland's fall-line trails and the coastal-plain
network around Bacon Ridge are ridden and hiked by the same people.

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

## What changes for foot and bike capture

### Sampling: the one number that has to hold

Frame spacing exists to preserve overlap between consecutive views. How much a
view changes per metre depends on **how far away the scene is**, and how many
candidates you get to pick the sharpest from depends on **speed against extract
rate**. Those two facts set every mode's numbers, and the target is the same
everywhere: three to five candidates per selection window.

| Mode | Speed | `extract_fps` | Candidate spacing | `spacing_m` | Candidates/window |
| --- | --- | --- | --- | --- | --- |
| Hike | 1.4 m/s | 8 | 0.17 m | 0.60 m | 3.4 |
| Bike | 5.0 m/s | 30 | 0.17 m | 0.75 m | 4.5 |
| Drive | 10 m/s | 24 | 0.42 m | 1.25 m | 3.0 |

Note the inversion. Hiking uses a *lower* extract rate than driving, biking a
*higher* one — and `spacing_m` for both trail modes is far tighter than the
road, because trail foliage sits 1–3 m out against a road scene's 5–30 m. Get
this wrong in the obvious direction ("it's slower, sample less") and
`spacing_m` silently stops engaging, which is a trap the road config already
documents.

Sharpness selection matters most on a bike, because most frames are blurred:
vibration and speed together make motion blur the dominant quality limit.

### Mount

**On foot**, a backpack pole beats a chest mount, for the same reason the roof
mount works on the car: it puts the camera **above** the operator, so the rigid
occluder is a narrow cone straight down instead of a torso filling the lower
third of every frame. `mask.py` finds the occluder by temporal median and fills
below a per-column boundary — a head and shoulders directly beneath the camera
is well inside what that handles; a chest mount looking past your own body is
not.

**On a bike**, prefer a bar, stem or chest mount over a helmet — and the reason
is not the one you would guess.

Structure-from-motion does not mind a rotating camera. It solves a pose per
image, and a rider looking through a corner actually *improves* coverage of the
geometry that matters most. With a 360 camera it costs nothing that "forward"
stops being yaw 0, because we sample the whole sphere anyway, and the corridor
is built from GPS positions, which carry no orientation.

What a helmet breaks is narrower and more annoying:

- **The static occluder mask.** `mask.py` assumes the occluder is rigid in the
  camera frame. A helmet rotates relative to the shoulders and pack beneath it,
  so they sweep across the frame instead of sitting still, and the temporal
  median stops finding them. A bar or chest mount restores the assumption. If
  you must ride a helmet cam, plan on a hand-drawn generous lower-cone mask
  (`masks/<cam>/_mask.png`, which the code already supports as the escape
  hatch).
- **View aiming.** A fixed `pitch:` in the view plan points wherever the head
  happens to be pointing. See below — the fix already exists in the telemetry.

**A bonus worth taking**: a bar mount sits ~1.0 m up, a helmet ~1.8 m. Two
riders, or one rider on two laps, gives a **vertical baseline** — the exact
thing the driving case cannot get from a single roof mount, and the reason road
crown stays ambiguous there. On a trail, that comes nearly free.

The lens-seam work carries over unchanged: same camera, same profile, same
per-lens view planning (see [SEAM.md](SEAM.md)). Nothing about it is
vehicle-specific.

**Stabilization must be OFF.** HyperSmooth and equivalents rotate the image
relative to the sensor, which breaks the fixed relationship between the lens
boundary and the frame that the whole per-lens view plan depends on. This is
the easiest rule to break by accident, and it gets easier the rougher the ride:
walking bounce tempts you, singletrack chatter tempts you far more.

### Level the views against gravity (proposed, not implemented)

A helmet pitches with the rider's head; a pack pole sways; a car pitches on
hills and leans on camber. In every case a fixed `pitch:` in the view plan
aims somewhere other than intended.

The fix is already sitting in the telemetry we read. GoPro's GPMF carries
`GravityVector`, alongside `CameraOrientation`, `ImageOrientation`,
`Accelerometer` and `Gyroscope` — all confirmed present in the MAX 2 `.360`
files (one document group per payload, i.e. at least 1 Hz, with sub-samples
inside each). `ingest.extract_telemetry` already parses GPMF document groups;
pulling gravity out is the same shape of work as pulling GPS out.

With a per-frame gravity vector, `viewplan` can rotate the view basis so pitch
is measured **against the world, not the camera body**. That would:

- make a helmet cam behave like a fixed mount for aiming purposes;
- stabilise the down-pitched ring that keeps trail tread and road surface in
  frame, on hills and camber as well as on the flat;
- and put every virtual camera in a consistent world-relative orientation,
  which is a better starting point for the rig work described in
  [HANDOFF.md](HANDOFF.md).

This is the single highest-leverage unimplemented feature for the trail target,
and it is not trail-specific.

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

- **mode** — hiked or ridden. Not cosmetic: a rider wants line choice, features
  and sight lines through corners; a hiker wants footing, junctions and
  outlook. Same trail, different questions, and the capture heights differ too.
- **season** — leaf-on vs leaf-off is a different trail, and trailworks already
  models leaf-off as a separate dataset
- **direction** — an out-and-back looks different each way, and the pipeline
  already keeps both passes (`meta.json` records `passes`)
- **date** — blowdowns, reroutes, washouts

So a rendering is keyed by *(trail, mode, date, direction, season)*, and the picker
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
- `configs/trail-hike.yaml`, Tier 1 scale from [HANDOFF.md](HANDOFF.md) §2.4
- check first: does `splatpipe verify` find GPS at all under canopy, and what
  fraction of frames register

The registration rate on that single chunk answers the only question that
matters — whether the trail case needs the along-track chunking change before
it can work at all.

Then ride the same path with `configs/trail-bike.yaml` and a bar mount. Same
trail, two modes, deliberately: it tests the sampling table above, gives the
first real look at motion blur at speed, and produces the two-height pair that
the vertical-baseline note is about — all on ground you already have a walking
reconstruction of to compare against.
