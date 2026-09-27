<!-- SPDX-License-Identifier: Apache-2.0 -->
# Scaling gaussian splats to a road network: a working journal

**What this is.** A running record of what breaks when you take 3D Gaussian Splatting off the
benchmark scenes and point it at *kilometres of public road*. Not a tutorial and not a paper —
a lab notebook with numbers, kept because the failure modes turn out to be more interesting than
the successes, and because almost none of them are visible in the literature's usual setting.

**Why the setting matters.** Published 3DGS work is overwhelmingly *bounded*: an object, a room,
a courtyard, a single street segment, a drone orbit. A road network is a different regime in four
ways that each break something:

1. it is **open** — no enclosing camera ring, every view is a corridor;
2. it is **ordered** — the camera path is one-dimensional, so anything you try to fit to it is
   ill-conditioned across the path;
3. it is **revisited** — the same road is driven more than once, at different times, under
   different light, and those passes must agree;
4. it is **partitioned** — you cannot bundle-adjust 50,000 images at once, so the world is cut
   into pieces that are solved independently and must then be made to agree *with each other*.

Point 4 is where most of the interesting failures live, and it is structurally absent from any
single-scene benchmark.

**Methodological stance.** Every number here was measured on a real capture. Where a check was
introduced, a control was run to confirm the check could fail — several of the entries below exist
*because* an earlier check passed vacuously.

---

## Entry 1 — Arrowhead Farms, 2026-09-26/27

### 1.0 The capture

| | |
|---|---|
| camera | GoPro Max 2, roof rig, 2.32 m ± 0.05 above road (measured, not fitted) |
| footage | 3 chapters, 20.7 min, 31 GB, 2 × 5952×1920 EAC tracks at 204 Mbit/s |
| driven | 11.95 km of road over a 1.72 × 2.02 km box, suburban, heavy canopy |
| ingested | 5,075 capture positions × 6 virtual views = 30,786 images |
| chunks | 29 cells at `cell_m` 200, median 1,818 images per chunk |
| compute | 5 × GH200 (Grace/Hopper, aarch64), ~5.5 GPU-hours total |
| result | 6.18M gaussians, 29 streamable tiles, 1.5 GB PLY |

---

### 1.1 The central finding: registration is not correctness

This is the entry that reframed everything else, and I think it generalises well beyond splats.

The obvious health metric for an SfM-backed pipeline is **how many images registered**. It is what
COLMAP prints, it is what every tutorial checks, and on this capture it was *actively misleading*:

```
chunk_x-3_y0   513/513 positions registered (100%)   internally split by 64 m
chunk_x-3_y1   389/389 positions registered (100%)   internally split by 75 m
```

Every image found a consistent pose *within some component of the model*. The components did not
agree with each other. A global mapper produces exactly one model and will happily return one
containing two copies of the same road at different heights — and every image in it registers
perfectly, because each one is consistent with its own copy.

The driver's experience of this is unmistakable and was how it was found: *"the world oddly
twists and goes at an angle, then flattens back out several hundred feet later."* The twist is the
overlap region of the two copies; the flattening is where one of them ends.

**The general lesson:** a metric that counts *participation* cannot detect *disagreement*. For any
partitioned reconstruction, participation metrics are necessary and nowhere near sufficient.

### 1.2 Free invariants are the best validators

Having no ground truth, the useful question becomes: *what do I know about this capture that the
reconstruction does not?* Three such invariants turned out to be powerful, and all three are free.

**(a) The rig is rigid.** The camera was bolted to the roof, so **camera height above the road is a
constant** for the entire capture. Any per-pass deviation is reconstruction error, measurable
without any external reference:

```
chunk_x3_y-2 (healthy)   +2.6 +2.6 +2.7 +2.7 +2.8     spread 0.2 m
chunk_x0_y-2 (broken)    -4.4 -3.7 +2.6 +3.3 +6.9 +7.3   spread 11.8 m
```

Passes with the camera *below the ground* are not a subtle defect, and no registration count sees
them.

**(b) Chunks overlap by construction.** The locality grid gives every chunk a halo, so wherever two
chunks' camera paths cross the same ground they independently measure their relative height. That
is an **observation network**, and it turns "are my chunks consistent?" into a solvable estimation
problem rather than an opinion (§1.4).

**(c) Roads are where roads are.** OSM centrelines are not ground truth for *height*, but they are
excellent for *plan position*. Measuring every corridor point against the nearest drivable
centreline gave median **2.14 m** — and the correct interpretation is the interesting part: the
centreline is the middle of the road and a car drives in a lane, so **half a lane width is the
right answer**. A median near zero would have indicated something fitted that should not have been.
An invariant that tells you when you are *too* accurate is unusually valuable.

### 1.3 Priors have a *reach*, and the units are not metres

The single most expensive bug of the run, and the most generalisable.

Spatial matching uses GPS priors to decide which image pairs to attempt. Its parameter,
`spatial_radius`, is expressed in **capture positions** — and at 2.35 m spacing, the default 4
positions is about **9 m of reach**. GPS error under continuous canopy on this capture reached
**15–19 m** (per-chunk median DOP up to 19).

When reach is smaller than prior error, a specific and silent thing happens: on the return pass
down the same road, each image's "spatial neighbours" are *the wrong stretch of road*. The two
passes never match each other, the mapper has nothing tying them together, and they float apart —
overwhelmingly in the vertical, which is the least constrained direction for a ground vehicle.

Widening the reach to 12 positions (~28 m):

```
                  cross-pass observations   median Δz   max Δz
radius 4                     56               7.75 m    17.03 m
radius 12                   769               1.41 m     1.79 m
```

and 1626/1626 images in a *single* connected component. Cost: ~4.5× matching time.

**The transferable rule:** *any* prior-gated matcher has a reach, and it must exceed the prior's
error — not its nominal precision, its realised error in the worst part of the capture. Expressing
reach in samples rather than metres hides the comparison, because the conversion depends on
capture speed, which varies within a single drive.

### 1.4 Independent alignment makes a mosaic, not a world

Each chunk is geo-registered independently, fitting a similarity transform to *its own* GPS
priors. GPS **altitude** is the noisiest component of a consumer fix (typically 1.5–2× the
horizontal error), so chunks solved from different parts of a drive are levelled differently:

```
chunk_x-2_y1 <-> chunk_x-3_y1     50.7 m apart
chunk_x-1_y0 <-> chunk_x0_y1      32.9 m
chunk_x-1_y0 <-> chunk_x-1_y-1     0.03 m    <- a good neighbour, same chunk
```

Not one bad chunk: a *group* sitting ~25 m low, and that group was exactly the chunks with median
DOP 18–19. The defect is systematic and correlated with a measurable covariate.

The fix borrows from surveying rather than from vision. The chunk overlaps are observations of
*relative* height; the unknowns are one offset per chunk; the solution is least squares with
outlier rejection. Across the main 26-chunk component:

```
before    median |Δz| 0.34 m, worst 50.7 m
after     residual median 0.02 m, worst 1.31 m
```

Three details that mattered more than the algorithm:

- **Gauge.** The network determines only *relative* heights; adding a constant to every chunk
  satisfies every observation equally. Fixing the datum by forcing zero weighted-mean adjustment
  keeps the world where it is, which matters when a downstream consumer owns global placement.
- **Outlier rejection is not optional.** One bad observation distorts every chunk connected to it.
  Four of the six rejected observations involved a single *internally split* chunk — a chunk that
  disagrees with itself cannot give a trustworthy observation about anything, which is a neat
  coupling between §1.1 and this section.
- **Connectivity is not guaranteed.** Three chunks had no overlap with the main network, so their
  absolute height is unconstrained and was left unchanged and *flagged*. A levelling network on a
  road graph can be disconnected in a way a grid never is.

**Open problem.** Post-hoc levelling of a single scalar per chunk is the cheap version. The honest
version is joint alignment — solving all chunks against each other (and against an absolute
reference) simultaneously, with full rigid transforms rather than a vertical offset. At 944 chunks
(§1.7) the post-hoc approach is still tractable, but the residual after a 1-DOF fix is where the
remaining error will live.

### 1.5 Retrieval-based loop closure fails in built environments

A genuinely counter-intuitive result, and a counter-example to standard practice.

Weak cross-pass matching has a textbook remedy: retrieval-based loop closure (vocabulary tree),
which finds pairs that *look* alike regardless of where the prior says they are. It is exactly
right when priors are untrustworthy. This repo had a measured win from it on rural back roads.

On a subdivision it made things dramatically worse:

```
                        alignment residual   connected component   positions
spatial + sequential      25.5 / 28.1 m             —               318/318
+ vocab retrieval        7961  / 42.9 m         1554/1908           259/318
```

The cause is the built environment itself. A subdivision is constructed from a handful of house
plans; mailboxes, lamp posts, driveway aprons and rooflines repeat at ~20 m intervals. Retrieval
cannot distinguish one cul-de-sac from the next, and the **cross-pass pair counts went up**
(14,741 vs 10,042) while the reconstruction disintegrated. A mean three orders of magnitude above
the median is cameras thrown kilometres away.

**Two lessons.** First, appearance-based place recognition degrades with *architectural
repetition*, which is a property of the built environment and anti-correlated with the rural
settings where such methods are usually demonstrated. Second — and more useful — **pair counts are
not a quality signal**. More matches made the model worse. Registered fraction and
connected-component size are the honest ones.

### 1.6 Metrics that mislead

Three, all encountered in one run.

**PSNR against masked pixels.** The capture vehicle occupies 23–36% of every frame and is masked.
gsplat zeroes masked pixels in the *render* but not in the *ground truth*, so a mask-trained model
is scored against the vehicle it was told to ignore. Measured gap: **+5.9 dB** (median 13.84 →
19.75 dB) once scored on visible pixels only. Comparing runs on the naive number silently rewards
*not masking*.

**Alignment residual as a quality metric.** `model_aligner`'s reported error is the residual
against the *GPS priors*, so it inherits their noise. Residuals here ranged 1.7–50 m with **no
relationship** to reconstruction quality: the worst-residual chunk had good GPS (DOP 4.3), and a
chunk with DOP 19 had a middling residual. Treating it as quality sends you chasing the datum.

**Plane fits to a corridor.** An early attempt to detect "askew" tiles fitted a ground plane to
the camera path. A corridor is nearly a *line*, so the gradient across it is unconstrained and the
fit returned a meaningless 69.6° / 537 m rise. Measuring *along* arclength is the well-posed
version. **Any statistic computed over a 1-D camera path needs its conditioning checked**, which is
a standing hazard in this domain and absent from bounded-scene work.

### 1.6a The same ill-conditioning, walked into twice

Worth recording because it is the clearest evidence that this hazard is
structural to corridor captures rather than a one-off mistake.

Having diagnosed the dz-only levelling of §1.4 as insufficient — six chunks
remained in >3 m disagreement with neighbours while being internally consistent
to 0.02–1.67 m, which is the signature of a chunk *tilted* as a block — the
obvious generalisation is to solve a vertical **plane** per chunk,
`dz(x,y) = a + b·x + c·y`, using every overlapping point pair rather than
pairwise medians. Thousands of observations, 3 unknowns per chunk: comfortably
determined, on paper.

The solution came back wanting to tilt chunks by **45°, 35° and 21°** and shift
them by **±35 m**. Nonsense, and nonsense of a specific kind: the observations
that tie two chunks together lie along the **road they share**, which is a
*curve*, not an area. The tilt component across that curve is unconstrained,
so the solver is free to assign arbitrarily large tilts that cancel along the
observed direction. Exactly the failure of §1.6's plane-fit, one level up.

**The rule this suggests:** in a corridor capture, *any* parameter that is only
observed along the driven path is under-determined in the perpendicular
direction, no matter how many observations you have — because the observations
are not independent samples of a 2-D field, they are a 1-D curve embedded in
one. Counting observations is not a test of conditioning.

The consequence is that a rotated chunk **cannot be corrected from corridor
overlaps alone**. Fixing it needs an absolute reference with genuine 2-D
support — lidar DTM is the obvious candidate — which promotes "absolute datum"
from a nicety to a requirement at network scale. That is now open problem 1.

### 1.7 Cost structure, and the scaling trap

Measured unit costs (GH200, per chunk):

| stage | cost | scales with |
|---|---|---|
| ingest | 2.43× realtime (1.75× threaded) | video duration |
| poses | 32.4 GPU-min | images per chunk (matching is superlinear) |
| train | ~57 GPU-min | **iterations, ~constant** |
| merge | seconds | gaussians |

Training is `--max-steps` bound, so **per-chunk training cost is independent of chunk size**. That
makes total cost scale with **chunk count**, and produces a direct conflict with §1.1:

> The fix for two-roads-in-one-cell is a *smaller* cell. Smaller cells mean more chunks. More
> chunks mean proportionally more training. The quality fix and the cost are in opposition, and
> the optimum is not obvious.

Extrapolating to the target network (166 km of named back roads, measured from OSM, 944 cells at
200 m):

```
poses    944 × 32.4 min  =   510 GPU-h
train    944 × 57   min  =   897 GPU-h
                           ~1,400 GPU-hours   ≈ 7.3 days on 8 GPUs
```

And the constraints that bite *before* GPU time: ~8.4 h of driving, ~771 GB of footage, ~4.2 TB of
working storage. Capture logistics, not compute, is the near-term bottleneck — which is itself a
finding, and one that does not appear when the dataset is a download.

### 1.8 Representation cost is not one number

For output, two *independent* levers, and which matters depends entirely on the consumer:

| lever | saves | binds |
|---|---|---|
| drop SH bands 1–3 | **73% of bytes** (45 of 59 properties) | bandwidth |
| decimate by opacity × footprint | the gaussian **count** | rasteriser fill/sort |

A consumer streaming to a real GPU is bandwidth-bound; a software rasteriser in a headless test is
count-bound, and *compression does not help it at all* — the bytes decompress to the same count.
This was a real misdiagnosis: a request for `.spz` (a byte-compression format) would not have
fixed the problem it was requested for.

Applying both at 1/8: **6.18M → 773k gaussians, 1460 MB → 43 MB (3.0%)**, with the kept gaussians
carrying **75.9%** of total contribution. Ranking must be done on the *post-activation* quantities
(sigmoid opacity, exp scale) — ranking the stored parameterisation ranks the wrong thing.

### 1.9 A note on the sensor: never rebuild the sphere

Included because it is the one thing that was got right in advance and is invisible when it works.

A 360 camera is two ~95° hemispheres a few centimetres apart. The intuitive pipeline stitches them
into an equirectangular sphere and reconstructs from that. This is wrong in a way that is fatal and
non-obvious: **parallax between the lenses is disparity, and disparity is depth.** A stitcher's job
is to destroy it. Worse, after a depth- or flow-based warp the pixels near the seam belong to no
consistent central projection, and SfM either fails there or triangulates confidently wrong. The
observed symptom was doubled geometry and holes.

The resolution is not a better blend: it is to **never produce an image that needs one**. Virtual
pinhole views are planned *inside each lens' cone* — here 3 views per lens at ±45.25° with 90° FOV,
against a 94.74° half-angle derived from the file's own blend width — and each (view, lens) pair is
a distinct camera with its own optical centre. Where both lenses see the same direction, SfM gets
two genuine views a few centimetres apart, which is *information* rather than conflict.

The cost is a small blind wedge at the lens boundary (0.4% of the horizon), filled by other frames
because the rig is moving.

---

## Open problems

Ordered by how much they will hurt at network scale.

1. **Joint alignment at scale.** Post-hoc 1-DOF levelling works at 29 chunks. At ~1,000, with a
   road graph that can be disconnected, the right formulation is a global pose-graph over chunks
   with an absolute reference (lidar DTM, where it exists). What is the right observation model —
   overlapping camera positions, or co-visible 3D structure?
2. **Partitioning that respects topology, not geometry.** Cells are a grid; roads are a graph.
   A grid cell can contain two roads that never see each other, which is §1.1's root cause.
   Partitioning by *corridor segment* rather than by cell would remove the failure entirely, at
   the cost of irregular, overlapping chunk bounds and a harder ownership rule at merge.
3. **Appearance across sessions.** A network cannot be captured in one drive. Two passes hours or
   weeks apart differ in sun angle, foliage and parked cars. Per-image appearance embeddings are
   the standard answer for unstructured photo collections; whether they suffice across *sessions*
   on the same road, and what they cost in a chunked pipeline, is open.
4. **Place recognition in repetitive built environments** (§1.5). If retrieval is actively harmful,
   what replaces it when priors are bad? Sequence-level matching? Geometric verification against
   the road graph?
5. **Validation without ground truth.** §1.2 found three free invariants. Are there more? A
   catalogue of cheap invariants would be worth more to this field than another metric, precisely
   because the failures are silent.
6. **Moving objects.** Passing cars and pedestrians become smeared ghosts. Corridor pruning removes
   the worst floaters; a principled transient-aware formulation is untouched here.
7. **The cost/quality conflict of §1.7.** Smaller cells fix topology violations and multiply
   training cost. Is there a formulation where chunk count and training budget decouple?

## Anti-patterns, collected

Short list, all paid for:

- trusting **registered fraction** as health (§1.1)
- expressing a matcher's **reach in samples** while its error is in metres (§1.3)
- aligning partitions **independently** and expecting a world (§1.4)
- reading **pair counts** as quality (§1.5)
- comparing **PSNR across masked and unmasked** runs (§1.6)
- fitting a **plane to a corridor** (§1.6)
- assuming **byte compression** helps a count-bound consumer (§1.8)
- **stitching** a 360 sphere before reconstruction (§1.9)
- checking a cluster is idle by **grepping pod names** instead of asking the scheduler
  — not a splat problem, but the same shape: a check that cannot fail

