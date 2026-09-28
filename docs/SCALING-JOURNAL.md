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

---

## Entry 2 — Gosheff Lane, 2026-09-27: the metric and the defect

Entry 1 ended with a world that registered cleanly and drove badly. This entry is about the
gap between those two sentences, and it turned out to be a story about *measurement* rather
than about reconstruction. Every failure below is a case of the number moving while the defect
stayed, or the defect moving while the number stayed.

The practical question was narrow: a second bake of one street at higher settings *looked*
better — was it, and was it worth 5.9× the bytes? Answering it required three separate
corrections to how we were measuring, and the third one changed the conclusion.

### 2.1 PSNR is not comparable across training resolutions — by 0.13 dB

The survey bake renders views at 1620 px; the high-quality bake at 2160 px. Both reported PSNR
against their own held-out views, and those numbers were being read side by side. They are not
comparable: a model trained at 2160 px is scored against a target containing high-frequency
detail that simply does not exist in the 1620 px target. More pixels is a harder exam.

So we added `eval --at-width`, which renders both at a common width — scaling intrinsics,
downsampling ground truth with `INTER_AREA` (so the model is not penalised for failing to
reproduce aliasing we introduced ourselves) and the vehicle mask with `INTER_NEAREST` (so the
rig stays excluded rather than being feathered into the comparison).

**The correction was worth 0.13 dB.** The direction was right and the magnitude was wrong by
roughly an order of magnitude against expectation.

This is worth recording precisely *because* it is a negative result. The hypothesis was
plausible, the reasoning was sound, and the effect was negligible — and there was no way to
know that without building the instrument. The cost of measuring your own correction is
usually small next to the cost of carrying an unquantified worry through every subsequent
comparison. A worry cannot be traded off against anything; 0.13 dB can.

With that settled, the fair comparison over the same ground:

| | PSNR (visible, @1620 px) | bytes / 1000 m² |
|---|---|---|
| survey | 21.43 dB | 2.7 MB |
| high-quality | 22.14 dB | 15.8 MB |

**+0.7 dB for 5.9× the storage.** On that evidence the expensive bake is not worth it.

Except that PSNR was never measuring the thing anyone had complained about.

### 2.2 The metric that tracks the defect

The reported problem was not softness. It was a *step in the road*, and splats hanging above
the road surface on one side of it. That is §1.4's disease — chunks levelled independently
against their own GPS priors — and PSNR is close to blind to it, because PSNR is dominated by
large smooth regions and barely moves when a chunk sits half a metre wrong.

The invariant is free, in the §1.2 sense. Chunks overlap by design, so wherever two chunks'
camera paths pass through the same place, *their heights must agree*. Disagreement there is not
noise to be averaged; it is the defect, already localised to a pair. Measured over the same
ground:

| | seams | median \|dz\| | worst |
|---|---|---|---|
| survey | 6 | 0.82 m | **4.88 m** |
| high-quality | 8 | 0.53 m | **1.44 m** |

Two seams near 4.8 m in the survey; nothing above 3 m anywhere in the high-quality bake.

Put the two tables together and the point is stark: **over identical ground, PSNR separates
these worlds by 0.7 dB while the worst seam separates them by 3.4 m.** One of those numbers
tracks what a driver hits and the other does not. We had been optimising, comparing and
reporting the one that does not, for no better reason than that it is the number the
literature reports — and the literature reports it because bounded scenes do not *have* seams.

The general form: **when you move a technique to a new regime, the field's standard metric
travels with it and its validity does not.** The metric was not wrong for bounded scenes. It
became wrong when partitioning was introduced, because partitioning created a failure mode
that the metric cannot see.

### 2.3 Scope is a confounder when worlds differ in extent

The first version of that comparison used each world's *whole* extent, and the survey looked far
worse — worst seam 13.68 m, twelve chunks in >3 m disagreements. That comparison is invalid.
The survey covers 1.16 km² including sparse fringes driven once under canopy; the
high-quality bake covers 0.13 km² of well-covered street. The fringes are where the bad seams
live, so a whole-world comparison credits the small bake for *not containing the hard parts*.

`seams --within <other world's chunks>` restricts to chunks at least 30% inside the other
world's footprint. That is what produces the 4.88 m figure above, and it is the honest one.
Worth stating plainly because the invalid comparison flatters the conclusion we already
believed, which is exactly when a comparison needs checking hardest.

### 2.4 Two variables, one config

The high-quality config changed *both* resolution (`px_per_deg` 18→24, `spacing_m` 1.75→1.1,
`jpeg_quality` 95→97) and chunk geometry (`cell_m` 200→120, `spatial_radius` 12→20), then beat
the survey on both metrics. It therefore cannot say which change earned which win — and the two
halves have wildly different costs. Resolution costs 5.9× the bytes. Smaller cells cost
essentially nothing in storage.

The hypothesis is that they map cleanly onto the two metrics: pixels buy PSNR, and smaller
cells buy seam agreement, because a shorter chunk accumulates less drift before it has to agree
with its neighbour. If that holds, the defect that actually matters is fixed by the free half,
and re-baking a whole neighbourhood stays affordable.

A control is running as this is written: survey resolution, high-quality geometry, one
variable. Note that `spatial_radius` had to be held at 12 rather than copied — §1.3's lesson,
that the radius is counted in *capture positions* while the error it must cover is in metres.
The control keeps the survey's 1.75 m spacing, so 12 positions reach 21 m, against the
high-quality bake's 20 × 1.1 = 22 m. Copying the 20 would have reached 35 m and quietly made
this a two-variable experiment again.

**The control falsified the hypothesis, and the per-seam breakdown falsified the obvious
replacement too.** Both bakes share a chunk grid, so they compare seam for seam:

| seam | survey res (1620) | high res (2160) | Δ |
|---|---|---|---|
| x4_y0 / x5_y-1 | **3.85 m** | **0.51 m** | **3.34** |
| x2_y-3 / x3_y-3 | 0.77 | 1.10 | −0.33 |
| x4_y-3 / x5_y-3 | 0.14 | 0.17 | −0.03 |
| x3_y-3 / x4_y-3 | 1.01 | 0.99 | 0.02 |
| x5_y-2 / x5_y-3 | 0.11 | 0.13 | −0.02 |
| x4_y-3 / x5_y-2 | 0.17 | 0.18 | −0.01 |
| x5_y-1 / x5_y-2 | 0.54 | 0.54 | 0.00 |
| x1_y-3 / x2_y-3 | 1.44 | 1.44 | 0.00 |

Seven of eight seams are identical — five of them within 3 cm. Resolution did not improve seams
systematically, and the 1.44 m worst seam is *the same seam at the same value* in both. Remove the
one outlier and the two worlds are indistinguishable: worst 1.44 m, median 0.54 m, both.

So the entire 1.44 → 3.85 m gap is one chunk pair. And `x5_y-1` agrees with `x5_y-2` at 0.54 m in
*both* runs, which localises the fault to `x4_y0` alone.

`x4_y0` sits 299 m from the capture centre against a 350 m ingest radius — the outermost chunk
with anything like full coverage. The tighter `spacing_m` of the high-resolution config gave it
**233 frames (169 core) where the control got 192 (128)**, a 24% difference concentrated exactly
where coverage was already tapering. Every other chunk received near-identical frame counts.

The correct reading is therefore neither "resolution buys seam quality" nor "geometry buys seam
quality":

> **A chunk's levelling is a draw, and coverage sets the odds.** Well-covered chunks land within
> centimetres of each other across completely different configurations. A coverage-starved chunk
> can land 3.85 m out or 0.51 m out, and which one you get is not determined by the settings that
> were varied.

This has three consequences for scale, and they are the reason this entry exists:

1. **The decision the experiment was run to make looked settled, and §2.4a shows it was not.**
   Survey resolution with 120 m cells reproduces the expensive bake's seam structure on this
   street at 2.7 MB per 1000 m² instead of 15.8, and the resolution finding holds. The claim
   that this settled the neighbourhood did not survive baking the neighbourhood.
2. **Worst-case seam is not a property of a configuration.** It is an order statistic over draws.
   Reporting "worst seam 1.44 m" for a 9-chunk bake says little about what 300 chunks will do —
   tuning against a single small bake's worst case is tuning against noise.
3. **At scale you cannot avoid bad draws; you must detect and repair them.** A neighbourhood is
   hundreds of chunks and some fraction will be coverage-starved at a capture boundary. The useful
   artefact is not a better config, it is §2.2's seam check pointing at `chunk_x4_y0` by name —
   which it did, unprompted, with a number attached.

### 2.4a The neighbourhood refuted most of 2.4

The 120 m re-bake of the whole capture (49 chunks against the survey's 29) landed, and it does
not support the conclusion above. Same ground, same resolution, cell size the only change:

| | chunks | seams | median-of-medians | worst | chunks >3 m |
|---|---|---|---|---|---|
| survey, 200 m | 29 | 32 | 1.18 m | 13.68 m | 12 of 27 (44%) |
| re-bake, 120 m | 49 | 62 | **0.76 m** | **67.33 m** | 20 of 47 (43%) |

Smaller cells improved the TYPICAL seam by a third and left the failure *rate* unchanged (43%
against 44%; an earlier draft said 38%, measured before the last chunks had finished writing
their corridors — the queue was drained but the files were not). They also produced a catastrophe the small bake gave no hint of: four contiguous chunks in the
west (`x-4_y0`, `x-4_y-1`, `x-5_y-1`, `x-5_y0`) mutually disagreeing by 18–67 m.

Two corrections follow, and the second is the one that matters.

**"Coverage sets the odds" is not supported at this sample size.** That claim came from a single
chunk in a nine-chunk bake. Across 47 chunks, core frame count does not separate the failures:

    chunks with <60 core frames    40% blew the 3 m budget
    chunks with >=60               38%

(Those two proportions were computed on the same premature read as the 38% above. The settled
figure is 20 of 47; the *separation* between thin and fat chunks is what matters here and it is
absent either way, but the exact percentages should be re-derived before anyone quotes them.)

In the survey the relationship is actually *inverted* — the chunks that failed had a median of
192 core frames against 127 for the ones that passed. Whatever decides a bad draw, it is not how
much data the chunk got. §2.4's mechanism should be read as an untested hypothesis that one
chunk was consistent with, and the generalisation was mine, not the data's.

**Generalising a tail statistic from nine chunks to fifty was the actual error.** §2.4 itself
says worst-case seam is an order statistic rather than a property of a configuration — and then
the recommendation was made on a nine-chunk bake's worst case anyway. The median generalised
fine. The tail did not, and the tail is what makes a world undriveable. A small bake can measure
a central tendency; it cannot measure a rare failure, because it does not contain enough draws
to have seen one.

The practical upshot is unchanged in one respect and reversed in another: 120 m cells are still
the better setting for the typical seam, and the re-bake is still **not** publishable — the 3 m
gate rejects it, correctly, and it would have shipped a 67 m step into the world otherwise. The
open question is no longer "which config" but "what makes a region fail", which §2.4's framing
was actively unhelpful for.

A cheaper prediction also falls out, still to be tested: if coverage sets the odds, the fix for a
boundary chunk is *more frames there*, not more pixels everywhere — a local `spacing_m`, or simply
declining to publish chunks whose core-frame count falls below what neighbouring chunks receive.

### 2.5 Two checks that could not fail, in one afternoon

Entry 1 closed on a check that could not fail. Two more, both mine, both caught only by
deliberate adversarial testing:

**A grep that swallowed a crash.** The fair-resolution comparison ran as
`splatpipe eval ... 2>&1 | grep -a "PSNR"`. The job produced *no output whatsoever*. The cause
was a `SyntaxError` in an edit made minutes earlier — and because the filter selected only the
success string, a crash and a clean run that happened to print nothing were indistinguishable.
The filter was reporting on itself.

**A failure that exited zero.** The new `seams --fail-over` threshold printed `FAIL` and
returned a status that `main()` was discarding, so the process exited 0. A CI gate built on it
would have been strictly worse than no gate: it would have produced a green tick and the
authority that comes with one.

Both are the same shape, and it is the shape §1.2 warned about from the other side. *Silence is
not success.* A monitor, a filter, or a gate must be shown to emit something on the failure path
before its silence means anything. The discipline that catches it is cheap and mechanical: after
writing a check, **make it fail on purpose**. `seams` against `--fail-over 3.0` exits 0 and
against `--fail-over 0.5` exits 1 on the same world — that pair of runs is what makes the
threshold trustworthy, and neither run alone would.

This is not a splat problem. It is the dominant failure mode of automated verification in
general, and it is more dangerous than an absent check because it manufactures confidence.

### 2.6 What this entry actually changed

Not the reconstruction. The instruments:

- PSNR is now reported at a stated common width or not compared at all;
- seam agreement is a first-class command with a CI threshold, and is the number that gates a
  world as driveable;
- comparisons across worlds of different extent are restricted to shared ground;
- checks are verified to fail before their passes are believed.

The reconstruction improvements of Entry 1 were found *by* instruments like these. That is the
pattern worth extracting: in a regime where the standard metrics were validated somewhere else,
**the highest-leverage work is often building the measurement, not improving the model.** Three
of the last four real improvements to this pipeline came from noticing that a number did not
mean what it appeared to mean.

---

## Entry 3 — 2026-09-28: we partition before we solve, and everyone else solves before they partition

Open problem 10 asked what makes a *region* fail. The answer appears to be architectural, and it
is visible by comparing our pipeline's ORDER against the reference implementations rather than by
any further measurement of our own.

**What we do.** Cut the capture into chunks, then run COLMAP/GLOMAP independently inside each
chunk, then align each chunk to its own GPS priors with `model_aligner`. Every chunk arrives at
its datum alone.

**What the reference large-scale 3DGS implementation does.** Hierarchical 3DGS (Kerbl et al.,
SIGGRAPH 2024) — the system built for exactly our regime, kilometre-scale street capture — runs
*global* SfM over the whole capture first, aligns and scales that single reconstruction to
metric, and only then cuts it into chunks. Each chunk inherits its poses from the global
reconstruction; per-chunk bundle adjustment then refines locally. Their partitioning is a
**training** optimisation applied to an already-consistent world.

That is the whole difference. In their pipeline the chunks cannot disagree about the height of a
shared road, because the height was decided once, before the cut. In ours, agreement is something
we hope for and then measure.

### 3.1 It explains every observation we could not explain

Independent per-chunk datums predict exactly what we see, and the alternatives do not:

- **Failure rate is ~44% at 200 m and ~43% at 120 m.** If each chunk's datum is an independent
  draw, cell size changes how many draws there are and not the odds of each — which is precisely
  what §2.4a measured and could not account for.
- **Frame count does not predict failure** (§2.4a), and in the survey the relationship inverts.
  Under this model it should not predict anything: the error is in how the datum was chosen, not
  in how much data supported it.
- **Failures come in contiguous regions**, four chunks at 18–67 m, not a scatter. Neighbouring
  chunks share a stretch of GPS. Under continuous canopy that stretch is biased the same way for
  all of them, so they drift *together*, and the visible seam is at the boundary with chunks
  outside the bad stretch. A per-chunk lottery would scatter; a shared bad prior clusters.
- **Loop closure inside a chunk made things worse** (§1.5) and smaller cells only moved the
  median. Both are interventions *within* a chunk, and the defect is *between* chunks.

It also explains why no configuration change has flattened the tail, and predicts none will. We
have been tuning the partition while the disagreement is created by the ordering.

### 3.2 Two ways out, and they cost very differently

**A. Solve globally, then cut.** The Inria ordering. Our survey is 5,075 frames, which is
comfortably inside the range these systems target, and COLMAP ships `hierarchical_mapper` for
scenes where incremental SfM is too slow — it partitions into *overlapping* sub-models,
reconstructs them independently and merges them, which is the same idea one level down. This is
the correct fix and it is a pipeline restructure, not a knob.

**B. Stop treating chunks as independent, without re-solving.** Our chunks already overlap by
40 m, which means neighbouring chunks contain the *same frames*. COLMAP merges sub-models
precisely when "those sub-models have common registered images", and recommends a global bundle
adjustment afterwards to improve the alignment. So the constraint we need is already in the data
and is currently thrown away: estimate the relative transform between each chunk pair from their
shared registrations, optimise a pose graph over chunks, and demote GPS from per-chunk datum to a
weak global prior.

**An honest caveat on B.** This is the levelling network of §1.4 and §1.6a, which failed. But it
failed for a reason that B avoids: those attempts levelled against *GPS altitude* and then
against a *DTM* — two noisy external references measuring different things. Shared camera
registrations are an internal constraint of a completely different quality: two chunks that
contain the same frame must place that frame identically, and there is no datum to argue about.
Whether that survives the corridor's conditioning is untested, and the corridor literature is
explicit that in corridor-like environments without good loops, drift in some directions is
especially hard to correct — which is §1.6a from the SLAM side.

### 3.3 What this says about the last two days

The measurement work of Entry 2 was not wasted — `seams` is what made the failure legible, and
the gate is what stops an undriveable world shipping. But the *experiments* it powered were all
searches over configuration, and the answer was never in the configuration. Three bakes were
spent asking which cell size and which resolution, when the informative comparison was between
our pipeline's shape and a published one's, and cost a search rather than a GPU.

Worth generalising: when a defect survives every setting you vary, stop varying settings. The
question "what is structurally different about how we do this?" is cheap, and we reached for it
third.

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
8. **A driveability metric, not a seam metric.** §2.2's seam agreement catches steps between
   chunks. It says nothing about error *within* a chunk, and nothing about lateral or along-track
   error — the reported misalignment of ~2 m to the east is invisible to it. The general problem:
   what is the smallest set of free invariants that bounds every way a corridor world can be
   wrong in a way a driver notices?
9. **Seam count versus seam size.** Measured in §2.4a and it is not a clean trade: smaller cells
   improved the median by a third AND made the worst seam five times worse. Median and tail move
   independently, so a single cell size cannot be tuned against both.
10. **What makes a REGION fail?** *Probable answer in Entry 3: chunks are solved and levelled
   independently, so each picks its own datum, and neighbours sharing a bad GPS stretch drift
   together. Untested until the ordering is changed.* The 120 m re-bake's damage is four contiguous chunks at 18–67 m,
   not a scatter. Frame count does not predict it (§2.4a) and the reconstructions did not
   visibly fracture. Until this is understood, no config change should be claimed to fix seams —
   this is now the central open problem, and everything in §2.4 is downstream of it.
11. **How many chunks must a trial bake contain to measure a tail?** Nine was enough for the
   median and badly insufficient for the worst case. Trials are how we iterate; not knowing the
   size at which their rare failures become measurable makes every trial result suspect.

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
- comparing **PSNR across training resolutions** (§2.1) — though the correction is only 0.13 dB
- optimising the field's **standard metric** after changing the regime that validated it (§2.2)
- comparing worlds of **different extent** without restricting to shared ground (§2.3)
- generalising a **tail statistic** (worst seam) from a trial too small to contain a rare
  failure, while writing in the same entry that it is an order statistic (§2.4a)
- changing **resolution and geometry in one config** and reading the result as either (§2.4)
- filtering a log for the **success string**, so a crash and a clean run look identical (§2.5)
- a gate whose **failure path exits zero** — worse than no gate, because it manufactures
  confidence (§2.5)

