<!-- SPDX-License-Identifier: Apache-2.0 -->
# The lens seam: what GoPro does, and why we deliberately do not copy it

A GoPro MAX 2 shoots with two lenses about 3 cm apart, each seeing 94.74° from
its own axis (the pipeline derives that number from the file, not a spec
sheet — see below). Where their coverage meets, the same direction exists
twice, photographed from two different places.

GoPro Player resolves that beautifully. Our early ingest resolved it by
cross-fading, and the result was a doubled truck and a house with a bite taken
out of the roof. So the obvious question — *GoPro does this cleanly, can we do
what they do?* — is worth answering properly, because the answer turns out to
be **no, and doing it their way would make our output worse, not better.**

## What GoPro actually does

Their in-camera / in-Player stitcher is branded **D.WARP**, and the patents
say plainly what it is:

- [US 11,568,516](https://patents.google.com/patent/US11568516) — *Depth-based
  image stitching for handling parallax*
- [US 11,748,952](https://patents.google.com/patent/US11748952) — *Apparatus
  and method for optimized image stitching based on optical flow*
- [US 10,699,375](https://patents.google.com/patent/US10699375) — *Method and
  apparatus for image adjustment for panoramic image stitching*

Estimate depth (or dense optical flow) across the overlap, then **locally warp
one hemisphere onto the other until the disparity cancels**, and photometrically
match the two exposures on the way through. GoPro's own write-up on
[the art of stitching spherical content](https://gopro.com/en/us/news/the-art-of-stitching-spherical-content)
describes composing the two images "into a seamless, unified whole."

That is the correct goal — *for a picture*. A viewer wants one sphere from one
apparent viewpoint, and warping away the parallax is exactly how you fake that.

## Why it is the wrong goal for us

Parallax between the two lenses is not noise. It is **disparity**, and
disparity is depth. A stitcher's job is to destroy it; a reconstruction
pipeline's job is to consume it.

Worse, the warp is not a camera. After D.WARP, the pixels near the seam belong
to no consistent projection — they have been dragged, per-pixel, by a
depth-dependent amount. Structure-from-motion assumes each image is a central
projection through one optical centre. Hand it a stitched sphere and it will
either fail to triangulate near the seam or triangulate confidently *wrong*.
The output looks like exactly what we saw: geometry that is doubled where the
warp could not decide, and holes where it decided incorrectly.

So there are three options, and only two are honest:

| Approach | Looks like | Good as reconstruction input |
| --- | --- | --- |
| Hard cut at the boundary | visible colour step, geometry jump | no — a discontinuity mid-image |
| Cross-fade the overlap | soft seam, ghosting on near objects | **no — this is what doubled the truck** |
| Depth/flow warp (D.WARP) | seamless | **no — destroys disparity, breaks the projection model** |
| **Never cross the boundary** | no seam, because there is no join | **yes** |

## What everyone else who hit this concluded

- **FFmpeg's `gopromax_opencl` filter**
  ([patch](https://patchwork.ffmpeg.org/project/ffmpeg/patch/20240803005601.44246-2-aimingoff@pc.nifty.jp/))
  is the reference open implementation of .360 → equirect. Its overlap
  handling is a *linear alpha ramp over 64 px* — i.e. the same cross-fade we
  removed. It is fine for viewing and ghosts identically for reconstruction.
- **max2sphere** (Paul Bourke / Trek View, Apache-2.0), which our EAC decoder
  is derived from, makes a hard choice per pixel. Also fine for viewing.
- **Facebook Surround360** (BSD, archived) and
  [MungoMeng/Panorama-OpticalFlow](https://github.com/MungoMeng/Panorama-OpticalFlow)
  are the open optical-flow stitchers. They are good at the thing we do not
  want done.
- **[Seam360GS](https://arxiv.org/abs/2508.20080)** (ICCV 2025) attacks exactly
  this problem for gaussian splatting and reaches the same diagnosis: the
  artifact comes from the **baseline between the two optical centres**, and the
  fix is to model the rig as two centres rather than pretend it is one.
- Commercial 360-splat tooling has landed in the same place — the current
  advice for Insta360 rigs is to feed the **two fisheye streams separately**
  rather than an Insta360 Studio stitch, precisely to avoid stitching artifacts
  on near subjects.
- Photogrammetry practice agrees: raw fisheye in, not stitched equirect.

## What gaussworks does

**Plan the virtual cameras inside each lens' cone** (`splatpipe/viewplan.py`).

The old ring — six views at 60° yaw spacing — was laid out with no knowledge of
the hardware, which put the lens boundary at yaw ±90° *inside four of the six
images*. On a driving capture, yaw ±90° is broadside: the parked cars, the
mailboxes, the house fronts. The nearest objects in the scene, which carry the
most parallax, landed exactly on the join.

The new plan asks the profile where the lenses point and how far each one sees,
then tiles each cone:

```
[eac] 5952x1920: side=2016 face=1920 blend=96px -> each lens sees 94.74 deg from its axis
[viewplan] 6 view(s) / frame
   front: 3 x 80 deg 1920x1440 (24.0 px/deg)  yaw -51.9, 0, 51.9
    rear: 3 x 80 deg 1920x1440 (24.0 px/deg)  yaw 128.1, 180, 231.9
```

Front covers ±91.9°, rear covers 88.1°–271.9°. The union is the whole sphere,
the two lenses genuinely overlap by ~4°, and **no single image contains data
from more than one lens**. Six images per frame — the same count the seam-
crossing ring produced. The fix is free.

Where both lenses see the same direction, the pipeline now gets *two images
with a real 3 cm baseline* instead of one image with a contradiction. That is
information, and SfM is built to use it.

### Where the 94.74° comes from

Not from a spec sheet. A .360 stores each side cube face as two half-images —
one per lens — that overlap in a duplicated strip. For the Max 2's 5952×1920
tracks the side section is 2016 px and the face is 1920, so the strip is 96 px,
and the equi-angular mapping turns that into **4.74° of overlap past 90°**.
The first-generation MAX at 4096×1344 gives 32 px → 2.20°. The camera tells us
its own lens coverage; we just had to read it.

## Residual limits

- The ~4° double-covered band is the only place the two lenses can be compared.
  Directions outside it are seen once, so no amount of processing recovers
  stereo there. Vertical baseline still requires a second physical camera.
- The outer edge of any fisheye is its worst region (lowest angular resolution,
  most flare). Profiles can trim it with `usable_half_fov_deg`; we currently
  use the full cone.
- Pre-stitched input (Mapillary, an Insta360 Studio export) has the seam
  already baked in and cannot be undone. The `equirect-360` profile says so out
  loud, and `mask --seam-band-deg` remains available to drop those pixels —
  the best available move, not a good one.
