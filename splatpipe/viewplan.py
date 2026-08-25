# SPDX-License-Identifier: Apache-2.0
"""Lens-aware view planning: choose virtual cameras that never cross a seam.

The old plan was a ring of virtual cameras at fixed yaw spacing, laid out with
no knowledge of the hardware. On a two-lens 360 camera that guarantees the
worst case: half the views straddle the lens boundary, so half the images are
two viewpoints glued together, and the boundary sits at yaw +-90 -- broadside,
which on a driving capture is the roadside. Houses and parked cars, the things
closest to the camera and therefore with the most parallax, land exactly on
it. That is the doubled truck and the half-missing roof.

The fix is not a better blend. It is to stop producing images that need one:

    plan the virtual cameras INSIDE each lens' cone.

For a back-to-back pair that sees ~95 deg per lens, three 80 deg views per
lens at +-52 deg from its axis tile the whole hemisphere with 28 deg of mutual
overlap and never reach the boundary. Six images per frame -- the same count
the seam-crossing ring produced -- and every one of them is a photograph taken
through a single lens from a single optical centre, which is the only kind of
image a structure-from-motion pipeline is entitled to assume it has.

The seam does not disappear from the world; it stops being inside an image.
Where two lenses see the same direction the pipeline now gets two views with a
few centimetres of real baseline, which is information rather than conflict.

Explicit view lists still work (`views:` in a config). They are assigned to
whichever lenses actually cover them, split into one image per lens, and the
uncovered part is masked -- correct, but it spends images on masked pixels.
Auto planning is the better default.
"""

from __future__ import annotations

import math

import numpy as np

from .eac import view_dirs

# coarse ray grid used for coverage tests during planning; the real masks come
# from the driver at full resolution
_PROBE = (48, 36)


def _even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


def _covered_fraction(driver, lens: int, yaw: float, pitch: float, fov: float,
                      aspect: float) -> float:
    w, h = _PROBE
    dirs = view_dirs(w, _even(w * aspect), fov, yaw, pitch)
    return float(driver.covers(dirs, lens).mean())


def _corner_dirs(fov: float, aspect: float, yaw: float, pitch: float) -> np.ndarray:
    """The four extreme rays of a rectilinear view, in the rig frame.

    A pinhole view's furthest ray from its own axis is always a corner, so
    testing four rays is exact -- and a sampled grid is not: its outermost
    samples sit inside the true corners, which is how a 'fully covered' plan
    still clipped 0.4% of the frame.
    """
    t = math.tan(math.radians(fov) / 2.0)
    d = np.array([[sx * t, 1.0, -sy * t * aspect]
                  for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, yw = math.radians(pitch), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(yw), math.sin(yw), 0],
                   [-math.sin(yw), math.cos(yw), 0], [0, 0, 1]])
    return d @ (rz @ rx).T


def _inside(driver, lens: int, yaw: float, pitch: float, fov: float,
            aspect: float) -> bool:
    """Ask the DRIVER, not a cone. An EAC lens boundary is a cube-face edge
    extended by the overlap, which is not a cone around the lens axis; planning
    against a cone left 0.1% of each outer view outside the lens."""
    return bool(driver.covers(_corner_dirs(fov, aspect, yaw, pitch), lens).all())


def _max_offset(driver, lens: int, pitch: float, fov: float,
                aspect: float) -> float:
    """Largest yaw offset from the lens axis that keeps the view inside it."""
    base = driver.lenses[lens].yaw_deg
    if not _inside(driver, lens, base, pitch, fov, aspect):
        return 0.0     # the view is wider than the lens; centre it and clip
    lo, hi = 0.0, 180.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _inside(driver, lens, base + mid, pitch, fov,
                                      aspect) else (lo, mid)
    return lo


def plan_auto(driver, *, fov: float = 80.0, px_per_deg: float = 24.0,
              pitches=(-12.0,), aspect: float = 0.75,
              view_overlap_deg: float = 12.0) -> list[dict]:
    """Tile every lens' cone with pinhole views that stay inside it."""
    width = _even(fov * px_per_deg)
    height = _even(width * aspect)
    plan: list[dict] = []
    for li, lens in enumerate(driver.lenses):
        full_sphere = driver.usable_half_fov_deg(li) >= 179.0
        for pitch in pitches:
            step = max(fov - view_overlap_deg, 1.0)
            if full_sphere:
                # a stitched sphere has no boundary to stay clear of, so tile
                # the whole circle instead of an arc (and don't duplicate the
                # wrap-around view the arc form would put at both ends)
                n = max(1, math.ceil(360.0 / step))
                offsets = [i * 360.0 / n for i in range(n)]
            else:
                omax = _max_offset(driver, li, pitch, fov, aspect)
                n = 1 if omax <= 1e-6 else 1 + math.ceil(2 * omax / step)
                offsets = [0.0] if n == 1 else list(np.linspace(-omax, omax, n))
            for off in offsets:
                plan.append({"yaw": round(lens.yaw_deg + off, 3), "pitch": pitch,
                             "fov": fov, "width": width, "height": height,
                             "lens": li, "lens_name": lens.name})
    return plan


def plan_explicit(driver, views: list[dict], *, aspect: float = 0.75,
                  min_coverage: float = 0.10) -> list[dict]:
    """Split a hand-written view list across the lenses that actually see it."""
    plan: list[dict] = []
    for v in views:
        asp = v.get("height", 0) / v["width"] if v.get("width") else aspect
        emitted = 0
        for li, lens in enumerate(driver.lenses):
            frac = _covered_fraction(driver, li, v["yaw"], v.get("pitch", 0.0),
                                     v["fov"], asp)
            if frac < min_coverage:
                continue
            plan.append({**v, "lens": li, "lens_name": lens.name,
                         "coverage": round(frac, 3)})
            emitted += 1
        if emitted == 0:
            raise SystemExit(f"[viewplan] no lens covers {min_coverage:.0%} of "
                             f"view {v}; check the profile's lens axes")
        if emitted > 1:
            print(f"[viewplan] view yaw={v['yaw']} straddles the lens boundary "
                  f"-> {emitted} images, each masked to its own lens. "
                  f"Auto planning avoids this; see `plan: auto`.")
    return plan


def plan_views(driver, cfg: dict | None = None) -> list[dict]:
    """Resolve an ingest config's view settings into concrete view instances."""
    cfg = dict(cfg or {})
    defaults = dict(driver.profile.defaults or {})
    views = cfg.get("views")
    if views:
        return plan_explicit(driver, views,
                             min_coverage=cfg.get("min_coverage", 0.10))
    pitches = cfg.get("pitch", defaults.get("pitch", [-12.0]))
    if isinstance(pitches, (int, float)):
        pitches = [float(pitches)]
    return plan_auto(driver,
                     fov=float(cfg.get("fov", defaults.get("fov", 80.0))),
                     px_per_deg=float(cfg.get("px_per_deg",
                                              defaults.get("px_per_deg", 24.0))),
                     pitches=tuple(float(p) for p in pitches),
                     aspect=float(cfg.get("aspect", 0.75)),
                     view_overlap_deg=float(cfg.get("view_overlap_deg", 12.0)))


def sphere_coverage(driver, plan: list[dict], width: int = 512) -> dict:
    """What fraction of the sphere the plan actually images.

    Views are clipped to their own lens, so a plan can leave a wedge at the
    lens boundary uncovered -- on a first-generation MAX (2.2 deg of overlap)
    it does. That is not necessarily a problem, because the rig is moving: a
    direction blind at yaw 90 now was imaged head-on a second ago. But it must
    not be silent, so the planner measures and prints it.
    """
    from .eac import equirect_dirs                     # noqa: PLC0415
    dirs = equirect_dirs(width)
    seen = np.zeros(dirs.shape[:-1], bool)
    for v in plan:
        t = math.tan(math.radians(v["fov"]) / 2.0)
        aspect = v["height"] / v["width"]
        p_, y_ = math.radians(v["pitch"]), math.radians(v["yaw"])
        rx = np.array([[1, 0, 0], [0, math.cos(p_), -math.sin(p_)],
                       [0, math.sin(p_), math.cos(p_)]])
        rz = np.array([[math.cos(y_), math.sin(y_), 0],
                       [-math.sin(y_), math.cos(y_), 0], [0, 0, 1]])
        local = dirs @ (rz @ rx)          # inverse of the plan's rotation
        fwd = local[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            in_view = (fwd > 1e-9) & (np.abs(local[..., 0] / fwd) <= t) \
                & (np.abs(local[..., 2] / fwd) <= t * aspect)
        seen |= in_view & driver.covers(dirs, v["lens"])
    lat = np.arcsin(np.clip(dirs[..., 2], -1, 1))
    # +-10 deg: narrow enough that a down-pitched plan covers it vertically by
    # construction, so what this measures is YAW completeness -- i.e. whether
    # the per-lens views leave a wedge at the boundary. A wider band would
    # mostly report the pitch, which is a deliberate choice, not a gap.
    band = np.abs(lat) <= math.radians(10)
    return {"sphere": float(seen.mean()),
            "horizon_band": float(seen[band].mean())}


def describe(plan: list[dict], driver=None) -> str:
    by_lens: dict = {}
    for v in plan:
        by_lens.setdefault(v["lens_name"], []).append(v)
    lines = [f"[viewplan] {len(plan)} view(s) / frame"]
    for name, vs in by_lens.items():
        yaws = ", ".join(f"{v['yaw']:g}" for v in vs)
        v0 = vs[0]
        lines.append(f"  {name:>6}: {len(vs)} x {v0['fov']:g} deg "
                     f"{v0['width']}x{v0['height']} "
                     f"({v0['width'] / v0['fov']:.1f} px/deg)  yaw {yaws}")
    if driver is not None:
        c = sphere_coverage(driver, plan)
        note = ""
        if c["horizon_band"] < 0.999:
            note = ("  <-- blind wedge at the lens boundary; the rig is moving, "
                    "so those directions are still imaged from other frames")
        lines.append(f"  covers {c['horizon_band']:.1%} of the horizon "
                     f"(+-10 deg band), {c['sphere']:.1%} of the full "
                     f"sphere{note}")
    return "\n".join(lines)
