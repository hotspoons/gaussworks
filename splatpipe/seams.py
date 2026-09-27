"""Do neighbouring chunks agree about the height of the same road?

Every chunk is aligned independently, against its own GPS priors, and GPS
altitude is the noisiest component of a consumer fix. So two chunks covering
the same stretch of road can be levelled differently, and the seam between
them becomes a step in the world -- which is what a driver actually hits, and
what a screenshot of splats floating over the road surface is showing.

Chunks overlap by design (the `overlap_m` halo), so wherever two chunks' camera
paths pass through the same place their heights MUST agree. Disagreement there
is not noise to be averaged away; it is the defect, localised to a pair.

This is deliberately not PSNR. PSNR is dominated by large smooth regions and
barely moves when a chunk is levelled half a metre wrong, so a world can gain
PSNR while getting worse to drive. Measured on arrowhead vs gosheff-hq over the
same ground, PSNR separated the two worlds by 0.7 dB while the worst seam
separated them by 3.4 m.
"""
import json
import math
from pathlib import Path

import numpy as np

# How close two corridor points must be, in metres, to count as "the same
# place". The corridor is sampled along the driven line, so this wants to be a
# little larger than that sampling or genuinely coincident passes are missed.
CELL_M = 6.0
# Below this many coincident points a pair is a glancing corner contact, not a
# shared stretch of road, and its median is not worth reporting.
MIN_PAIRS = 20


def _footprint(chunk: Path):
    """The chunk's bounds as (lat_min, lat_max, lon_min, lon_max)."""
    m = json.loads((chunk / "meta.json").read_text())
    o, b = m["origin"], m["bounds_enu_m"]
    lat = math.radians(o["lat"])
    m_lat = 111132.92 - 559.82 * math.cos(2 * lat)
    m_lon = 111412.84 * math.cos(lat)
    ll = lambda e, n: (o["lat"] + n / m_lat, o["lon"] + e / m_lon)
    p, q = ll(b[0], b[1]), ll(b[2], b[3])
    return (min(p[0], q[0]), max(p[0], q[0]), min(p[1], q[1]), max(p[1], q[1]))


def _overlap_fraction(a, b) -> float:
    """How much of footprint `a` lies inside footprint `b`."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    wo, wi = max(a[2], b[2]), min(a[3], b[3])
    if hi <= lo or wi <= wo:
        return 0.0
    return ((hi - lo) * (wi - wo)) / ((a[1] - a[0]) * (a[3] - a[2]))


def _hull(chunks_dir: Path):
    boxes = [_footprint(d) for d in sorted(chunks_dir.glob("chunk_*"))
             if (d / "meta.json").exists()]
    if not boxes:
        raise SystemExit(f"no chunks with a meta.json under {chunks_dir}")
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def measure(chunks_dir: Path, within=None, min_inside=0.30):
    """Median |dz| per overlapping chunk pair.

    `within` restricts to chunks at least `min_inside` inside another world's
    footprint. Comparing a whole survey against a bake of one street otherwise
    credits the small bake for merely not containing the survey's sparse
    fringes, which is where the bad seams live.
    """
    keep = None
    if within is not None:
        hull = _hull(within)
        keep = {d.name for d in sorted(chunks_dir.glob("chunk_*"))
                if (d / "meta.json").exists()
                and _overlap_fraction(_footprint(d), hull) > min_inside}

    tracks = {}
    for d in sorted(chunks_dir.glob("chunk_*")):
        if keep is not None and d.name not in keep:
            continue
        p = d / "corridor.json"
        if not p.exists():
            continue
        pts = [q for ps in json.loads(p.read_text()).get("passes") or []
               for q in ps["points"]]
        if pts:
            tracks[d.name] = np.asarray(pts, dtype=float)
    if not tracks:
        raise SystemExit(f"no chunk under {chunks_dir} has a corridor.json to compare")

    grid = {}
    for name, P in tracks.items():
        for x, y, z in P:
            grid.setdefault((int(x // CELL_M), int(y // CELL_M)), []).append((name, x, y, z))

    pairs = {}
    for items in grid.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[0] == b[0]:
                    continue
                if math.dist((a[1], a[2]), (b[1], b[2])) > CELL_M:
                    continue
                pairs.setdefault(tuple(sorted((a[0], b[0]))), []).append(abs(a[3] - b[3]))

    rows = []
    for (a, b), v in pairs.items():
        if len(v) < MIN_PAIRS:
            continue
        arr = np.asarray(v)
        rows.append((a, b, len(v), float(np.median(arr)), float(np.percentile(arr, 90))))
    rows.sort(key=lambda r: -r[3])
    return rows, len(tracks)


def report(chunks_dir: Path, within=None, fail_over=None) -> int:
    rows, n = measure(chunks_dir, within=within)
    scope = f" (restricted to the footprint of {within})" if within else ""
    print(f"{n} chunks with a corridor{scope}; {len(rows)} overlapping pairs\n", flush=True)
    if not rows:
        # Not a pass. No overlapping pairs means the halo is too small or the
        # chunks do not actually touch, and either way nothing was checked.
        print("NOTHING MEASURED: no chunk pair shares enough road to compare.", flush=True)
        return 2
    print(f'{"chunk A":16} {"chunk B":16} {"pairs":>6} {"median dz":>10} {"p90 dz":>8}  verdict')
    for a, b, k, med, p90 in rows:
        v = "agree" if med < 1.0 else ("marginal" if med < 3.0 else "DISAGREE")
        print(f"{a:16} {b:16} {k:>6} {med:>10.2f} {p90:>8.2f}  {v}", flush=True)
    meds = [r[3] for r in rows]
    worst = max(meds)
    print(f"\nmedian-of-medians {np.median(meds):.2f} m, worst {worst:.2f} m")
    bad = sorted({c for a, b, _, med, _ in rows if med >= 3.0 for c in (a, b)})
    print(f"chunks in a >3 m disagreement: {len(bad)} {bad if bad else ''}")
    if fail_over is not None and worst > fail_over:
        print(f"\nFAIL: worst seam {worst:.2f} m exceeds --fail-over {fail_over:.2f} m", flush=True)
        return 1
    return 0
