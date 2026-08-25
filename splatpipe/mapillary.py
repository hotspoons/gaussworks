# SPDX-License-Identifier: Apache-2.0
"""Fetch car-mounted 360 sequences from Mapillary as pipeline input.

Mapillary serves original-resolution equirects (much of it GoPro Max capture)
with GPS — real street-level 360 with geotags before our own camera arrives.
Token: mapillary.com/dashboard/developers (free), env MAPILLARY_TOKEN.

Licensing: Mapillary imagery is CC BY-SA 4.0 (plus Mapillary's ToS). Use it
for pipeline development and testing; ShareAlike applies to derivatives, so
distributable game assets must come from our own capture instead. See
THIRD_PARTY.md.
"""

import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

from . import viewplan
from .drivers import get_driver
from .profiles import get as get_profile
from .eac import view_dirs
from .writer import FrameWriter

API = "https://graph.mapillary.com/images"
FIELDS = "id,captured_at,thumb_original_url,computed_geometry,computed_altitude,is_pano,sequence"


def _list_images(bbox: str, token: str, max_pages: int = 40) -> list[dict]:
    params = {"access_token": token, "fields": FIELDS, "bbox": bbox, "limit": 500}
    url, out = API, []
    for _ in range(max_pages):
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        out += [d for d in body.get("data", []) if d.get("is_pano")]
        nxt = body.get("paging", {}).get("next")
        if not nxt:
            break
        url, params = nxt, {}  # next URL carries the cursor + token
    return out


def fetch(bbox: str, out: Path, token: str | None = None, views: list[dict] | None = None,
          max_images: int = 2000, min_seq_len: int = 50) -> Path:
    token = token or os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("set MAPILLARY_TOKEN (mapillary.com/dashboard/developers)")

    images = _list_images(bbox, token)
    seqs: dict[str, list[dict]] = defaultdict(list)
    for d in images:
        seqs[d.get("sequence", "?")].append(d)
    ranked = sorted((s for s in seqs.values() if len(s) >= min_seq_len),
                    key=len, reverse=True)
    print(f"[mapillary] bbox={bbox}: {len(images)} pano images, "
          f"{len(ranked)} sequences >= {min_seq_len} frames")
    if not ranked:
        raise SystemExit("no usable 360 sequences in this bbox — widen it or lower --min-seq-len")

    out.mkdir(parents=True, exist_ok=True)
    # Mapillary serves stitched equirect, so the "equirect-360" profile is the
    # honest description: one (fictional) optical centre, seam already baked in.
    driver = get_driver(get_profile("equirect-360"))
    plan = viewplan.plan_views(driver, {"views": views} if views else {})
    dirs = [view_dirs(v["width"], v["height"], v["fov"], v["yaw"], v["pitch"])
            for v in plan]
    writer = FrameWriter(out, plan=plan, profile="equirect-360")
    kept = 0
    try:
        for seq in ranked:
            if kept >= max_images:
                break
            seq.sort(key=lambda d: d.get("captured_at", 0))
            for d in seq:
                if kept >= max_images:
                    break
                url = d.get("thumb_original_url")
                geom = (d.get("computed_geometry") or {}).get("coordinates")
                if not url or not geom:
                    continue
                for attempt in range(3):
                    try:
                        img_bytes = requests.get(url, timeout=120).content
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            img_bytes = None
                        time.sleep(2)
                if not img_bytes:
                    continue
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                lon, lat = geom
                cams = [driver.sample(("view", k), dd, [img], plan[k]["lens"])
                        for k, dd in enumerate(dirs)]
                writer.add_views(cams, lat, lon, d.get("computed_altitude") or 0.0,
                           t=(d.get("captured_at", 0)) / 1000.0,
                           meta={"mapillary_id": d["id"], "sequence": d.get("sequence")})
                kept += 1
                if kept % 100 == 0:
                    print(f"[mapillary] {kept} frames written")
    finally:
        writer.close()
    print(f"[mapillary] done: {kept} frames -> {out}")
    return out
