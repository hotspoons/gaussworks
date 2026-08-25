# SPDX-License-Identifier: Apache-2.0
"""FrameWriter: the one place that materializes the ingest output layout.

    out/
      images/camK/NNNNNN.jpg    pinhole views (or cam0 passthrough for flat)
      coverage/camK.png         255 where camK's lens holds real pixels
      cameras.json              intrinsics + which lens each camK came from
      frames.jsonl              one record per capture position
      geo.txt                   "camK/NNNNNN.jpg lat lon alt" per image file,
                                consumed by colmap model_aligner (--ref_is_gps)

A camK is a (view, lens) pair, not just a view: two images of the same
direction taken through different lenses are different cameras, because they
have different optical centres. Keeping them apart is the whole point -- see
splatpipe/viewplan.py.

Coverage masks are written ONCE per camera rather than per image: the lens
geometry is fixed for the whole run, so a per-frame copy would be tens of
thousands of identical PNGs. `mask` intersects them with the per-frame vehicle
mask when it builds what COLMAP and the trainer actually read.

EXIF GPS is written into every jpg so COLMAP's spatial matcher can use
position priors straight from the database.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import piexif


def _deg_to_dms_rational(deg: float):
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = Fraction((deg - d) * 3600 - m * 60).limit_denominator(10000)
    return ((d, 1), (m, 1), (s.numerator, s.denominator))


def gps_exif(lat: float, lon: float, alt: float) -> bytes:
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
        piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100),
    }
    return piexif.dump({"GPS": gps})


class FrameWriter:
    def __init__(self, out_dir: Path, plan: list[dict] | None = None,
                 jpeg_quality: int = 95, profile: str = "",
                 coverage: list[np.ndarray] | None = None):
        self.out = Path(out_dir)
        self.quality = jpeg_quality
        self.plan = plan or []
        n_cams = max(1, len(self.plan))
        for k in range(n_cams):
            (self.out / "images" / f"cam{k}").mkdir(parents=True, exist_ok=True)

        # We SYNTHESISE these pinhole views, so their intrinsics are known
        # exactly. Recording them matters: left to guess, COLMAP assumes
        # fx = 1.2*max(w,h), which for a 100 deg view is ~3x the truth and the
        # incremental mapper cannot bootstrap from that (5/424 images
        # registered, observed on real capture).
        if self.plan:
            cams = []
            for k, v in enumerate(self.plan):
                f = 0.5 * v["width"] / math.tan(math.radians(v["fov"]) / 2)
                cams.append({"cam": f"cam{k}", "model": "PINHOLE",
                             "width": v["width"], "height": v["height"],
                             "fx": f, "fy": f,
                             "cx": v["width"] / 2, "cy": v["height"] / 2,
                             "yaw": v["yaw"], "pitch": v["pitch"], "fov": v["fov"],
                             "lens": v.get("lens", 0),
                             "lens_name": v.get("lens_name", "")})
            (self.out / "cameras.json").write_text(json.dumps(cams, indent=1))
        if profile:
            (self.out / "profile.txt").write_text(profile + "\n")

        self.coverage_frac: list[float] = []
        if coverage:
            cov_dir = self.out / "coverage"
            cov_dir.mkdir(exist_ok=True)
            for k, m in enumerate(coverage):
                self.coverage_frac.append(float(np.mean(m)))
                if m.all():
                    continue      # fully covered: no file, nothing to intersect
                cv2.imwrite(str(cov_dir / f"cam{k}.png"),
                            (m.astype(np.uint8) * 255))

        self._frames = open(self.out / "frames.jsonl", "w")
        self._geo = open(self.out / "geo.txt", "w")
        self.seq = 0

    def add(self, img_bgr: np.ndarray, lat: float | None, lon: float | None,
            alt: float | None, t: float | None = None, meta: dict | None = None):
        self.add_views([img_bgr], lat, lon, alt, t=t, meta=meta)

    def add_views(self, cams: list[np.ndarray], lat: float | None, lon: float | None,
                  alt: float | None, t: float | None = None, meta: dict | None = None):
        exif = gps_exif(lat, lon, alt or 0.0) if lat is not None else None
        rec = {"seq": self.seq, "t": t, "lat": lat, "lon": lon, "alt": alt,
               "images": {}, **(meta or {})}
        for k, cam_img in enumerate(cams):
            rel = f"cam{k}/{self.seq:06d}.jpg"
            path = self.out / "images" / rel
            ok, buf = cv2.imencode(".jpg", cam_img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                raise RuntimeError(f"jpeg encode failed for {rel}")
            path.write_bytes(buf.tobytes())
            if exif is not None:
                piexif.insert(exif, str(path))
                self._geo.write(f"{rel} {lat} {lon} {alt or 0.0}\n")
            rec["images"][f"cam{k}"] = rel
        self._frames.write(json.dumps(rec) + "\n")
        self.seq += 1

    def close(self):
        self._frames.close()
        self._geo.close()
        # an all-flat no-GPS run produces an empty geo.txt; drop it
        geo = self.out / "geo.txt"
        if geo.stat().st_size == 0:
            geo.unlink()
