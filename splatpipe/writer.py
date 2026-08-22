# SPDX-License-Identifier: Apache-2.0
"""FrameWriter: the one place that materializes the ingest output layout.

    out/
      images/camK/NNNNNN.jpg    pinhole views (or cam0 passthrough for flat)
      frames.jsonl              one record per capture position
      geo.txt                   "camK/NNNNNN.jpg lat lon alt" per image file,
                                consumed by colmap model_aligner (--ref_is_gps)

EXIF GPS is written into every jpg so COLMAP's spatial matcher can use
position priors straight from the database.
"""

import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import piexif

from .reframe import Reframer


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
    def __init__(self, out_dir: Path, projection: str = "equirect",
                 views: list[dict] | None = None, jpeg_quality: int = 95):
        self.out = Path(out_dir)
        self.projection = projection
        self.quality = jpeg_quality
        # "equirect" reframes internally; "eac" gets pre-made views via
        # add_views (see ingest's .360 path); "flat" passes through.
        self.reframer = Reframer(views) if projection == "equirect" else None
        from .reframe import DEFAULT_VIEWS
        n_cams = len(views or DEFAULT_VIEWS) if projection in ("equirect", "eac") else 1
        for k in range(n_cams):
            (self.out / "images" / f"cam{k}").mkdir(parents=True, exist_ok=True)
        self._frames = open(self.out / "frames.jsonl", "w")
        self._geo = open(self.out / "geo.txt", "w")
        self.seq = 0

    def add(self, img_bgr: np.ndarray, lat: float | None, lon: float | None,
            alt: float | None, t: float | None = None, meta: dict | None = None):
        cams = self.reframer(img_bgr) if self.reframer else [img_bgr]
        self.add_views(cams, lat, lon, alt, t=t, meta=meta)

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
