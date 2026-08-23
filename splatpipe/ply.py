# SPDX-License-Identifier: Apache-2.0
"""Minimal 3DGS PLY reader/writer.

Just enough to filter and concatenate gaussian clouds without caring what the
per-gaussian properties are. Every field is carried through verbatim (means,
normals, SH coefficients, opacity, scales, quaternions), so this survives
trainer versions adding or renaming properties -- we never interpret them, we
only select rows.

Binary little-endian only, which is what every 3DGS trainer writes.
"""

from pathlib import Path

import numpy as np


def read(path: Path) -> tuple[np.ndarray, list[str]]:
    """-> (structured array of vertices, header lines)."""
    with open(path, "rb") as fh:
        header, lines = [], []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            text = line.decode("ascii", "replace").strip()
            header.append(line)
            lines.append(text)
            if text == "end_header":
                break

        fmt = next((l for l in lines if l.startswith("format")), "")
        if "binary_little_endian" not in fmt:
            raise ValueError(f"{path}: only binary_little_endian is supported ({fmt!r})")

        count, dtype, in_vertex = 0, [], False
        for text in lines:
            if text.startswith("element "):
                _, name, n = text.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    count = int(n)
            elif text.startswith("property ") and in_vertex:
                _, ptype, pname = text.split()
                dtype.append((pname, _PLY_TYPES[ptype]))
        data = np.frombuffer(fh.read(count * np.dtype(dtype).itemsize),
                             dtype=np.dtype(dtype), count=count)
    return data, lines


_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}
_NP_TO_PLY = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char",
              "u2": "ushort", "i2": "short", "u4": "uint", "i4": "int"}


def write(path: Path, data: np.ndarray):
    """Write a structured vertex array as a binary_little_endian PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    for name in data.dtype.names:
        head.append(f"property {_NP_TO_PLY[data.dtype[name].str.lstrip('<>|')]} {name}")
    head.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(np.ascontiguousarray(data).tobytes())


def xyz(data: np.ndarray) -> np.ndarray:
    """N x 3 gaussian means."""
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
