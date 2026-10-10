#!/usr/bin/env python3
"""Rename existing captures' envelope files to envelope.json, in place.

    python scripts/rename-envelope.py [--dry-run] DIR [DIR ...]

Since 2026-10-10 every writer names a chunk's or a world's capture envelope `envelope.json`
(splatpipe/corridor.py, ENVELOPE_FILE). Captures made before that carry the old name,
`corridor.json`. splatpipe itself still reads the old name where the new one is absent, so nothing
here NEEDS this script; a deployment of the game that reads the new name only does. Run it over a
capture before it moves to such a deployment (or before you publish its world directory there).

DIR is anything above the files: a chunk, a world, a run's whole output, a volume of captures.
Under it, every `corridor.json` that is an envelope (a JSON object with `passes`) becomes
`envelope.json` beside it, and every `world.json` that names the old file
(`"corridor": "corridor.json"`) is pointed at the new one. Nothing else is touched:
`corridor_gps.json` (the GPS envelope kept beside an SfM one) and every other file keep their names.

Where both names already exist in one directory, the old file is left alone and reported: the new
one is what every reader takes.
"""

import argparse
import json
import sys
from pathlib import Path

OLD, NEW = "corridor.json", "envelope.json"


def is_envelope(p: Path) -> bool:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and "passes" in doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="say what would change, change nothing")
    a = ap.parse_args(argv)
    renamed = pointed = clashes = 0
    for root in a.dirs:
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        for old in sorted(root.rglob(OLD)):
            if not is_envelope(old):
                continue
            new = old.with_name(NEW)
            if new.exists():
                print(f"both names in {old.parent}: left {OLD} alone ({NEW} is what readers take)")
                clashes += 1
                continue
            print(f"{'would rename' if a.dry_run else 'renamed'} {old} -> {NEW}")
            if not a.dry_run:
                old.rename(new)
            renamed += 1
        for world in sorted(root.rglob("world.json")):
            try:
                doc = json.loads(world.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict) or doc.get("corridor") != OLD:
                continue
            print(f"{'would point' if a.dry_run else 'pointed'} {world} at {NEW}")
            if not a.dry_run:
                doc["corridor"] = NEW
                tmp = world.with_name(world.name + ".tmp")
                tmp.write_text(json.dumps(doc, indent=1))
                tmp.replace(world)
            pointed += 1
    verb = "would change" if a.dry_run else "changed"
    print(f"{verb}: {renamed} envelope file(s), {pointed} world.json; {clashes} director(ies) had both names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
