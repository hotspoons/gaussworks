# SPDX-License-Identifier: Apache-2.0
"""A claim-based work queue on shared storage.

Static sharding (`chunks[RANK::WORLD_SIZE]`) is fine until a chunk is slow or a
worker dies: then one rank holds the whole run's tail while the others idle, and
anything the dead rank owned is simply never done. Instead every worker loops
"claim the next unclaimed item, do it, mark it" until nothing is claimable, so
work self-balances and a crashed worker's item is picked up by someone else.

The lock is `os.mkdir`, which is atomic and fails if the directory exists -- the
one primitive that behaves on NFS and CephFS alike. State is plain files, so
`ls` tells you everything and a `rm` un-sticks anything by hand:

    <chunks>/.queue/<stage>/<item>.lock/     claimed (holds `owner`, mtime = heartbeat)
    <chunks>/.queue/<stage>/<item>.done      finished
    <chunks>/.queue/<stage>/<item>.failed    attempt log; retried until max_attempts
"""

import os
import socket
import time
from pathlib import Path


class WorkQueue:
    def __init__(self, root: Path, stage: str, max_attempts: int = 2,
                 stale_s: float = 3600.0):
        self.dir = Path(root) / ".queue" / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.stale_s = stale_s
        self.worker = f"{socket.gethostname()}:{os.environ.get('RANK', '0')}"

    # --- state helpers -------------------------------------------------------

    def _lock(self, name: str) -> Path:
        return self.dir / f"{name}.lock"

    def _attempts(self, name: str) -> int:
        path = self.dir / f"{name}.failed"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def _claim(self, name: str) -> bool:
        """Try to take ownership. Steals locks whose heartbeat went stale."""
        lock = self._lock(name)
        try:
            lock.mkdir()
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime < self.stale_s:
                    return False
            except FileNotFoundError:
                return False  # released while we looked; next pass picks it up
            print(f"[queue] reclaiming stale lock: {name}")
            (lock / "owner").write_text(self.worker)  # refresh, then take over
        else:
            (lock / "owner").write_text(self.worker)
        return True

    def _release(self, name: str):
        lock = self._lock(name)
        for child in lock.glob("*"):
            child.unlink(missing_ok=True)
        lock.rmdir()

    def heartbeat(self, name: str):
        """Long items should call this so their lock is not judged stale."""
        os.utime(self._lock(name), None)

    # --- driving -------------------------------------------------------------

    def run(self, items, fn, key=lambda i: getattr(i, "name", str(i))):
        """Claim and process until nothing is claimable. Returns (done, failed).

        `fn(item)` runs the work; raising marks the item failed (and eligible
        for retry by any worker until max_attempts).
        """
        done, failed = [], []
        while True:
            claimed = None
            for item in items:
                name = key(item)
                if (self.dir / f"{name}.done").exists():
                    continue
                if self._attempts(name) >= self.max_attempts:
                    continue
                if self._claim(name):
                    claimed = (item, name)
                    break
            if claimed is None:
                break

            item, name = claimed
            try:
                fn(item)
                (self.dir / f"{name}.done").write_text(
                    f"{self.worker} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                done.append(name)
            except Exception as exc:                       # noqa: BLE001
                with open(self.dir / f"{name}.failed", "a") as fh:
                    fh.write(f"{self.worker} {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                             f"{type(exc).__name__}: {exc}\n".replace("\n", " ") + "\n")
                failed.append(name)
                print(f"[queue] {name}: FAILED ({type(exc).__name__}: {exc})")
            finally:
                self._release(name)
        return done, failed

    # --- reporting -----------------------------------------------------------

    def status(self, items, key=lambda i: getattr(i, "name", str(i))) -> dict:
        out = {"done": [], "running": [], "failed": [], "pending": []}
        for item in items:
            name = key(item)
            if (self.dir / f"{name}.done").exists():
                out["done"].append(name)
            elif self._lock(name).exists():
                out["running"].append(name)
            elif self._attempts(name) >= self.max_attempts:
                out["failed"].append(name)
            else:
                out["pending"].append(name)
        return out
