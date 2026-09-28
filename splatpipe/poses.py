# SPDX-License-Identifier: Apache-2.0
"""Stage 3: per-chunk camera poses via COLMAP (+GLOMAP), geo-aligned to ENU.

Spatial matching uses the GPS priors we wrote into EXIF, which handles
multi-camera rigs and both-direction passes without sequential assumptions.
Work is claimed from a shared queue (see queue.py), so any number of workers
on any number of nodes can be pointed at the same chunk directory.
"""

import functools
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from .workqueue import WorkQueue


@functools.lru_cache(maxsize=1)
def _ba_gpu_available() -> bool:
    """Can bundle adjustment run on the GPU?

    COLMAP exposes --Mapper.ba_use_gpu, but it only does anything if the CERES
    it links was built with CUDA -- and the distro libceres is not. Passing the
    flag blindly on such a build either does nothing or fails deep inside the
    mapper, hours in. So probe the linkage instead of assuming: cuSOLVER /
    cuSPARSE in colmap's dependency list is the tell.

    (SIFT extraction and matching are a separate story and use the GPU on any
    CUDA-enabled COLMAP, which is where most of the pre-mapper time goes.)
    """
    exe = shutil.which("colmap")
    if not exe:
        return False
    try:
        out = subprocess.run(["ldd", exe], capture_output=True,
                             check=False).stdout.decode().lower()
    except OSError:
        return False
    return any(lib in out for lib in ("libcusolver", "libcusparse"))


def _run(cmd: list[str], cwd: Path | None = None):
    # flush: COLMAP writes straight to the fd, so unflushed python prints land
    # in the log long after the subprocess output they were meant to label
    print("[poses] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def list_chunks(chunks_dir: Path) -> list[Path]:
    return sorted(p for p in chunks_dir.glob("chunk_*") if (p / "images").is_dir())


def _largest_model(models_dir: Path) -> Path:
    models = [p for p in models_dir.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError(f"no model produced under {models_dir}")
    return max(models, key=lambda p: (p / "images.bin").stat().st_size
               if (p / "images.bin").exists() else 0)


def _rig_size(chunk: Path) -> tuple[int, int]:
    """(cameras, passes) — how many images share one capture position.

    meta.json's `passes` is the list of source VIDEOS, which undercounts: one
    chapter can drive through a cell several times (the street run passes the
    house three times in GS010002 alone). corridor.json already splits the
    chunk's frames into contiguous drives by gap and by video, so its pass
    count is the honest one; take the larger of the two.
    """
    cams_path, meta_path = chunk / "cameras.json", chunk / "meta.json"
    n_cams = len(json.loads(cams_path.read_text())) if cams_path.exists() else 1
    n_pass = 1
    if meta_path.exists():
        n_pass = max(1, len(json.loads(meta_path.read_text()).get("passes") or []))
    corridor = chunk / "corridor.json"
    if corridor.exists():
        n_pass = max(n_pass, len(json.loads(corridor.read_text()).get("passes") or []))
    return n_cams, n_pass


def _db_reusable(db: Path, chunk: Path) -> bool:
    """Does this database already hold features and matches for these images?

    Extraction and matching are the expensive GPU stages (73 min on a
    3,870-image chunk) and they depend only on the images and masks -- not on
    which mapper runs afterwards. Wiping the database to retry a FAILED MAPPER,
    or to switch from the incremental mapper to GLOMAP, throws away an hour of
    finished work for no reason. So reuse it when it is complete and matches
    the images on disk, and start clean otherwise.
    """
    if not db.exists():
        return False
    n_disk = sum(1 for _ in (chunk / "images").glob("*/*.jpg"))
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            n_img = con.execute("SELECT count(*) FROM images").fetchone()[0]
            n_kp = con.execute("SELECT count(*) FROM keypoints").fetchone()[0]
            n_pairs = con.execute(
                "SELECT count(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as exc:
        print(f"[poses] {chunk.name}: database unreadable ({exc}); rebuilding",
              flush=True)
        return False
    if n_img != n_disk or n_kp < n_img or n_pairs == 0:
        print(f"[poses] {chunk.name}: database incomplete "
              f"({n_img} images vs {n_disk} on disk, {n_kp} with keypoints, "
              f"{n_pairs} verified pairs); rebuilding", flush=True)
        return False
    print(f"[poses] {chunk.name}: reusing features and matches "
          f"({n_img} images, {n_pairs} verified pairs) -- mapping only",
          flush=True)
    return True


def _maybe_vocab(chunk: Path, db: Path, gpu: str, loop_closure: str) -> None:
    """Run retrieval matching once per database, whether fresh or reused."""
    if loop_closure != "vocab":
        return
    marker = chunk / ".loop_closure_vocab"
    if marker.exists():
        print(f"[poses] {chunk.name}: vocab loop closure already in the database",
              flush=True)
        return
    _vocab_tree_match(db, gpu)
    marker.write_text("vocab\n")


def _pass_connectivity(chunk: Path, db: Path, weak_ratio: float = 0.05) -> None:
    """Print the verified-pair matrix between corridor passes; warn on a weak one.

    A locality fence is not a road. On the street run a 150 m fence around the
    house took in a pass on a *different* street 89 m away: 179 + 81 verified
    pairs to the other passes, against 10,304 between two passes of the same
    road. GLOMAP produces exactly one model, so it hung that pass on the few
    pairs it had and bent the whole reconstruction (one flat road came out at
    z = 35 / 39 / 48 / 38 m per pass). The matrix makes that visible BEFORE
    the mapper spends an hour on it. Diagnostic only: splitting is a chunking
    decision, so this prints what to do rather than doing it.
    """
    cor_path = chunk / "corridor.json"
    if not cor_path.exists():
        return
    passes = json.loads(cor_path.read_text()).get("passes") or []
    if len(passes) < 2:
        return
    ranges = [tuple(p["seq_range"]) for p in passes]

    def pass_of(name: str) -> int:
        try:
            seq = int(Path(name).stem)
        except ValueError:
            return -1
        for i, (a, b) in enumerate(ranges):
            if a <= seq <= b:
                return i
        return -1

    n = len(ranges)
    mat = [[0] * n for _ in range(n)]
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            owner = {i: pass_of(name) for i, name in
                     con.execute("SELECT image_id, name FROM images")}
            for (pid,) in con.execute(
                    "SELECT pair_id FROM two_view_geometries WHERE rows > 15"):
                i2 = pid % 2147483647
                i1 = (pid - i2) // 2147483647
                a, b = owner.get(i1, -1), owner.get(i2, -1)
                if a < 0 or b < 0:
                    continue
                a, b = min(a, b), max(a, b)
                mat[a][b] += 1
                if a != b:
                    mat[b][a] += 1
        finally:
            con.close()
    except sqlite3.Error as exc:
        print(f"[poses] {chunk.name}: connectivity check skipped ({exc})", flush=True)
        return
    print(f"[poses] {chunk.name}: verified pairs (>15 inliers) by corridor pass:", flush=True)
    for i, p in enumerate(passes):
        cells = " ".join(f"{mat[i][j]:7d}" for j in range(n))
        print(f"[poses]   pass {i} {p.get('video')} seq {ranges[i][0]}-{ranges[i][1]}: {cells}",
              flush=True)
    for i in range(n):
        cross = sum(mat[i][j] for j in range(n) if j != i)
        if mat[i][i] and cross < weak_ratio * mat[i][i]:
            print(f"[poses] WARNING {chunk.name}: pass {i} shares only {cross} verified "
                  f"pairs with the other passes ({mat[i][i]} within itself). It is "
                  f"probably a different road inside the fence. A global mapper will "
                  f"place it by those few pairs and can bend the whole model; give it "
                  f"its own chunk (smaller --cell-m or --radius-m, or prune it from the "
                  f"database) before mapping.", flush=True)

    # The per-pass test above asks "is THIS pass weakly attached to the rest",
    # which misses the case that matters most: passes that split into two or
    # more internally healthy groups with nothing between them. Each pass is
    # then strongly connected to its own group and raises no warning, while the
    # chunk as a whole is two roads that no image sees across -- and a global
    # mapper reconstructs ONE component and silently drops the other, or floats
    # them relative to each other.
    #
    # Seen on this capture: a cell with passes 0,1 sharing 10,042 pairs and
    # passes 2,3 sharing 1,268, and exactly zero between the two groups.
    # Nothing warned.
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = set(), [i]
        while stack:                       # connected components over pass pairs
            a = stack.pop()
            if a in comp:
                continue
            comp.add(a)
            stack.extend(j for j in range(n)
                         if j not in comp and (mat[a][j] or mat[j][a]))
        seen |= comp
        groups.append(sorted(comp))
    if len(groups) > 1:
        desc = "; ".join("passes " + ",".join(str(j) for j in g) for g in groups)
        print(f"[poses] WARNING {chunk.name}: the passes form {len(groups)} "
              f"DISCONNECTED groups with no verified pairs between them "
              f"({desc}). These are separate roads inside one cell, not one "
              f"road driven twice. A global mapper will reconstruct one group "
              f"and drop or float the others; split the cell (smaller "
              f"--cell-m) so each road gets its own chunk.", flush=True)


# Two tree formats exist. Our COLMAP 3.11.1 is a FLANN build and reads the
# classic file; the faiss-format file (for faiss builds, 3.12+) makes it die
# with std::bad_alloc on load. Both are on the 3.11.1 release page.
VOCAB_TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words256K.bin"


def _vocab_tree_match(db: Path, gpu: str, num_images: int = 50,
                      max_num_features: int = 500) -> None:
    """Loop-closure matching by image retrieval: pairs that LOOK alike, wherever
    GPS says they are.

    Spatial matching trusts the GPS prior in every image. On the street run the
    first ~100 m of GPS were 15-19 m off (cold start under canopy), so the
    outbound pass's cross-pass "neighbours" were the wrong stretch of road:
    2,110 verified pairs to the homeward pass against 18,316 within itself,
    and GLOMAP had nothing to pin the pass with -- the model came out bent by
    9 m between passes at 0.8 px reprojection. Retrieval does not care where
    the GPS thinks an image is. Matches accumulate in the database, so this
    is additive to spatial + sequential. The tree lives on the PVC and is
    fetched once ($COLMAP_VOCAB_TREE overrides the path).

    max_num_features caps the features used for RETRIEVAL only (matching
    still uses every descriptor in the database). With the full ~10k per
    image, 3,324 images indexed for 70 minutes and then the pair generation
    aborted with std::bad_alloc; at 500 the whole pass took 25 minutes and
    finished. Retrieval does not need more than that to find the same street.

    WHERE THIS MAKES THINGS WORSE, measured rather than argued. The win above
    came from a RURAL back road, where two stretches that look alike are
    usually the same stretch. A SUBDIVISION is the opposite: the houses were
    built from a handful of plans, the mailboxes and lamp posts repeat, and
    retrieval cannot tell one cul-de-sac from the next.

    On the Arrowhead Farms capture, one 1,908-image chunk (2 roads, GPS fine
    at median DOP 4.3) ran with spatial+sequential and then again with vocab
    added to the same database:

        spatial + sequential      alignment error  25.5 mean / 28.1 median
        + vocab retrieval         alignment error  7961 mean / 42.9 median
                                  1554/1908 images in the connected component
                                  259/318 positions registered (was 318/318)

    The cross-pass pair counts looked BETTER with vocab (14,741 between passes
    against 10,042). They were false matches between different houses of the
    same design, and they tore the reconstruction apart -- a mean three orders
    of magnitude above the median is cameras thrown kilometres away.

    So this flag is not a general "make matching stronger" knob. Reach for it
    when GPS priors are untrustworthy AND the scene is visually distinctive,
    and re-measure after: registered fraction and connected-component size
    tell the truth, cross-pass pair COUNTS do not.
    """
    tree = Path(os.environ.get("COLMAP_VOCAB_TREE",
                               "/workspace/opt/colmap-vocab/vocab_tree_flickr100K_words256K.bin"))
    if not tree.exists():
        tree.parent.mkdir(parents=True, exist_ok=True)
        print(f"[poses] fetching vocabulary tree -> {tree}", flush=True)
        _run(["curl", "-fL", "--retry", "5", "-o", str(tree), VOCAB_TREE_URL])
    _run(["colmap", "vocab_tree_matcher", "--database_path", str(db),
          "--VocabTreeMatching.vocab_tree_path", str(tree),
          "--VocabTreeMatching.num_images", str(num_images),
          "--VocabTreeMatching.max_num_features", str(max_num_features),
          "--SiftMatching.use_gpu", gpu])


def solve_chunk(chunk: Path, matcher: str = "spatial", align: bool = True,
                use_gpu: bool = True, spatial_radius: int = 4,
                refresh: bool = False, mapper: str = "auto",
                loop_closure: str = "none"):
    db = chunk / "colmap.db"
    sparse = chunk / "sparse"
    if (sparse / "0").exists():
        print(f"[poses] {chunk.name}: sparse/0 exists, skipping", flush=True)
        return
    reuse = not refresh and _db_reusable(db, chunk)
    if db.exists() and not reuse:
        db.unlink()  # stale partial runs poison the database; start clean
    sparse.mkdir(exist_ok=True)
    shutil.rmtree(sparse / "raw", ignore_errors=True)   # half-built model

    gpu = "1" if use_gpu else "0"
    if reuse:
        # Loop closure has to be reachable on a REUSED database, or it cannot
        # be used as a remedy. It is normally wanted precisely when a chunk has
        # already been mapped once and come out badly -- and the alternative,
        # --refresh, throws away the features and matches (73 min on a 3,870
        # image chunk) to add matching that is purely additive to them.
        #
        # The marker is what makes it idempotent: vocab_tree_matcher would
        # otherwise re-run on every remap of that chunk, which costs the same
        # again and adds nothing the second time.
        _maybe_vocab(chunk, db, gpu, loop_closure)
        _pass_connectivity(chunk, db)
        _map_and_align(chunk, db, sparse, align, mapper)
        return

    extract = ["colmap", "feature_extractor",
               "--database_path", str(db), "--image_path", str(chunk / "images"),
               "--ImageReader.camera_model", "PINHOLE",
               "--ImageReader.single_camera_per_folder", "1",
               "--SiftExtraction.use_gpu", gpu]
    cams_path = chunk / "cameras.json"
    if cams_path.exists():
        cams = json.loads(cams_path.read_text())
        c = cams[0]
        key = ("fx", "fy", "cx", "cy", "width", "height")
        # camera_params is a single string applied to every folder, so it is
        # only safe when the view plan gave every camera the same intrinsics --
        # which auto planning does. A hand-written mixed-FOV view list would
        # silently get the first view's focal length imposed on all of them.
        if all(tuple(x[k] for k in key) == tuple(c[k] for k in key) for x in cams):
            extract += ["--ImageReader.camera_params",
                        f"{c['fx']:.6f},{c['fy']:.6f},{c['cx']:.6f},{c['cy']:.6f}"]
            print(f"[poses] intrinsics from cameras.json: fx={c['fx']:.1f} "
                  f"cx={c['cx']:.1f} ({c['width']}x{c['height']}, {c['fov']} deg)",
                  flush=True)
        else:
            print("[poses] WARNING: cameras.json holds mixed intrinsics, so they "
                  "cannot be pinned per folder; COLMAP will guess fx = 1.2*max(w,h) "
                  "and registration will suffer. Use one FOV/size across views.",
                  flush=True)
    if (chunk / "masks").is_dir():
        # keeps features off the capture vehicle, which is rigid in the camera
        # frame and would otherwise drag every pose toward itself
        extract += ["--ImageReader.mask_path", str(chunk / "masks")]
    _run(extract)

    if matcher == "spatial" and (chunk / "geo.txt").exists():
        n_cams, n_pass = _rig_size(chunk)
        # SIZE THE NEIGHBOURHOOD TO THE RIG, or spatial matching quietly
        # collapses. Every virtual view of one capture position carries that
        # position's GPS, and every pass down the road revisits it, so
        # n_cams * n_pass images sit at essentially the same coordinate --
        # 6 x 3 = 18 here. A fixed 32 neighbours is then +-1 position of road,
        # the match graph becomes a razor-thin chain, and the incremental
        # mapper builds ONE local component and stops: observed as a
        # contiguous block of 100 of 331 positions registered, every camera at
        # the identical rate.
        neighbors = max(32, n_cams * n_pass * spatial_radius * 2)
        print(f"[poses] {chunk.name}: {n_cams} cams x {n_pass} pass(es) at each "
              f"position -> {neighbors} spatial neighbours "
              f"(~{spatial_radius} positions either side)", flush=True)
        _run(["colmap", "spatial_matcher",
              "--database_path", str(db),
              "--SpatialMatching.ignore_z", "1",
              "--SpatialMatching.max_num_neighbors", str(neighbors),
              "--SiftMatching.use_gpu", gpu])
        # Then the long chain along the road. Image names are
        # camK/NNNNNN.jpg, so name order is per-camera and per-pass in capture
        # order: sequential matching adds exactly the reach spatial matching
        # cannot afford, at a fraction of the pairs. Matches accumulate in the
        # same database, so this is additive.
        _run(["colmap", "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", "15",
              "--SiftMatching.use_gpu", gpu])
        _maybe_vocab(chunk, db, gpu, loop_closure)
    elif matcher == "exhaustive":
        # small chunks: all-pairs matching links the rig's cam folders, which
        # sequential (name-ordered) matching never crosses
        _run(["colmap", "exhaustive_matcher",
              "--database_path", str(db),
              "--SiftMatching.use_gpu", gpu])
    else:
        _run(["colmap", "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", "15",
              "--SiftMatching.use_gpu", gpu])

    _pass_connectivity(chunk, db)
    _map_and_align(chunk, db, sparse, align, mapper)


def _pick_mapper(mapper: str) -> str:
    """Resolve 'auto' to a mapper that is actually installed.

    Explicit, because "whatever is on PATH" is how a pod silently ran the slow
    CPU mapper for a week -- and then, once GLOMAP was installed, how all four
    chunks silently failed instead. Both are configuration facts and should be
    stated, not discovered.
    """
    if mapper != "auto":
        return mapper
    return "glomap" if shutil.which("glomap") else "colmap"


def _map_and_align(chunk: Path, db: Path, sparse: Path, align: bool,
                   mapper: str = "auto"):
    raw = sparse / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mapper = _pick_mapper(mapper)
    if mapper == "glomap" and not shutil.which("glomap"):
        raise SystemExit("[poses] --mapper glomap requested but glomap is not "
                         "on PATH (build it with scripts/pod-build-stack.sh)")
    if mapper == "glomap":
        # Global SfM: solves rotation averaging then global positioning for all
        # images at once. The incremental mapper below instead adds images one
        # at a time with repeated bundle adjustment -- correct, but it is the
        # single longest stage in the pipeline by a wide margin (4.5 h on a
        # 3,870-image chunk, versus 73 min for GPU feature extraction and
        # matching combined) and it is CPU-bound with no CUDA path at all.
        # VERSION SKEW WARNING. GLOMAP 1.2.0 vendors a COLMAP from Oct 2025,
        # which models a multi-camera setup as a *rig*. Opening a database
        # written by COLMAP 3.11.1 migrates in empty rigs/rig_sensors/frames
        # tables, and GLOMAP then aborts at the very end -- after a complete,
        # successful reconstruction -- writing the model out:
        #     Check failed: existing_rig.RefSensorId() == rig.RefSensorId()
        # Observed on all four street chunks, ~2 h each, wasted. Either match
        # the COLMAP generation across the toolchain, or use GLOMAP 1.0.0.
        # The real fix is to populate the rig properly: our 6 virtual cameras
        # have EXACTLY known relative orientations (we synthesised them from
        # the view plan), so they are a genuine rigid rig and declaring it
        # would both satisfy this check and collapse 6 poses per position into
        # 1 pose plus 6 fixed offsets. See docs/HANDOFF.md.
        _run(["glomap", "mapper", "--database_path", str(db),
              "--image_path", str(chunk / "images"), "--output_path", str(raw)])
    elif mapper == "hierarchical":
        # COLMAP's own divide-and-conquer: it splits the scene into OVERLAPPING
        # sub-models by image connectivity, reconstructs each, and merges them.
        # Note what is different from what this pipeline used to do -- the
        # partition is over one shared database and the merge is part of the
        # solve, so the output is a SINGLE reconstruction with one datum. Our
        # old per-chunk solving partitioned the images and never merged, which
        # is why the chunks disagreed (docs/SCALING-JOURNAL.md, entry 3).
        #
        # This is the fallback for captures too large for a monolithic global
        # solve. COLMAP's own docs call it "usually less robust than the other
        # two pipelines", so it is a scale concession, not a default.
        hier = ["colmap", "hierarchical_mapper",
                "--database_path", str(db),
                "--image_path", str(chunk / "images"),
                "--output_path", str(raw),
                "--leaf_max_num_images", os.environ.get("HIER_LEAF", "500"),
                "--image_overlap", os.environ.get("HIER_OVERLAP", "50")]
        if (chunk / "cameras.json").exists():
            # same reasoning as the incremental branch: our pinholes are exact
            hier += ["--Mapper.ba_refine_focal_length", "0",
                     "--Mapper.ba_refine_principal_point", "0",
                     "--Mapper.ba_refine_extra_params", "0"]
        _run(hier)
    else:
        print("[poses] using COLMAP's incremental mapper: CPU-only and much "
              "slower on sequences this size (measured 4h53m on 3,870 images "
              "against 73 min for all the GPU stages combined)", flush=True)
        mapper = ["colmap", "mapper", "--database_path", str(db),
                  "--image_path", str(chunk / "images"), "--output_path", str(raw)]
        if (chunk / "cameras.json").exists():
            # Our views are synthesised pinholes: fx/fy/cx/cy are EXACT, and
            # letting BA "refine" them is how a long road bends. GLOMAP 1.0.0
            # cannot be told this (it drifted 1144.1 -> 1142..1155 on the
            # street) -- the incremental mapper can.
            mapper += ["--Mapper.ba_refine_focal_length", "0",
                       "--Mapper.ba_refine_principal_point", "0",
                       "--Mapper.ba_refine_extra_params", "0"]
        if _ba_gpu_available():
            mapper += ["--Mapper.ba_use_gpu", "1"]
            print("[poses] ceres has CUDA: bundle adjustment on GPU", flush=True)
        _run(mapper)
    model = _largest_model(raw)

    final = sparse / "0"
    enu_ref, gps_ref = chunk / "geo_enu.txt", chunk / "geo.txt"
    if align and (enu_ref.exists() or gps_ref.exists()):
        final.mkdir(exist_ok=True)
        if enu_ref.exists():
            # positions already in the project ENU frame -> every chunk lands in
            # the same world, which is what makes merging a concatenation
            ref_args = ["--ref_images_path", str(enu_ref), "--ref_is_gps", "0",
                        "--alignment_type", "custom"]
        else:
            ref_args = ["--ref_images_path", str(gps_ref), "--ref_is_gps", "1",
                        "--alignment_type", "enu"]
        _run(["colmap", "model_aligner",
              "--input_path", str(model), "--output_path", str(final),
              *ref_args, "--alignment_max_error", "3"])
    else:
        shutil.move(str(model), str(final))
    # The GPS corridor got us here; the solved cameras are the truth now. Not
    # fatal: a chunk without frame records just keeps the GPS corridor.
    try:
        from . import corridor as corridor_mod      # noqa: PLC0415
        corridor_mod.refresh_from_sparse(chunk)
    except Exception as exc:                        # noqa: BLE001
        print(f"[poses] {chunk.name}: corridor refresh skipped ({exc})", flush=True)
    print(f"[poses] {chunk.name}: done -> {final}", flush=True)


def solve_all(chunks_dir: Path, matcher: str = "spatial", align: bool = True,
              only: list[str] | None = None, spatial_radius: int = 4,
              refresh: bool = False, mapper: str = "auto",
              loop_closure: str = "none"):
    chunks = list_chunks(chunks_dir)
    if only:
        chunks = [c for c in chunks if any(o in c.name for o in only)]
        print(f"[poses] --only {only}: {len(chunks)} chunk(s)", flush=True)
    q = WorkQueue(chunks_dir, "poses")
    print(f"[poses] worker {q.worker}: {len(chunks)} chunk(s) in the pool", flush=True)
    done, failed = q.run(chunks, lambda c: solve_chunk(
        c, matcher=matcher, align=align, spatial_radius=spatial_radius,
        refresh=refresh, mapper=mapper, loop_closure=loop_closure))
    print(f"[poses] worker {q.worker}: solved {len(done)}, failed {len(failed)}", flush=True)
    if failed:
        raise SystemExit(f"[poses] failed chunks: {failed}")
