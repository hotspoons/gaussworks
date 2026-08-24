# SPDX-License-Identifier: Apache-2.0
"""Stage 7: splat -> textured mesh, for engines that cannot render gaussians.

Assetto Corsa, BeamNG, RBR and friends all want triangles. The route here is
depth-render-and-fuse, which keeps the whole chain permissively licensed
(gsplat Apache-2.0, Open3D MIT) unlike the mesh-from-splat research codebases
that carry non-commercial terms:

    trained splat --rasterize depth at every capture pose--> RGBD frames
    RGBD + real photos --TSDF fusion--> watertight-ish surface
    surface --corridor crop + decimate--> game-budget mesh

Colour comes from the ORIGINAL frames rather than from splat renders: the
photos are sharper, and the geometry is what we needed the splat for. The
result is vertex-coloured; a UV atlas bake is the next step, and is what a sim
actually wants for texture streaming.

Voxel size is the quality dial. 5 cm resolves road texture and kerbs; 10 cm is
plenty for scenery and a quarter of the memory.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_parser(chunk: Path, factor: int = 1):
    """Reuse gsplat's own COLMAP parser so poses match training exactly."""
    examples = Path(os.environ.get("GSPLAT_EXAMPLES", "/opt/gsplat/examples"))
    if not (examples / "datasets" / "colmap.py").exists():
        examples = Path("/workspace/opt/gsplat/examples")
    sys.path.insert(0, str(examples))
    from datasets.colmap import Parser           # noqa: PLC0415
    return Parser(data_dir=str(chunk), factor=factor, normalize=False)


def _load_splats(ckpt: Path, device):
    import torch                                 # noqa: PLC0415
    state = torch.load(ckpt, map_location=device, weights_only=False)
    splats = state["splats"] if "splats" in state else state
    # the trainer keeps scales in log space and opacities as logits
    return {
        "means": splats["means"].to(device),
        "quats": splats["quats"].to(device),
        "scales": torch.exp(splats["scales"].to(device)),
        "opacities": torch.sigmoid(splats["opacities"].to(device)).squeeze(),
        "sh": torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1),
    }


def build(chunk: Path, ckpt: Path | None = None, out: Path | None = None,
          voxel_m: float = 0.05, trunc_m: float | None = None,
          depth_max_m: float = 30.0, max_tris: int = 0,
          corridor_pad_m: float = 5.0, min_alpha: float = 0.6) -> Path:
    import open3d as o3d                          # noqa: PLC0415
    import torch                                  # noqa: PLC0415
    from gsplat import rasterization              # noqa: PLC0415
    from PIL import Image                         # noqa: PLC0415

    chunk = Path(chunk)
    out = Path(out or chunk / "mesh")
    out.mkdir(parents=True, exist_ok=True)
    if ckpt is None:
        ckpts = sorted((chunk / "splat" / "ckpts").glob("ckpt_*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint under {chunk}/splat/ckpts")
        ckpt = ckpts[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats = _load_splats(Path(ckpt), device)
    sh_degree = int(round(splats["sh"].shape[1] ** 0.5)) - 1
    parser = _load_parser(chunk)
    n_cams = len(parser.camtoworlds)
    print(f"[mesh] {n_cams} cameras, {len(splats['means'])} gaussians, "
          f"sh_degree={sh_degree}, voxel={voxel_m} m")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m,
        sdf_trunc=trunc_m if trunc_m is not None else voxel_m * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(n_cams):
        c2w = torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device=device)
        K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]],
                         dtype=torch.float32, device=device)
        photo = np.asarray(Image.open(parser.image_paths[i]).convert("RGB"))
        h, w = photo.shape[:2]

        with torch.no_grad():
            render, alphas, _ = rasterization(
                splats["means"], splats["quats"], splats["scales"],
                splats["opacities"], splats["sh"],
                torch.linalg.inv(c2w)[None], K[None], w, h,
                sh_degree=sh_degree, render_mode="RGB+ED",
                near_plane=0.1, far_plane=depth_max_m)
        # Only integrate pixels the splat actually covers. gsplat's expected
        # depth is a weighted mean, so in sky or empty space it returns a
        # plausible-looking number with no surface behind it -- integrating
        # those inflates the TSDF across the whole far field (and OOMs).
        alpha = alphas[0, ..., 0]
        depth = render[0, ..., 3]
        depth = torch.where((alpha >= min_alpha) & (depth <= depth_max_m),
                            depth, torch.zeros_like(depth))
        depth = depth.clamp(0, depth_max_m).cpu().numpy().astype(np.float32)
        covered = float((depth > 0).mean())

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(photo)),
            o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=depth_max_m,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        volume.integrate(rgbd, intr, np.linalg.inv(parser.camtoworlds[i]))
        if (i + 1) % 50 == 0:
            print(f"[mesh]   fused {i + 1}/{n_cams} (last frame {covered:.0%} covered)")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"[mesh] fused surface: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris")

    corridor_path = chunk / "corridor.json"
    if corridor_path.exists():
        from .merge import _corridor_mask         # noqa: PLC0415
        corridor = json.loads(corridor_path.read_text())
        corridor = {**corridor, "radius_m": corridor.get("radius_m", 25.0)
                    + corridor_pad_m}
        keep = _corridor_mask(np.asarray(mesh.vertices), corridor)
        mesh.remove_vertices_by_mask(~keep)
        print(f"[mesh] corridor crop -> {len(mesh.triangles)} tris")

    if max_tris and len(mesh.triangles) > max_tris:
        mesh = mesh.simplify_quadric_decimation(int(max_tris))
        mesh.compute_vertex_normals()
        print(f"[mesh] decimated -> {len(mesh.triangles)} tris")

    ply_path = out / "mesh.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)          # keeps vertex colours
    o3d.io.write_triangle_mesh(str(out / "mesh.obj"), mesh)  # geometry for DCC/engine
    (out / "mesh.json").write_text(json.dumps(
        {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
         "voxel_m": voxel_m, "source_ckpt": str(ckpt),
         "cameras": n_cams, "frame": "enu"}, indent=1))
    print(f"[mesh] wrote {ply_path} (+ .obj, .json)")
    return out
