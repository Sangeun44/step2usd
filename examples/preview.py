"""Render a preview image of a converted stage with a small software rasteriser.

    python examples/preview.py out/bracket.usda docs/preview.png

It reads the geometry back out of the USD file (through instances, transforms,
normals and colours), so the picture shows what was written, not the CAD input.
Shading uses the authored normals, so flipped normals show up as dark faces.
Needs Pillow, which is not a dependency of the converter itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

SUPERSAMPLE = 3


def stage_triangles(stage: Usd.Stage):
    """Yield world-space (positions (M,3,3), normals (M,3,3), colours (M,3)) per mesh."""
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        matrix = np.array(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        world = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]  # USD uses row vectors
        normals = np.array(mesh.GetNormalsAttr().Get(), dtype=float) @ np.linalg.inv(matrix[:3, :3]).T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        primvar = mesh.GetDisplayColorPrimvar()
        colors = np.array(primvar.Get(), dtype=float)
        if primvar.GetInterpolation() != UsdGeom.Tokens.uniform:
            colors = np.repeat(colors[:1], len(indices), axis=0)
        yield world[indices], normals[indices], colors


def render(stage_path: Path, image_path: Path, size=(960, 720), azimuth=-58.0, elevation=28.0) -> Path:
    stage = Usd.Stage.Open(str(stage_path))
    tris, normals, colors = map(np.concatenate, zip(*stage_triangles(stage)))

    # Orthographic camera looking at the model from (azimuth, elevation) around the up axis.
    up = np.array([0.0, 1.0, 0.0]) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else np.array([0.0, 0.0, 1.0])
    ground_x = np.array([1.0, 0.0, 0.0])
    ground_y = np.cross(up, ground_x)
    az, el = np.radians(azimuth), np.radians(elevation)
    to_camera = np.cos(el) * (np.cos(az) * ground_x + np.sin(az) * ground_y) + np.sin(el) * up
    right = np.cross(up, to_camera)
    right /= np.linalg.norm(right)
    cam_up = np.cross(to_camera, right)
    light = to_camera + 0.6 * cam_up - 0.5 * right
    light /= np.linalg.norm(light)

    width, height = size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE
    x, y, depth = tris @ right, tris @ cam_up, tris @ to_camera
    span = max(x.max() - x.min(), (y.max() - y.min()) * width / height)
    scale = 0.86 * width / span
    px = (x - (x.max() + x.min()) / 2) * scale + width / 2
    py = height / 2 - (y - (y.max() + y.min()) / 2) * scale

    image = np.ones((height, width, 3))
    zbuffer = np.full((height, width), -np.inf)
    for i in range(len(tris)):
        (x0, x1, x2), (y0, y1, y2) = px[i], py[i]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        c0, c1 = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))) + 1, width)
        r0, r1 = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))) + 1, height)
        if c0 >= c1 or r0 >= r1:
            continue
        gx, gy = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
        w1 = ((gx - x0) * (y2 - y0) - (x2 - x0) * (gy - y0)) / area
        w2 = ((x1 - x0) * (gy - y0) - (gx - x0) * (y1 - y0)) / area
        w0 = 1.0 - w1 - w2
        z = w0 * depth[i, 0] + w1 * depth[i, 1] + w2 * depth[i, 2]
        region = zbuffer[r0:r1, c0:c1]
        visible = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z > region)
        if not visible.any():
            continue
        n = w0[..., None] * normals[i, 0] + w1[..., None] * normals[i, 1] + w2[..., None] * normals[i, 2]
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30
        shade = 0.3 + 0.7 * np.clip(n @ light, 0.0, 1.0)
        region[visible] = z[visible]
        image[r0:r1, c0:c1][visible] = colors[i] * shade[visible][:, None]

    # displayColor is linear: average the supersamples first, then convert to sRGB.
    image = image.reshape(size[1], SUPERSAMPLE, size[0], SUPERSAMPLE, 3).mean(axis=(1, 3))
    pixels = (np.clip(image, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(image_path)
    return image_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"wrote {render(Path(sys.argv[1]), Path(sys.argv[2]))}")
