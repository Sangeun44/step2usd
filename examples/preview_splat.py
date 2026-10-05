"""Render a preview image of a Gaussian-splat stage with a small CPU splatting rasteriser.

    python examples/preview_splat.py out/bracket_splat.usdc docs/preview_splat.png

It reads the ParticleField prim back out of the USD file and draws it the way
3D Gaussian Splatting does: each Gaussian is projected to a 2D ellipse and the
ellipses are alpha-blended front to back. Only the view-independent (degree 0)
colour is used. Wrong quaternion order, scale units or opacity encoding in the
file all show up here as holes, fuzz or a washed-out image.
Needs Pillow, which is not a dependency of the converter itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdVol

from step2usd.splat import read_splat_usd


def _rotation_matrices(quats: np.ndarray) -> np.ndarray:
    w, x, y, z = quats.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def render(stage_path: Path, image_path: Path, size=(960, 720), azimuth=-58.0, elevation=28.0) -> Path:
    data = read_splat_usd(stage_path)
    stage = Usd.Stage.Open(str(stage_path))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdVol.ParticleField3DGaussianSplat))
    matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    to_world = matrix[:3, :3].T  # USD stores row vectors; this is the column-vector rotation
    positions = data.positions @ matrix[:3, :3] + matrix[3, :3]

    rotations = to_world @ _rotation_matrices(data.rotations.astype(float))
    scaled = rotations * data.scales[:, None, :]  # R S
    covariance = scaled @ scaled.transpose(0, 2, 1)  # R S S^T R^T

    # Orthographic camera at (azimuth, elevation) around the stage's up axis.
    up = np.array([0.0, 1.0, 0.0]) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else np.array([0.0, 0.0, 1.0])
    ground_x = np.array([1.0, 0.0, 0.0])
    ground_y = np.cross(up, ground_x)
    az, el = np.radians(azimuth), np.radians(elevation)
    to_camera = np.cos(el) * (np.cos(az) * ground_x + np.sin(az) * ground_y) + np.sin(el) * up
    right = np.cross(up, to_camera)
    right /= np.linalg.norm(right)
    cam_up = np.cross(to_camera, right)

    width, height = size
    x, y = positions @ right, positions @ cam_up
    pixels_per_unit = 0.86 * width / max(x.max() - x.min(), (y.max() - y.min()) * width / height)
    px = (x - (x.max() + x.min()) / 2) * pixels_per_unit + width / 2
    py = height / 2 - (y - (y.max() + y.min()) / 2) * pixels_per_unit

    project = pixels_per_unit * np.stack([right, -cam_up])  # world -> pixel offsets, 2x3
    cov2d = project @ covariance @ project.T + 0.3 * np.eye(2)  # small blur, as 3DGS does
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    conic_a, conic_b, conic_c = cov2d[:, 1, 1] / det, -cov2d[:, 0, 1] / det, cov2d[:, 0, 0] / det
    radius = np.ceil(3.0 * np.sqrt(np.maximum(cov2d[:, 0, 0], cov2d[:, 1, 1]))).astype(int)
    colors = data.colors()

    image = np.zeros((height, width, 3))
    transmittance = np.ones((height, width))
    for i in np.argsort(-(positions @ to_camera)):  # nearest first
        c0, c1 = max(int(px[i]) - radius[i], 0), min(int(px[i]) + radius[i] + 1, width)
        r0, r1 = max(int(py[i]) - radius[i], 0), min(int(py[i]) + radius[i] + 1, height)
        if c0 >= c1 or r0 >= r1:
            continue
        dx = np.arange(c0, c1) + 0.5 - px[i]
        dy = (np.arange(r0, r1) + 0.5 - py[i])[:, None]
        power = -0.5 * (conic_a[i] * dx * dx + 2 * conic_b[i] * dx * dy + conic_c[i] * dy * dy)
        alpha = np.minimum(0.99, data.opacities[i] * np.exp(power))
        remaining = transmittance[r0:r1, c0:c1]
        image[r0:r1, c0:c1] += (remaining * alpha)[..., None] * colors[i]
        remaining *= 1.0 - alpha

    image += transmittance[..., None]  # white background behind whatever is left
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(image_path)
    return image_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"wrote {render(Path(sys.argv[1]), Path(sys.argv[2]))}")
