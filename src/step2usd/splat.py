"""Gaussian splats: standard 3DGS .ply files in, OpenUSD ParticleField prims out.

A trained 3D Gaussian Splatting capture is usually saved as a .ply file in the
layout of the original implementation (Kerbl et al. 2023). OpenUSD 26.03 added
a native prim for the same data, UsdVol.ParticleField3DGaussianSplat. The two
store the same Gaussians with different conventions:

    PLY (training output)                  USD ParticleField
    ------------------------------------   -----------------------------------
    scale_*      log of the std deviation  scales       linear
    opacity      logit (pre-sigmoid)       opacities    linear, 0..1
    rot_0..3     w, x, y, z, unnormalised  orientations unit quaternions
    f_dc, f_rest colour-channel-major      SH coefficients grouped per particle

This module converts between them, and can also sample a CAD model's surfaces
into Gaussians so a part can sit in a splat scene or serve as test data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdVol, Vt

from . import __version__
from .model import Scene

SH_C0 = 0.28209479177387814  # the constant (degree 0) spherical-harmonics basis function
_REQUIRED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


class SplatError(ValueError):
    pass


@dataclass
class SplatData:
    """Gaussians in the activated, renderer-facing form that USD stores."""

    positions: np.ndarray  # (N, 3) float32
    rotations: np.ndarray  # (N, 4) float32, unit quaternions as (w, x, y, z)
    scales: np.ndarray  # (N, 3) float32, linear standard deviations
    opacities: np.ndarray  # (N,) float32, 0..1
    sh: np.ndarray  # (N, K, 3) float32, K = (degree + 1)^2 coefficients per Gaussian

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def degree(self) -> int:
        return math.isqrt(self.sh.shape[1]) - 1

    def colors(self) -> np.ndarray:
        """View-independent base colour of each Gaussian, from the degree 0 term."""
        return np.clip(0.5 + SH_C0 * self.sh[:, 0, :], 0.0, 1.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box around every Gaussian out to three standard deviations."""
        reach = 3.0 * self.scales.max(axis=1, keepdims=True)
        return (self.positions - reach).min(axis=0), (self.positions + reach).max(axis=0)

    def select(self, keep: np.ndarray) -> "SplatData":
        return SplatData(self.positions[keep], self.rotations[keep], self.scales[keep],
                         self.opacities[keep], self.sh[keep])


# --- PLY ----------------------------------------------------------------------


def _read_header(handle) -> tuple[int, list[tuple[str, str]]]:
    if handle.readline().strip() != b"ply":
        raise SplatError("not a PLY file")
    count, properties, in_vertex, fmt = None, [], False, None
    for _ in range(4096):
        line = handle.readline()
        if not line:
            raise SplatError("PLY header has no end_header")
        words = line.decode("ascii", "replace").split()
        if not words or words[0] == "comment":
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            in_vertex = words[1] == "vertex"
            if in_vertex:
                count = int(words[2])
            elif count is None:
                raise SplatError("PLY elements before 'vertex' are not supported")
        elif words[0] == "property" and in_vertex:
            if words[1] == "list" or words[1] not in _PLY_TYPES:
                raise SplatError(f"unsupported PLY property type: {' '.join(words[1:])}")
            properties.append((words[2], "<" + _PLY_TYPES[words[1]]))
        elif words[0] == "end_header":
            break
    else:
        raise SplatError("PLY header is too long")
    if fmt != "binary_little_endian":
        raise SplatError(f"only binary_little_endian PLY is supported, this file is {fmt}")
    if count is None:
        raise SplatError("PLY file has no vertex element")
    return count, properties


def read_ply(path: Path) -> SplatData:
    """Read a 3DGS .ply and apply the activations, giving renderer-facing values."""
    path = Path(path)
    if not path.is_file():
        raise SplatError(f"no such file: {path}")
    with path.open("rb") as handle:
        count, properties = _read_header(handle)
        names = [name for name, _ in properties]
        missing = [name for name in _REQUIRED if name not in names]
        if missing:
            raise SplatError(
                f"{path.name} is a PLY file but not a Gaussian splat: missing {', '.join(missing)}"
            )
        raw = np.fromfile(handle, dtype=np.dtype(properties), count=count)
    if len(raw) != count:
        raise SplatError(f"{path.name} is truncated: header says {count} Gaussians, found {len(raw)}")

    def columns(*wanted: str) -> np.ndarray:
        return np.stack([raw[name] for name in wanted], axis=1).astype(np.float32)

    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    per_channel, remainder = divmod(len(rest_names), 3)
    coefficients = per_channel + 1
    degree = math.isqrt(coefficients) - 1
    if remainder or (degree + 1) ** 2 != coefficients:
        raise SplatError(f"{len(rest_names)} f_rest properties do not form a whole SH degree")

    sh = np.empty((count, coefficients, 3), dtype=np.float32)
    sh[:, 0, :] = columns("f_dc_0", "f_dc_1", "f_dc_2")
    if per_channel:
        # f_rest holds all red coefficients, then all green, then all blue.
        rest = columns(*rest_names).reshape(count, 3, per_channel)
        sh[:, 1:, :] = rest.transpose(0, 2, 1)

    rotations = columns("rot_0", "rot_1", "rot_2", "rot_3")
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = np.where(norms > 0, rotations / np.where(norms > 0, norms, 1.0), [1.0, 0.0, 0.0, 0.0])

    with np.errstate(over="ignore"):
        scales = np.exp(columns("scale_0", "scale_1", "scale_2"))
        opacities = 1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float32)))
    return SplatData(columns("x", "y", "z"), rotations.astype(np.float32), scales, opacities, sh)


def write_ply(data: SplatData, path: Path) -> Path:
    """Write Gaussians as a 3DGS .ply, undoing the activations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, coefficients = len(data), data.sh.shape[1]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * (coefficients - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    opacity = np.clip(data.opacities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    table = np.concatenate(
        [
            data.positions,
            np.zeros((count, 3)),  # normals: present in the format, unused by renderers
            data.sh[:, 0, :],
            data.sh[:, 1:, :].transpose(0, 2, 1).reshape(count, -1),
            np.log(opacity / (1.0 - opacity))[:, None],
            np.log(np.maximum(data.scales, 1e-12)),
            data.rotations,
        ],
        axis=1,
    ).astype("<f4")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in names] + ["end_header", ""]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(table.tobytes())
    return path


# --- cleaning -----------------------------------------------------------------


def clean(data: SplatData, min_opacity: float = 0.0) -> tuple[SplatData, dict]:
    """Drop Gaussians a renderer cannot use, and report how many of each kind went."""
    finite = (
        np.isfinite(data.positions).all(axis=1)
        & np.isfinite(data.scales).all(axis=1)
        & np.isfinite(data.rotations).all(axis=1)
        & np.isfinite(data.opacities)
        & np.isfinite(data.sh).all(axis=(1, 2))
    )
    faint = finite & (data.opacities < min_opacity)
    dropped = {"non_finite": int((~finite).sum()), "below_min_opacity": int(faint.sum())}
    return data.select(finite & ~faint), dropped


# --- USD ----------------------------------------------------------------------


@dataclass
class SplatWriteOptions:
    up_axis: str = "Z"
    meters_per_unit: float = 1.0
    rotate_x: float = 0.0  # degrees, applied on the root prim; the Gaussians are left untouched
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.up_axis not in ("Z", "Y"):
            raise ValueError(f"up_axis must be 'Z' or 'Y', got {self.up_axis!r}")
        if self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be positive")


def write_splat_usd(data: SplatData, path: Path, options: SplatWriteOptions | None = None) -> Path:
    """Write the Gaussians as a ParticleField3DGaussianSplat prim under an Xform root."""
    options = options or SplatWriteOptions()
    if len(data) == 0:
        raise SplatError("there are no Gaussians to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = Sdf.Layer.Find(str(path))
    if cached:
        cached.Clear()
        stage = Usd.Stage.Open(cached)
    else:
        stage = Usd.Stage.CreateNew(str(path))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if options.up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, options.meters_per_unit)
    stage.GetRootLayer().customLayerData = {"step2usd": {"version": __version__, **options.metadata}}

    root = UsdGeom.Xform.Define(stage, "/Capture")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
    if options.rotate_x:
        root.AddRotateXOp().Set(float(options.rotate_x))

    splat = UsdVol.ParticleField3DGaussianSplat.Define(stage, "/Capture/Splat")
    splat.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data.positions, np.float32)))
    # Vt quaternions are laid out (x, y, z, w); ours are (w, x, y, z).
    xyzw = np.ascontiguousarray(data.rotations[:, [1, 2, 3, 0]], np.float32)
    splat.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(xyzw))
    splat.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data.scales, np.float32)))
    splat.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(np.ascontiguousarray(data.opacities, np.float32)))
    splat.CreateRadianceSphericalHarmonicsDegreeAttr(data.degree)
    splat.CreateRadianceSphericalHarmonicsCoefficientsAttr(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data.sh.reshape(-1, 3), np.float32))
    )
    lo, hi = data.bounds()
    splat.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi]).astype(np.float32)))

    stage.GetRootLayer().Save()
    return path


def read_splat_usd(path: Path) -> SplatData:
    """Read the first ParticleField3DGaussianSplat prim in a stage back into SplatData."""
    stage = Usd.Stage.Open(str(path))
    prim = next((p for p in stage.Traverse() if p.IsA(UsdVol.ParticleField3DGaussianSplat)), None)
    if prim is None:
        raise SplatError(f"{path} contains no Gaussian splat prim")
    splat = UsdVol.ParticleField3DGaussianSplat(prim)
    positions = np.array(splat.GetPositionsAttr().Get(), dtype=np.float32).reshape(-1, 3)
    xyzw = np.array(splat.GetOrientationsAttr().Get(), dtype=np.float32).reshape(-1, 4)
    degree = splat.GetRadianceSphericalHarmonicsDegreeAttr().Get()
    sh = np.array(splat.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(), dtype=np.float32)
    return SplatData(
        positions,
        xyzw[:, [3, 0, 1, 2]],
        np.array(splat.GetScalesAttr().Get(), dtype=np.float32).reshape(-1, 3),
        np.array(splat.GetOpacitiesAttr().Get(), dtype=np.float32),
        sh.reshape(len(positions), (degree + 1) ** 2, 3),
    )


def convert_splat(
    source: Path, output: Path, options: SplatWriteOptions | None = None, min_opacity: float = 0.0
) -> dict:
    """Convert a 3DGS .ply to USD, read the result back, and return a report."""
    options = options or SplatWriteOptions()
    options.metadata = {"source": Path(source).name, **options.metadata}
    loaded = read_ply(source)
    data, dropped = clean(loaded, min_opacity)
    write_splat_usd(data, output, options)

    back = read_splat_usd(output)
    round_trip = {
        "count_matches": len(back) == len(data),
        "max_position_error": float(np.abs(back.positions - data.positions).max()),
        "max_scale_error": float(np.abs(back.scales - data.scales).max()),
        "max_opacity_error": float(np.abs(back.opacities - data.opacities).max()),
        "max_rotation_error": float(np.abs(back.rotations - data.rotations).max()),
        "max_sh_error": float(np.abs(back.sh - data.sh).max()),
    }
    round_trip["ok"] = round_trip["count_matches"] and all(
        value < 1e-6 for key, value in round_trip.items() if key.startswith("max_")
    )
    lo, hi = data.bounds()
    return {
        "source": Path(source).name,
        "output": Path(output).name,
        "gaussians_read": len(loaded),
        "gaussians_written": len(data),
        "dropped": dropped,
        "sh_degree": data.degree,
        "up_axis": options.up_axis,
        "meters_per_unit": options.meters_per_unit,
        "bounds": [lo.tolist(), hi.tolist()],
        "opacity": {
            "mean": float(data.opacities.mean()),
            "nearly_transparent": int((data.opacities < 0.05).sum()),
        },
        "scale": {"median": float(np.median(data.scales)), "max": float(data.scales.max())},
        "round_trip": round_trip,
    }


# --- CAD surfaces to Gaussians ------------------------------------------------


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _quaternions_from_z(normals: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) that rotate the +Z axis onto each normal."""
    z = normals[:, 2]
    quats = np.stack([1.0 + z, -normals[:, 1], normals[:, 0], np.zeros_like(z)], axis=1)
    flipped = z < -1.0 + 1e-8  # +Z onto -Z: turn half a revolution about X
    quats[flipped] = [0.0, 1.0, 0.0, 0.0]
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def _spread_over_surface(
    tris: np.ndarray, normals: np.ndarray, count: int, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pick count well-spread points on the triangles: (triangle index, position) per point.

    Independent random points clump and leave holes. So draw many more candidates
    than needed and keep one per grid cell of roughly the target spacing, which
    bounds how dense any spot can get. Cells are keyed by facing direction too,
    so the two sides of a thin wall do not compete for the same cell.
    """
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    candidates = 8 * count
    chosen = rng.choice(len(tris), size=candidates, p=areas / areas.sum())
    u, v = rng.random(candidates), rng.random(candidates)
    over = u + v > 1.0  # fold the unit square onto the triangle for uniform coverage
    u, v = np.where(over, 1.0 - u, u), np.where(over, 1.0 - v, v)
    t = tris[chosen]
    positions = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])

    n = normals[chosen]
    axis = np.abs(n).argmax(axis=1)
    facing = 2 * axis + (n[np.arange(candidates), axis] > 0)
    cells = np.column_stack([np.floor(positions / spacing).astype(np.int64), facing])
    _, first = np.unique(cells, axis=0, return_index=True)

    if len(first) >= count:
        keep = rng.choice(first, size=count, replace=False)
    else:  # top up from the unused candidates
        unused = np.setdiff1d(np.arange(candidates), first, assume_unique=True)
        keep = np.concatenate([first, rng.choice(unused, size=count - len(first), replace=False)])
    keep.sort()
    return chosen[keep], positions[keep]


def splats_from_scene(scene: Scene, count: int = 12000, seed: int = 0) -> SplatData:
    """Sample a CAD scene's surfaces into flat, surface-aligned Gaussians, in metres.

    This is a synthetic splat, not a trained capture: colours are the CAD
    colours with a fixed key light baked in so the shape reads, and there is no
    view-dependent appearance (SH degree 0).
    """
    if count <= 0:
        raise SplatError("count must be positive")
    triangles, normals, colors = [], [], []

    def visit(node, parent: np.ndarray) -> None:
        world = parent @ node.transform
        if node.part is not None and not node.part.mesh.is_empty:
            mesh, part = node.part.mesh, node.part
            points = mesh.points @ world[:3, :3].T + world[:3, 3]
            triangles.append(points[mesh.triangles])
            base = part.color or (0.7, 0.7, 0.7)
            colors.append(np.array([part.face_colors.get(int(f), base) for f in mesh.face_ids]))
            vertex_normals = mesh.normals @ world[:3, :3].T
            normals.append(vertex_normals[mesh.triangles].mean(axis=1))
        for child in node.children:
            visit(child, world)

    visit(scene.root, np.eye(4))
    if not triangles:
        raise SplatError("the scene has no geometry to sample")
    tris = np.concatenate(triangles) * 0.001  # mm -> m
    face_normals = np.concatenate(normals)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_colors = np.concatenate(colors)

    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    spacing = math.sqrt(areas.sum() / count)  # typical distance between neighbouring samples
    chosen, positions = _spread_over_surface(tris, face_normals, count, spacing, np.random.default_rng(seed))
    sample_normals = face_normals[chosen]
    scales = np.tile([0.65 * spacing, 0.65 * spacing, 0.06 * spacing], (count, 1))

    light = np.array([0.35, -0.5, 0.8]) / np.linalg.norm([0.35, -0.5, 0.8])
    shade = 0.35 + 0.65 * np.clip(sample_normals @ light, 0.0, 1.0)
    srgb = _linear_to_srgb(face_colors[chosen] * shade[:, None])
    sh = ((srgb - 0.5) / SH_C0)[:, None, :]

    return SplatData(
        positions.astype(np.float32),
        _quaternions_from_z(sample_normals).astype(np.float32),
        scales.astype(np.float32),
        np.full(count, 0.95, dtype=np.float32),
        sh.astype(np.float32),
    )
