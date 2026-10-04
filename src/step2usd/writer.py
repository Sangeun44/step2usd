"""Write the neutral scene model as a USD stage.

Layout of the result:

    /<Root>                 Xform, defaultPrim, the top assembly
      /<Assembly>           Xform per sub-assembly, with its placement
        /<Part>             Xform per placed part
          /Mesh             geometry, when the part is used once
    /_Prototypes            class prim, skipped by renderers and traversals
      /<Part>/Mesh          geometry of each part that is placed more than once

A part placed more than once is written a single time under /_Prototypes and
each placement is an instanceable reference to it, so forty bolts cost one mesh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from . import __version__
from .model import MeshData, Node, Part, Scene

MM_TO_M = 0.001


@dataclass
class WriteOptions:
    units: str = "m"  # "m" rescales the geometry to metres; "mm" keeps CAD units
    up_axis: str = "Z"  # "Z" matches CAD and robotics; "Y" turns the model for DCC tools
    instancing: bool = True
    density: float | None = None  # kg/m^3; when set, colliders and exact masses are authored
    face_ids: bool = False  # keep the B-rep face index of every triangle as a primvar
    metadata: dict = field(default_factory=dict)  # extra provenance for the layer

    def __post_init__(self) -> None:
        if self.units not in ("m", "mm"):
            raise ValueError(f"units must be 'm' or 'mm', got {self.units!r}")
        if self.up_axis not in ("Z", "Y"):
            raise ValueError(f"up_axis must be 'Z' or 'Y', got {self.up_axis!r}")
        if self.density is not None and self.density <= 0:
            raise ValueError("density must be positive")

    @property
    def scale(self) -> float:
        """Multiplier from the model's millimetres to stage units."""
        return MM_TO_M if self.units == "m" else 1.0

    @property
    def meters_per_unit(self) -> float:
        return 1.0 if self.units == "m" else MM_TO_M


@dataclass
class WriteResult:
    path: Path
    root_path: str
    prototypes: int = 0  # parts written once and instanced
    instances: int = 0  # placements that reference a prototype
    renamed: list[tuple[str, str]] = field(default_factory=list)  # (CAD name, prim name)
    mirrored: list[str] = field(default_factory=list)  # prim paths with a mirroring transform


def make_identifier(name: str, fallback: str = "Part") -> str:
    """Turn a CAD name into a valid prim name: 'M8 Bolt (rev B)' -> 'M8_Bolt_rev_B'."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not cleaned:
        return fallback
    return f"_{cleaned}" if cleaned[0].isdigit() else cleaned


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 1
    while candidate in taken:
        candidate = f"{name}_{n}"
        n += 1
    taken.add(candidate)
    return candidate


class _Writer:
    def __init__(self, scene: Scene, path: Path, options: WriteOptions):
        self.scene = scene
        self.options = options
        self.scale = options.scale
        self.result = WriteResult(path=path, root_path="")
        # Converting to the same path twice in one process must not trip over the cached layer.
        cached = Sdf.Layer.Find(str(path))
        if cached:
            cached.Clear()
            self.stage = Usd.Stage.Open(cached)
        else:
            self.stage = Usd.Stage.CreateNew(str(path))
        self.prototype_paths: dict[str, Sdf.Path] = {}
        self.prototype_names: set[str] = set()
        counts = scene.occurrences()
        self.shared = {key for key, n in counts.items() if options.instancing and n > 1}

    # -- geometry ------------------------------------------------------------

    def _write_mesh(self, parent: Sdf.Path, part: Part) -> None:
        data: MeshData = part.mesh
        if data.is_empty:
            return
        mesh = UsdGeom.Mesh.Define(self.stage, parent.AppendChild("Mesh"))
        points = (data.points * self.scale).astype(np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(data.triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(data.triangles.ravel()))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(data.normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(
            Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)]))
        )
        # An open shell has no inside, so show both sides of it.
        mesh.CreateDoubleSidedAttr(not part.is_solid)

        base = part.color or (0.7, 0.7, 0.7)
        if part.face_colors:
            per_triangle = np.array(
                [part.face_colors.get(int(f), base) for f in data.face_ids], dtype=np.float32
            )
            color = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
            color.Set(Vt.Vec3fArray.FromNumpy(per_triangle))
        else:
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*base)])

        if self.options.face_ids:
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "cadFaceId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform
            )
            primvar.Set(Vt.IntArray.FromNumpy(data.face_ids))

        if self.options.density is not None:
            self._write_physics(mesh.GetPrim(), part)

    def _write_physics(self, prim: Usd.Prim, part: Part) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
        if part.volume_mm3 is None:
            return  # not a solid: collider only, no mass to compute
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(self.options.density * part.volume_mm3 * MM_TO_M**3)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*(c * self.scale for c in part.center_of_mass_mm)))

    def _prototype(self, part: Part) -> Sdf.Path:
        if part.key not in self.prototype_paths:
            name = _unique(make_identifier(part.name), self.prototype_names)
            path = Sdf.Path("/_Prototypes").AppendChild(name)
            UsdGeom.Xform.Define(self.stage, path)
            self._write_mesh(path, part)
            self.prototype_paths[part.key] = path
            self.result.prototypes += 1
        return self.prototype_paths[part.key]

    # -- hierarchy -----------------------------------------------------------

    def _write_node(self, node: Node, parent: Sdf.Path, siblings: set[str], is_root: bool) -> Sdf.Path:
        wanted = make_identifier(node.name, "Assembly" if node.is_assembly else "Part")
        name = _unique(wanted, siblings)
        path = parent.AppendChild(name)
        xform = UsdGeom.Xform.Define(self.stage, path)
        prim = xform.GetPrim()
        if name != node.name:
            prim.SetDisplayName(node.name)
            self.result.renamed.append((node.name, name))

        if is_root and self.options.up_axis == "Y":
            xform.AddRotateXOp().Set(-90.0)  # CAD +Z becomes stage +Y
        if not np.allclose(node.transform, np.eye(4)):
            local = node.transform.copy()
            local[:3, 3] *= self.scale
            xform.AddTransformOp().Set(Gf.Matrix4d(*local.T.ravel()))  # USD stores row vectors
            if np.linalg.det(local[:3, :3]) < 0:
                self.result.mirrored.append(str(path))

        if node.is_assembly:
            kind = Kind.Tokens.assembly if is_root else Kind.Tokens.group
            Usd.ModelAPI(prim).SetKind(kind)
            taken: set[str] = set()
            for child in node.children:
                self._write_node(child, path, taken, is_root=False)
            return path

        Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)
        if node.part.key in self.shared:
            prim.GetReferences().AddInternalReference(self._prototype(node.part))
            prim.SetInstanceable(True)
            self.result.instances += 1
        else:
            self._write_mesh(path, node.part)
        return path

    def write(self) -> WriteResult:
        stage, options = self.stage, self.options
        UsdGeom.SetStageMetersPerUnit(stage, options.meters_per_unit)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if options.up_axis == "Z" else UsdGeom.Tokens.y)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        stage.GetRootLayer().customLayerData = {
            "step2usd": {"version": __version__, "source": self.scene.source, **options.metadata}
        }

        if self.shared:
            stage.CreateClassPrim("/_Prototypes")
        root_path = self._write_node(self.scene.root, Sdf.Path.absoluteRootPath, {"_Prototypes"}, True)
        root = stage.GetPrimAtPath(root_path)
        stage.SetDefaultPrim(root)
        if options.density is not None:
            UsdPhysics.RigidBodyAPI.Apply(root)

        stage.GetRootLayer().Save()
        self.result.root_path = str(root_path)
        return self.result


def write_usd(scene: Scene, path: Path, options: WriteOptions | None = None) -> WriteResult:
    """Write scene to path (.usda, .usdc or .usd) and return what was authored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _Writer(scene, path, options or WriteOptions()).write()
