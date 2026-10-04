"""Read a STEP file into the neutral scene model using OpenCascade (via the OCP bindings).

The reader keeps what a plain mesh export throws away: the assembly tree, part
names, which parts are the same part placed many times, colours, and the exact
volume and centre of mass of each solid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib_ToolTriangulatedShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import Sequence_TDF_Label
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import (
    XCAFDoc_ColorCurv,
    XCAFDoc_ColorGen,
    XCAFDoc_ColorSurf,
    XCAFDoc_DocumentTool,
)

from .model import Color, MeshData, Node, Part, Scene


class StepReadError(RuntimeError):
    pass


def _label_name(label: TDF_Label) -> str | None:
    attr = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return None
    name = attr.Get().ToExtString().strip()
    # Occurrences without a name of their own come back as "=>[0:1:1:2]".
    return None if not name or name.startswith("=>[") else name


def _label_key(label: TDF_Label) -> str:
    entry = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, entry)
    return entry.ToCString()


def _rgb(color: Quantity_Color) -> Color:
    return (color.Red(), color.Green(), color.Blue())


def _matrix(location: TopLoc_Location) -> np.ndarray:
    trsf = location.Transformation()
    m = np.eye(4)
    for row in range(3):
        for col in range(4):
            m[row, col] = trsf.Value(row + 1, col + 1)
    return m


def tessellate(shape, linear_deflection: float, angular_deflection: float) -> tuple[MeshData, int, int]:
    """Triangulate a B-rep shape. Returns (mesh, face count, faces that failed to mesh).

    Vertices are not shared between B-rep faces, so the per-vertex normals taken
    from the underlying surfaces stay smooth inside a face and crisp across edges.
    """
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    points, normals, triangles, face_ids = [], [], [], []
    offset = face_count = failed = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        face_index = face_count
        face_count += 1
        explorer.Next()

        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            failed += 1
            continue

        BRepLib_ToolTriangulatedShape.ComputeNormals_s(face, triangulation)
        trsf = location.Transformation()
        reversed_face = face.Orientation() == TopAbs_REVERSED
        sign = -1.0 if reversed_face else 1.0

        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(trsf)
            n = triangulation.Normal(i).Transformed(trsf)
            points.append((p.X(), p.Y(), p.Z()))
            normals.append((sign * n.X(), sign * n.Y(), sign * n.Z()))
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            if reversed_face:
                b, c = c, b
            triangles.append((a - 1 + offset, b - 1 + offset, c - 1 + offset))
            face_ids.append(face_index)
        offset += triangulation.NbNodes()

    mesh = MeshData(
        points=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        face_ids=np.asarray(face_ids, dtype=np.int32),
    )
    return mesh, face_count, failed


class _Reader:
    def __init__(self, path: Path, linear_deflection: float, angular_deflection: float):
        self.path = path
        self.linear_deflection = linear_deflection
        self.angular_deflection = angular_deflection
        self.parts: dict[str, Part] = {}

        self.doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise StepReadError(f"{path} is not a readable STEP file")
        if not reader.Transfer(self.doc):
            raise StepReadError(f"could not transfer {path} into a CAD document")
        self.shapes = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self.colors = XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main())

    # -- colours -------------------------------------------------------------

    def _label_color(self, label: TDF_Label) -> Color | None:
        color = Quantity_Color()
        for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv):
            if self.colors.GetColor_s(label, kind, color):
                return _rgb(color)
        return None

    def _face_colors(self, shape) -> dict[int, Color]:
        found: dict[int, Color] = {}
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        index = 0
        while explorer.More():
            color = Quantity_Color()
            if self.colors.GetColor(explorer.Current(), XCAFDoc_ColorSurf, color):
                found[index] = _rgb(color)
            index += 1
            explorer.Next()
        return found

    # -- parts ---------------------------------------------------------------

    def _part(self, label: TDF_Label) -> Part:
        key = _label_key(label)
        if key in self.parts:
            return self.parts[key]

        shape = self.shapes.GetShape_s(label)
        mesh, face_count, failed = tessellate(shape, self.linear_deflection, self.angular_deflection)
        part = Part(
            key=key,
            name=_label_name(label) or f"Part_{len(self.parts) + 1}",
            mesh=mesh,
            color=self._label_color(label),
            face_colors=self._face_colors(shape),
            face_count=face_count,
            failed_faces=failed,
            is_solid=TopExp_Explorer(shape, TopAbs_SOLID).More(),
            is_valid=BRepCheck_Analyzer(shape).IsValid(),
        )
        if part.is_solid:
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, props)
            center = props.CentreOfMass()
            part.volume_mm3 = props.Mass()
            part.center_of_mass_mm = (center.X(), center.Y(), center.Z())
        self.parts[key] = part
        return part

    # -- tree ----------------------------------------------------------------

    def _node(self, label: TDF_Label) -> Node:
        """Build the node for a label, following it if it is a placed reference."""
        transform = np.eye(4)
        occurrence_name = None
        definition = label
        if self.shapes.IsReference_s(label):
            definition = TDF_Label()
            self.shapes.GetReferredShape_s(label, definition)
            transform = _matrix(self.shapes.GetLocation_s(label))
            occurrence_name = _label_name(label)

        if self.shapes.IsAssembly_s(definition):
            components = Sequence_TDF_Label()
            self.shapes.GetComponents_s(definition, components)
            children = [self._node(components.Value(i)) for i in range(1, components.Length() + 1)]
            name = occurrence_name or _label_name(definition) or "Assembly"
            return Node(name=name, transform=transform, children=children)

        part = self._part(definition)
        return Node(name=occurrence_name or part.name, transform=transform, part=part)

    def read(self) -> Scene:
        free = Sequence_TDF_Label()
        self.shapes.GetFreeShapes(free)
        if free.Length() == 0:
            raise StepReadError(f"{self.path} contains no shapes")
        labels = [free.Value(i) for i in range(1, free.Length() + 1)]
        roots = [self._node(label) for label in labels]
        root = roots[0] if len(roots) == 1 else Node(name=self.path.stem, children=roots)

        # Exact bounds straight from the B-rep, independent of the tessellation,
        # so the written USD can be checked against the CAD model afterwards.
        box = Bnd_Box()
        for label in labels:
            BRepBndLib.AddOptimal_s(self.shapes.GetShape_s(label), box, False, False)
        bounds = None
        if not box.IsVoid():
            box.SetGap(0.0)
            lo, hi = box.CornerMin(), box.CornerMax()
            bounds = (np.array([lo.X(), lo.Y(), lo.Z()]), np.array([hi.X(), hi.Y(), hi.Z()]))

        return Scene(
            source=self.path.name, root=root, parts=list(self.parts.values()), bounds_mm=bounds
        )


def read_step(path: Path, linear_deflection: float = 0.1, angular_deflection: float = 0.5) -> Scene:
    """Read a STEP file.

    linear_deflection is the largest allowed gap between a curved surface and
    its triangles, in millimetres. angular_deflection is in radians.
    """
    path = Path(path)
    if not path.is_file():
        raise StepReadError(f"no such file: {path}")
    return _Reader(path, linear_deflection, angular_deflection).read()
