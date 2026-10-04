"""Generate the sample STEP assembly used by the tests and the README.

    python examples/make_sample.py examples/bracket_assembly.step

The assembly is built to exercise the converter, in millimetres:

    Bracket Assembly
      Base Plate (rev B)     120 x 80 x 10 plate with four 9 mm holes, one painted face
      M8 Bolt  x4            the same part placed four times
      Upright Assy           a nested sub-assembly
        Upright              60 x 8 x 50 wall
        Rib  x2              the same wedge placed twice, once turned 180 degrees
"""

from __future__ import annotations

import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeWedge
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

PLATE = (120.0, 80.0, 10.0)
HOLE_DIAMETER = 9.0
HOLE_INSET = 12.0
BOLT_POSITIONS = [
    (HOLE_INSET, HOLE_INSET),
    (PLATE[0] - HOLE_INSET, HOLE_INSET),
    (PLATE[0] - HOLE_INSET, PLATE[1] - HOLE_INSET),
    (HOLE_INSET, PLATE[1] - HOLE_INSET),
]


def _vertical_axis(x: float, y: float, z: float = 0.0) -> gp_Ax2:
    return gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1))


def base_plate():
    shape = BRepPrimAPI_MakeBox(*PLATE).Shape()
    for x, y in BOLT_POSITIONS:
        hole = BRepPrimAPI_MakeCylinder(_vertical_axis(x, y), HOLE_DIAMETER / 2, PLATE[2]).Shape()
        shape = BRepAlgoAPI_Cut(shape, hole).Shape()
    return shape


def bolt():
    """M8 bolt, origin at the underside of the head, shank pointing down -Z."""
    shank = BRepPrimAPI_MakeCylinder(_vertical_axis(0, 0, -25.0), 4.0, 25.0).Shape()
    head = BRepPrimAPI_MakeCylinder(_vertical_axis(0, 0, 0.0), 6.5, 5.0).Shape()
    return BRepAlgoAPI_Fuse(shank, head).Shape()


def rib():
    """Triangular gusset: 6 thick (x), 20 deep (y), 30 tall (z), vertical edge on y = 0."""
    wedge = BRepPrimAPI_MakeWedge(20.0, 30.0, 6.0, 0.0).Shape()
    # The wedge is built depth-x, height-y, thickness-z; cycle the axes so it stands up.
    stand = gp_Trsf()
    stand.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 1)), 2.0 * 3.141592653589793 / 3.0)
    return BRepBuilderAPI_Transform(wedge, stand, True).Shape()


def _translation(x: float, y: float, z: float) -> TopLoc_Location:
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(x, y, z))
    return TopLoc_Location(trsf)


def _top_face(shape, z: float):
    """The largest planar face lying at height z."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    best, best_area = None, 0.0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(explorer.Current(), props)
        if abs(props.CentreOfMass().Z() - z) < 1e-6 and props.Mass() > best_area:
            best, best_area = explorer.Current(), props.Mass()
        explorer.Next()
    return best


def build_document() -> TDocStd_Document:
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    def name(label, text: str):
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(text))
        return label

    def part(shape, text: str, rgb):
        label = name(shapes.AddShape(shape, False), text)
        colors.SetColor(label, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorGen)
        return label

    plate_shape = base_plate()
    plate = part(plate_shape, "Base Plate (rev B)", (0.55, 0.57, 0.6))
    colors.SetColor(
        _top_face(plate_shape, PLATE[2]),
        Quantity_Color(0.8, 0.1, 0.1, Quantity_TOC_RGB),
        XCAFDoc_ColorSurf,
    )
    bolt_part = part(bolt(), "M8 Bolt", (0.1, 0.1, 0.12))
    upright = part(BRepPrimAPI_MakeBox(60.0, 8.0, 50.0).Shape(), "Upright", (0.2, 0.35, 0.7))
    rib_part = part(rib(), "Rib", (0.2, 0.35, 0.7))

    upright_assy = name(shapes.NewShape(), "Upright Assy")
    shapes.AddComponent(upright_assy, upright, _translation(0.0, 0.0, 0.0))
    shapes.AddComponent(upright_assy, rib_part, _translation(5.0, 8.0, 0.0))
    turned = gp_Trsf()
    turned.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 3.141592653589793)
    moved = gp_Trsf()
    moved.SetTranslation(gp_Vec(55.0, 0.0, 0.0))
    shapes.AddComponent(upright_assy, rib_part, TopLoc_Location(moved.Multiplied(turned)))

    root = name(shapes.NewShape(), "Bracket Assembly")
    shapes.AddComponent(root, plate, _translation(0.0, 0.0, 0.0))
    for x, y in BOLT_POSITIONS:
        shapes.AddComponent(root, bolt_part, _translation(x, y, PLATE[2]))
    shapes.AddComponent(root, upright_assy, _translation(30.0, 36.0, PLATE[2]))

    shapes.UpdateAssemblies()
    return doc


def write_sample(path: Path, unit: str = "MM") -> Path:
    """Write the sample. unit is the length unit declared in the file: "MM", "M" or "INCH"."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Interface_Static.SetCVal_s("write.step.unit", unit)
    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    if not writer.Transfer(build_document(), STEPControl_AsIs):
        raise RuntimeError("could not transfer the document to STEP")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not write {path}")
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/bracket_assembly.step")
    print(f"wrote {write_sample(target)}")
