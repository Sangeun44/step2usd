"""Tests run against a generated assembly whose dimensions are known exactly."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Kind, Usd, UsdGeom, UsdPhysics

from step2usd.cli import convert, format_tree, main
from step2usd.reader import StepReadError, read_step
from step2usd.writer import WriteOptions, make_identifier, write_usd

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("make_sample", REPO / "examples" / "make_sample.py")
make_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_sample)

# Known geometry of the sample, in millimetres.
PLATE_VOLUME = 120 * 80 * 10 - 4 * math.pi * 4.5**2 * 10
BOLT_VOLUME = math.pi * 4.0**2 * 25 + math.pi * 6.5**2 * 5
UPRIGHT_VOLUME = 60 * 8 * 50
RIB_VOLUME = 0.5 * 20 * 30 * 6
STEEL = 7850.0


@pytest.fixture(scope="session")
def step_file(tmp_path_factory):
    return make_sample.write_sample(tmp_path_factory.mktemp("cad") / "bracket.step")


@pytest.fixture(scope="session")
def scene(step_file):
    return read_step(step_file)


def world_bounds_mm(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    scale = UsdGeom.GetStageMetersPerUnit(stage) * 1000.0
    return np.array(box.GetMin()) * scale, np.array(box.GetMax()) * scale


# --- reading ----------------------------------------------------------------


def test_reads_the_product_structure(scene):
    assert format_tree(scene).splitlines() == [
        "Bracket Assembly/",
        "  Base Plate (rev B)  [10 faces]",
        "  M8 Bolt  [5 faces, 1 of 4]",
        "  M8 Bolt  [5 faces, 1 of 4]",
        "  M8 Bolt  [5 faces, 1 of 4]",
        "  M8 Bolt  [5 faces, 1 of 4]",
        "  Upright Assy/",
        "    Upright  [6 faces]",
        "    Rib  [5 faces, 1 of 2]",
        "    Rib  [5 faces, 1 of 2]",
    ]
    assert [p.name for p in scene.parts] == ["Base Plate (rev B)", "M8 Bolt", "Upright", "Rib"]


def test_exact_volumes_come_from_the_brep(scene):
    volumes = {p.name: p.volume_mm3 for p in scene.parts}
    assert volumes["Base Plate (rev B)"] == pytest.approx(PLATE_VOLUME, rel=1e-9)
    assert volumes["M8 Bolt"] == pytest.approx(BOLT_VOLUME, rel=1e-9)
    assert volumes["Upright"] == pytest.approx(UPRIGHT_VOLUME, rel=1e-9)
    assert volumes["Rib"] == pytest.approx(RIB_VOLUME, rel=1e-9)
    assert all(p.is_solid and p.is_valid and p.failed_faces == 0 for p in scene.parts)


def test_tessellation_is_closed_and_wound_outwards(scene):
    for part in scene.parts:
        tri = part.mesh.points[part.mesh.triangles]
        # Divergence theorem: a closed, outward-wound mesh has a positive signed
        # volume close to the solid's. Flipped or missing faces break this.
        signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        assert signed == pytest.approx(part.volume_mm3, rel=0.02), part.name

        geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        authored = part.mesh.normals[part.mesh.triangles].sum(axis=1)
        assert (np.einsum("ij,ij->i", geometric, authored) > 0).all(), part.name
        assert np.allclose(np.linalg.norm(part.mesh.normals, axis=1), 1.0)


def test_finer_deflection_gives_more_triangles(step_file):
    coarse = read_step(step_file, linear_deflection=1.0)
    fine = read_step(step_file, linear_deflection=0.01)
    bolt = lambda s: next(p for p in s.parts if p.name == "M8 Bolt")  # noqa: E731
    assert len(bolt(fine).mesh.triangles) > len(bolt(coarse).mesh.triangles)


def test_file_units_do_not_change_the_result(tmp_path, scene):
    inches = read_step(make_sample.write_sample(tmp_path / "inch.step", unit="INCH"))
    assert "INCH" in (tmp_path / "inch.step").read_text().upper()
    for a, b in zip(scene.bounds_mm, inches.bounds_mm):
        assert np.allclose(a, b, atol=1e-6)


def test_unreadable_input_raises(tmp_path):
    with pytest.raises(StepReadError):
        read_step(tmp_path / "missing.step")
    garbage = tmp_path / "garbage.step"
    garbage.write_text("this is not a STEP file")
    with pytest.raises(StepReadError):
        read_step(garbage)


# --- writing ----------------------------------------------------------------


def test_repeated_parts_are_instanced_once(scene, tmp_path):
    result = write_usd(scene, tmp_path / "out.usda")
    stage = Usd.Stage.Open(str(result.path))

    assert (result.prototypes, result.instances) == (2, 6)
    assert len(stage.GetPrototypes()) == 2
    bolts = [stage.GetPrimAtPath(f"/Bracket_Assembly/M8_Bolt{s}") for s in ("", "_1", "_2", "_3")]
    assert all(b.IsInstance() for b in bolts)
    assert len({b.GetPrototype().GetPath() for b in bolts}) == 1
    # A part used once is written in place, with no indirection.
    assert not stage.GetPrimAtPath("/Bracket_Assembly/Base_Plate_rev_B").IsInstance()
    assert stage.GetPrimAtPath("/Bracket_Assembly/Base_Plate_rev_B/Mesh").IsA(UsdGeom.Mesh)
    # The prototype container is abstract, so it is not drawn or traversed.
    assert stage.GetPrimAtPath("/_Prototypes").IsAbstract()
    assert all(not p.GetPath().HasPrefix("/_Prototypes") for p in stage.Traverse())


def test_names_kinds_and_stage_metadata(scene, tmp_path):
    stage = Usd.Stage.Open(str(write_usd(scene, tmp_path / "out.usda").path))
    root = stage.GetDefaultPrim()
    assert root.GetPath() == "/Bracket_Assembly"
    assert root.GetDisplayName() == "Bracket Assembly"
    assert stage.GetPrimAtPath("/Bracket_Assembly/Base_Plate_rev_B").GetDisplayName() == "Base Plate (rev B)"
    kind = lambda p: Usd.ModelAPI(stage.GetPrimAtPath(p)).GetKind()  # noqa: E731
    assert kind("/Bracket_Assembly") == Kind.Tokens.assembly
    assert kind("/Bracket_Assembly/Upright_Assy") == Kind.Tokens.group
    assert kind("/Bracket_Assembly/Upright_Assy/Rib_1") == Kind.Tokens.component
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetRootLayer().customLayerData["step2usd"]["source"] == "bracket.step"


def test_nested_and_rotated_placements_land_where_the_cad_says(scene, tmp_path):
    stage = Usd.Stage.Open(str(write_usd(scene, tmp_path / "out.usda").path))
    # Upright Assy sits at (30, 36, 10); the second rib is turned 180 degrees about Z
    # and moved +55 in x inside it, so it occupies x 79..85, y 16..36, z 10..40.
    lo, hi = world_bounds_mm(stage, "/Bracket_Assembly/Upright_Assy/Rib_1")
    assert np.allclose(lo, [79, 16, 10], atol=1e-3)
    assert np.allclose(hi, [85, 36, 40], atol=1e-3)
    lo, hi = world_bounds_mm(stage, "/Bracket_Assembly/M8_Bolt_2")
    assert np.allclose(lo, [108 - 6.5, 68 - 6.5, -15], atol=0.1)
    assert np.allclose(hi, [108 + 6.5, 68 + 6.5, 15], atol=0.1)


@pytest.mark.parametrize("units", ["m", "mm"])
@pytest.mark.parametrize("up_axis", ["Z", "Y"])
@pytest.mark.parametrize("instancing", [True, False])
def test_bounds_match_the_cad_model_in_every_mode(step_file, tmp_path, units, up_axis, instancing):
    options = WriteOptions(units=units, up_axis=up_axis, instancing=instancing)
    report = convert(step_file, tmp_path / "out.usdc", options)
    check = report["bounds_check"]
    assert check["ok"], check
    stage = Usd.Stage.Open(str(tmp_path / "out.usdc"))
    assert UsdGeom.GetStageMetersPerUnit(stage) == (1.0 if units == "m" else 0.001)
    meshes = [p for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 8
    assert bool(stage.GetPrimAtPath("/_Prototypes")) == instancing


def test_bounds_check_catches_a_lost_placement_and_a_wrong_scale(scene, tmp_path):
    from step2usd.report import verify_bounds

    options = WriteOptions()
    result = write_usd(scene, tmp_path / "out.usda", options)
    assert verify_bounds(scene, result, options, tolerance_mm=0.2)["ok"]

    # Drop the sub-assembly's placement, as a converter bug would.
    stage = Usd.Stage.Open(str(result.path))
    UsdGeom.Xformable(stage.GetPrimAtPath("/Bracket_Assembly/Upright_Assy")).ClearXformOpOrder()
    stage.GetRootLayer().Save()
    assert not verify_bounds(scene, result, options, tolerance_mm=0.2)["ok"]

    # Geometry left in millimetres on a stage that claims metres.
    wrong_units = write_usd(scene, tmp_path / "mm.usda", WriteOptions(units="mm"))
    assert not verify_bounds(scene, wrong_units, WriteOptions(units="m"), tolerance_mm=0.2)["ok"]


def test_face_colours_and_face_ids(scene, tmp_path):
    result = write_usd(scene, tmp_path / "out.usda", WriteOptions(units="mm", face_ids=True))
    stage = Usd.Stage.Open(str(result.path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Bracket_Assembly/Base_Plate_rev_B/Mesh"))
    primvar = mesh.GetDisplayColorPrimvar()
    assert primvar.GetInterpolation() == UsdGeom.Tokens.uniform
    colors = np.array(primvar.Get())
    points = np.array(mesh.GetPointsAttr().Get())
    triangles = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    red = np.isclose(colors, [0.8, 0.1, 0.1], atol=1e-3).all(axis=1)
    # Only the top face (z = 10 mm) was painted red in the CAD model.
    assert red.any() and not red.all()
    assert np.allclose(points[triangles[red]][..., 2], 10.0)
    assert not np.isclose(points[triangles[~red]][..., 2], 10.0).all(axis=1).any()

    face_ids = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("cadFaceId").Get())
    assert len(face_ids) == len(triangles)
    assert sorted(set(face_ids.tolist())) == list(range(10))
    assert len(set(face_ids[red].tolist())) == 1


def test_density_authors_colliders_and_exact_mass(step_file, tmp_path):
    report = convert(step_file, tmp_path / "out.usda", WriteOptions(density=STEEL))
    stage = Usd.Stage.Open(str(tmp_path / "out.usda"))
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)

    plate = stage.GetPrimAtPath("/Bracket_Assembly/Base_Plate_rev_B/Mesh")
    assert plate.HasAPI(UsdPhysics.CollisionAPI)
    mass = UsdPhysics.MassAPI(plate)
    assert mass.GetMassAttr().Get() == pytest.approx(STEEL * PLATE_VOLUME * 1e-9, rel=1e-6)
    assert np.allclose(mass.GetCenterOfMassAttr().Get(), [0.060, 0.040, 0.005], atol=1e-7)

    total = PLATE_VOLUME + 4 * BOLT_VOLUME + UPRIGHT_VOLUME + 2 * RIB_VOLUME
    assert report["mass_kg"] == pytest.approx(STEEL * total * 1e-9, rel=1e-9)
    # Summing the authored masses through the instances gives the same total.
    authored = sum(
        UsdPhysics.MassAPI(p).GetMassAttr().Get()
        for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.MassAPI)
    )
    assert authored == pytest.approx(report["mass_kg"], rel=1e-6)


def test_no_physics_is_authored_without_a_density(scene, tmp_path):
    stage = Usd.Stage.Open(str(write_usd(scene, tmp_path / "out.usda").path))
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)


def test_converting_twice_to_the_same_path_works(step_file, tmp_path):
    first = convert(step_file, tmp_path / "out.usda", WriteOptions(instancing=False))
    second = convert(step_file, tmp_path / "out.usda")
    assert first["counts"]["instances"] == 0 and second["counts"]["instances"] == 6
    assert Usd.Stage.Open(str(tmp_path / "out.usda")).GetPrimAtPath("/_Prototypes")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("M8 Bolt", "M8_Bolt"),
        ("Base Plate (rev B)", "Base_Plate_rev_B"),
        ("6204-2RS", "_6204_2RS"),
        ("Gehäuse/Deckel", "Geh_use_Deckel"),
        ("***", "Part"),
        ("", "Part"),
        ("already_valid", "already_valid"),
    ],
)
def test_make_identifier(name, expected):
    assert make_identifier(name) == expected


def test_bad_options_are_rejected():
    for bad in ({"units": "cm"}, {"up_axis": "X"}, {"density": 0.0}):
        with pytest.raises(ValueError):
            WriteOptions(**bad)


# --- command line -----------------------------------------------------------


def test_cli_convert_writes_stage_and_report(step_file, tmp_path, capsys):
    out, report_path = tmp_path / "cli" / "bracket.usda", tmp_path / "cli" / "report.json"
    code = main(["convert", str(step_file), "-o", str(out), "--density", "7850", "--report", str(report_path)])
    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "assemblies": 2,
        "part_occurrences": 8,
        "unique_parts": 4,
        "prototypes": 2,
        "instances": 6,
        "triangles_stored": report["counts"]["triangles_stored"],
        "triangles_placed": report["counts"]["triangles_placed"],
    }
    assert report["counts"]["triangles_placed"] > report["counts"]["triangles_stored"]
    assert report["warnings"] == []
    assert {"cad_name": "M8 Bolt", "prim_name": "M8_Bolt"} in report["renamed"]
    assert "bounds match the CAD model" in capsys.readouterr().out


def test_cli_inspect_and_errors(step_file, tmp_path, capsys):
    assert main(["inspect", str(step_file)]) == 0
    assert "Upright Assy/" in capsys.readouterr().out
    assert main(["convert", str(tmp_path / "missing.step")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_committed_example_converts(tmp_path):
    report = convert(REPO / "examples" / "bracket_assembly.step", tmp_path / "out.usda")
    assert report["bounds_check"]["ok"]
    assert report["counts"]["part_occurrences"] == 8
