"""Tests for the half of the Isaac Sim example that is plain USD and runs anywhere.

The simulation itself needs Isaac Sim and an RTX GPU, so it is not exercised here.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from step2usd.cli import convert
from step2usd.writer import WriteOptions

REPO = Path(__file__).resolve().parents[1]
STEP = REPO / "examples" / "bracket_assembly.step"

_spec = importlib.util.spec_from_file_location("isaac_drop_test", REPO / "examples" / "isaac_drop_test.py")
drop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drop)


@pytest.fixture
def asset(tmp_path):
    report = convert(STEP, tmp_path / "bracket.usda", WriteOptions(density=7850.0))
    return tmp_path / "bracket.usda", report


def test_scene_is_built_around_the_asset(asset, tmp_path):
    asset_path, report = asset
    plan = drop.build_drop_scene(asset_path, tmp_path / "scene.usda", drop_height=0.1)
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetPrimAtPath("/World/PhysicsScene").IsA(UsdPhysics.Scene)
    assert stage.GetPrimAtPath("/World/Ground").HasAPI(UsdPhysics.CollisionAPI)
    # The asset arrives by reference with its rigid body and its 8 colliders intact.
    body = stage.GetPrimAtPath(drop.ASSET_PATH)
    assert body.HasAPI(UsdPhysics.RigidBodyAPI)
    colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 8
    assert not stage.GetPrimAtPath("/World/Ground").HasAPI(UsdPhysics.RigidBodyAPI)  # static

    # The bolt tips are the lowest point, 15 mm under the plate, so that is the rest height.
    assert plan["expected"]["rest_height_m"] == pytest.approx(0.015, abs=1e-6)
    assert plan["expected"]["mass_kg"] == pytest.approx(report["mass_kg"], rel=1e-6)


def test_asset_starts_centred_and_clear_of_the_ground(asset, tmp_path):
    asset_path, _ = asset
    drop.build_drop_scene(asset_path, tmp_path / "scene.usda", drop_height=0.25)
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(drop.ASSET_PATH)).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    assert lo[2] == pytest.approx(0.25, abs=1e-6)  # lowest point is exactly the drop height up
    assert np.allclose((lo[:2] + hi[:2]) / 2, 0.0, atol=1e-6)

    ground = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Ground")).ComputeAlignedRange()
    assert ground.GetMax()[2] == pytest.approx(0.0, abs=1e-9)  # top of the slab is z = 0
    assert ground.GetMin()[0] < lo[0] and ground.GetMax()[0] > hi[0]


def test_camera_frames_the_part_at_rest(asset, tmp_path):
    asset_path, _ = asset
    plan = drop.build_drop_scene(asset_path, tmp_path / "scene.usda")
    stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
    # Move the asset to where the simulation should leave it.
    UsdGeom.Xformable(stage.GetPrimAtPath(drop.ASSET_PATH)).GetOrderedXformOps()[0].Set(
        Gf.Vec3d(*plan["expected"]["rest_position_m"])
    )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(drop.ASSET_PATH)).ComputeAlignedRange()

    camera = UsdGeom.Camera(stage.GetPrimAtPath(drop.CAMERA_PATH)).GetCamera()
    frustum = camera.frustum
    # 16:9 render: keep the horizontal field of view, narrow the vertical one.
    frustum.SetPerspective(camera.GetFieldOfView(Gf.Camera.FOVHorizontal), True, 16 / 9, 0.01, 1000.0)
    view_projection = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
    for i in range(8):
        ndc = view_projection.Transform(box.GetCorner(i))
        assert all(-0.9 < c < 0.9 for c in (ndc[0], ndc[1])), f"corner {i} is out of frame: {ndc}"
        assert -1.0 < ndc[2] < 1.0
    near, _far = UsdGeom.Camera(stage.GetPrimAtPath(drop.CAMERA_PATH)).GetClippingRangeAttr().Get()
    assert near < 0.1  # USD's default near plane of 1 m would clip a part this small


def test_rejects_assets_the_drop_test_cannot_use(tmp_path):
    convert(STEP, tmp_path / "no_physics.usda")
    with pytest.raises(drop.SceneError, match="--density"):
        drop.build_drop_scene(tmp_path / "no_physics.usda", tmp_path / "scene.usda")
    convert(STEP, tmp_path / "y_up.usda", WriteOptions(density=7850.0, up_axis="Y"))
    with pytest.raises(drop.SceneError, match="Z-up"):
        drop.build_drop_scene(tmp_path / "y_up.usda", tmp_path / "scene.usda")
    convert(STEP, tmp_path / "mm.usda", WriteOptions(density=7850.0, units="mm"))
    with pytest.raises(drop.SceneError, match="metres"):
        drop.build_drop_scene(tmp_path / "mm.usda", tmp_path / "scene.usda")


def test_evaluate_passes_good_results_and_fails_bad_ones():
    expected = {"mass_kg": 1.0106, "rest_height_m": 0.015}
    good = {"mass_kg": 1.0107, "rest_position_m": [0, 0, 0.0158], "tilt_deg": 0.1, "settled": True, "seconds": 0.9}
    assert drop.evaluate(expected, good)["passed"]

    for change, failing in [
        ({"mass_kg": 0.9}, "mass"),  # e.g. instanced colliders were not picked up
        ({"rest_position_m": [0, 0, 0.0]}, "rest_height"),  # sank into the ground
        ({"tilt_deg": 90.0}, "tilt"),  # tipped over
        ({"settled": False}, "settled"),
    ]:
        result = drop.evaluate(expected, {**good, **change})
        assert not result["passed"]
        assert [name for name, c in result["checks"].items() if not c["ok"]] == [failing]

    # Without a mass reading the other checks still decide the outcome.
    assert "mass" not in drop.evaluate(expected, {**good, "mass_kg": None})["checks"]


def test_build_only_command(asset, capsys):
    asset_path, _ = asset
    assert drop.main([str(asset_path), "--build-only"]) == 0
    out = capsys.readouterr().out
    assert asset_path.with_name("bracket_drop_scene.usda").is_file()
    assert json.loads(out[out.index("{"):])["rest_height_m"] == pytest.approx(0.015, abs=1e-6)
    assert drop.main([str(asset_path.with_name("missing.usda")), "--build-only"]) == 2
