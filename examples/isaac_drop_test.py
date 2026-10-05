"""Drop a converted assembly in NVIDIA Isaac Sim and check the simulation against the CAD numbers.

Convert with a density first, so the stage carries colliders and exact masses:

    step2usd convert examples/bracket_assembly.step -o out/bracket.usda --density 7850

Then run this script with Isaac Sim's own Python (not your project venv):

    C:\\isaacsim\\python.bat examples\\isaac_drop_test.py out\\bracket.usda      (Windows)
    ~/isaacsim/python.sh examples/isaac_drop_test.py out/bracket.usda           (Linux)

What it does:

  1. Builds a small scene next to the asset: ground, gravity, lights, a camera,
     and the asset referenced in above the ground.  (plain USD, no Isaac needed)
  2. Opens the scene in Isaac Sim and asks PhysX what the rigid body weighs.
  3. Plays the simulation until the part comes to rest.
  4. Compares three things with what the CAD model says they should be:
     the mass PhysX computed, the height the part rests at, and how far it tilted.
  5. Saves a rendered image and a JSON result.

Flags:  --gui shows the Isaac Sim window.  --build-only stops after step 1; the
scene file can then be opened in Isaac Sim by hand (File > Open, then Play).

This file only imports pxr, numpy and the standard library at the top, so the
scene-building half can be run and tested without Isaac Sim installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ASSET_PATH = "/World/Asset"
CAMERA_PATH = "/World/Camera"
REST_TOLERANCE_M = 0.003  # contact offsets keep bodies a hair above the surface
TILT_TOLERANCE_DEG = 2.0
MASS_TOLERANCE = 1e-3  # relative


class SceneError(ValueError):
    pass


def _asset_facts(asset_path: Path) -> dict:
    """Read what the drop test needs from the converted asset: bounds and authored mass."""
    if not asset_path.is_file():
        raise SceneError(f"no such file: {asset_path}")
    stage = Usd.Stage.Open(str(asset_path))
    root = stage.GetDefaultPrim()
    if not root:
        raise SceneError(f"{asset_path} has no defaultPrim")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SceneError("the drop test expects a Z-up asset; convert with --up-axis Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise SceneError("the drop test expects metres; convert with --units m")
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise SceneError("the asset has no rigid body; convert with --density, e.g. --density 7850")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    mass = sum(
        UsdPhysics.MassAPI(prim).GetMassAttr().Get() or 0.0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.MassAPI)
    )
    return {"min": list(box.GetMin()), "max": list(box.GetMax()), "mass_kg": mass}


def _reference_path(asset_path: Path, scene_path: Path) -> str:
    try:
        relative = os.path.relpath(asset_path.resolve(), scene_path.resolve().parent)
        return "./" + relative.replace(os.sep, "/")
    except ValueError:  # different drives on Windows
        return asset_path.resolve().as_posix()


def build_drop_scene(asset_path: Path, scene_path: Path, drop_height: float = 0.1) -> dict:
    """Write the drop scene and return the values the simulation is expected to reproduce."""
    asset_path, scene_path = Path(asset_path), Path(scene_path)
    facts = _asset_facts(asset_path)
    lo, hi = Gf.Vec3d(*facts["min"]), Gf.Vec3d(*facts["max"])
    size = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    # Height of the asset's origin when its lowest point touches the ground.
    rest_height = -lo[2]

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    cached = Sdf.Layer.Find(str(scene_path))  # re-running in one process must not clash
    if cached:
        cached.Clear()
        stage = Usd.Stage.Open(cached)
    else:
        stage = Usd.Stage.CreateNew(str(scene_path))
    stage.SetTimeCodesPerSecond(60.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr(9.81)

    # Ground: a wide slab whose top face is the z = 0 plane.
    extent, thickness = max(10.0 * size, 1.0), 0.02
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
    ground_xform = UsdGeom.XformCommonAPI(ground)
    ground_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -thickness / 2))
    ground_xform.SetScale(Gf.Vec3f(extent, extent, thickness))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    asset = UsdGeom.Xform.Define(stage, ASSET_PATH)
    asset.GetPrim().GetReferences().AddReference(_reference_path(asset_path, scene_path))
    centre_x, centre_y = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    start = Gf.Vec3d(-centre_x, -centre_y, rest_height + drop_height)
    asset.AddTranslateOp().Set(start)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.5)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(50.0, 0.0, 30.0))

    # Camera aimed at where the part will come to rest.
    target = Gf.Vec3d(0.0, 0.0, (hi[2] - lo[2]) / 2)
    direction = Gf.Vec3d(0.8, -1.0, 0.65).GetNormalized()
    eye = target + direction * (3.2 * size)
    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))  # the default near plane is 1 m
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera.AddTransformOp().Set(view.GetInverse())

    stage.GetRootLayer().customLayerData = {"step2usd": {"example": "isaac_drop_test", "asset": asset_path.name}}
    stage.GetRootLayer().Save()

    return {
        "asset": asset_path.name,
        "scene": scene_path.name,
        "drop_height_m": drop_height,
        "start_position_m": list(start),
        "expected": {
            "mass_kg": facts["mass_kg"],
            "rest_height_m": rest_height,
            "rest_position_m": [start[0], start[1], rest_height],
        },
    }


def evaluate(expected: dict, measured: dict) -> dict:
    """Compare what the simulator reported with what the CAD model implies."""
    checks = {}
    if measured.get("mass_kg") is not None:
        error = abs(measured["mass_kg"] - expected["mass_kg"]) / expected["mass_kg"]
        checks["mass"] = {"ok": error <= MASS_TOLERANCE, "relative_error": error}
    height_error = abs(measured["rest_position_m"][2] - expected["rest_height_m"])
    checks["rest_height"] = {"ok": height_error <= REST_TOLERANCE_M, "error_m": height_error}
    checks["tilt"] = {"ok": measured["tilt_deg"] <= TILT_TOLERANCE_DEG, "tilt_deg": measured["tilt_deg"]}
    checks["settled"] = {"ok": bool(measured["settled"]), "seconds": measured["seconds"]}
    return {"passed": all(c["ok"] for c in checks.values()), "checks": checks}


def _pose(stage: Usd.Stage) -> tuple[list[float], float]:
    """World position of the asset and its tilt away from upright, in degrees."""
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(ASSET_PATH))
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    up = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
    return list(matrix.ExtractTranslation()), tilt


def run_in_isaac(scene_path: Path, plan: dict, gui: bool, max_seconds: float, image_path: Path) -> dict:
    """Open the scene in Isaac Sim, simulate until rest, and return the measurements."""
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise SceneError(
            "Isaac Sim is not available in this Python. Run this script with Isaac Sim's "
            "python.bat / python.sh, or pass --build-only and open the scene by hand."
        ) from exc

    app = SimulationApp({"headless": not gui})

    # Omniverse modules can only be imported once the app is running.
    import omni.timeline
    import omni.usd
    from pxr import PhysicsSchemaTools, UsdUtils

    context = omni.usd.get_context()
    if not context.open_stage(str(scene_path.resolve())):
        app.close()
        raise SceneError(f"Isaac Sim could not open {scene_path}")
    stage = context.get_stage()
    for _ in range(10):
        app.update()

    measured: dict = {"mass_kg": None, "center_of_mass_m": None}

    # Ask PhysX for the mass it computed from the colliders and MassAPI data.
    try:
        from omni.physx import get_physx_property_query_interface
        from omni.physx.bindings._physx import PhysxPropertyQueryResult

        done = {"finished": False}

        def on_rigid_body(info):
            if info.result == PhysxPropertyQueryResult.VALID:
                measured["mass_kg"] = float(info.mass)
                measured["center_of_mass_m"] = [float(c) for c in info.center_of_mass]

        get_physx_property_query_interface().query_prim(
            stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
            prim_id=PhysicsSchemaTools.sdfPathToInt(ASSET_PATH),
            rigid_body_fn=on_rigid_body,
            finished_fn=lambda: done.update(finished=True),
            timeout_ms=5000,
        )
        for _ in range(300):  # the answer arrives on a later frame
            if done["finished"]:
                break
            app.update()
    except Exception as exc:  # keep going: the drop itself is still worth running
        print(f"[drop-test] mass query unavailable: {exc}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    start_height, drop_height = plan["start_position_m"][2], plan["drop_height_m"]
    heights: list[float] = []
    settled = False
    for _ in range(int(max_seconds * 240)):  # frame cap; simulated time is the real limit
        app.update()
        position, _ = _pose(stage)
        heights.append(position[2])
        window = heights[-30:]
        has_fallen = drop_height <= 0 or position[2] < start_height - 0.5 * drop_height
        # At rest once 30 frames in a row stay within 10 microns, after the fall has happened.
        if has_fallen and len(window) == 30 and max(window) - min(window) < 1e-5:
            settled = True
            break
        if timeline.get_current_time() >= max_seconds:
            break
    seconds = timeline.get_current_time()
    timeline.pause()  # stop() would reset the part to where it started

    position, tilt = _pose(stage)
    measured.update(
        {"rest_position_m": position, "tilt_deg": tilt, "settled": settled, "seconds": seconds}
    )

    try:
        import omni.replicator.core as rep
        from PIL import Image

        render_product = rep.create.render_product(CAMERA_PATH, (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(render_product)
        rep.orchestrator.step(rt_subframes=32)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb.get_data()[..., :3]).save(image_path)
        measured["image"] = image_path.name
    except Exception as exc:
        print(f"[drop-test] image capture unavailable: {exc}")

    result = {**plan, "measured": measured, **evaluate(plan["expected"], measured)}
    # Report before closing: SimulationApp.close() can end the process.
    _report(result, image_path.with_suffix(".json"))
    app.close()
    return result


def _report(result: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    expected, measured = result["expected"], result["measured"]
    print(f"[drop-test] {result['asset']} dropped from {result['drop_height_m']} m")
    if measured["mass_kg"] is not None:
        print(f"[drop-test]   mass   PhysX {measured['mass_kg']:.4f} kg, CAD {expected['mass_kg']:.4f} kg")
    print(
        f"[drop-test]   rest   z = {measured['rest_position_m'][2]:.4f} m, expected {expected['rest_height_m']:.4f} m"
    )
    state = "settled after" if measured["settled"] else "still moving after"
    print(f"[drop-test]   tilt   {measured['tilt_deg']:.2f} degrees, {state} {measured['seconds']:.2f} s")
    for name, check in result["checks"].items():
        print(f"[drop-test]   {'ok  ' if check['ok'] else 'FAIL'} {name}")
    print(f"[drop-test] {'PASS' if result['passed'] else 'FAIL'}  (details in {json_path})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("asset", type=Path, help="a stage written by 'step2usd convert --density ...'")
    parser.add_argument("--drop-height", type=float, default=0.1, metavar="M", help="default: 0.1")
    parser.add_argument("--max-seconds", type=float, default=5.0, help="give up waiting for rest after this")
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    parser.add_argument("--build-only", action="store_true", help="write the scene and stop")
    args = parser.parse_args(argv)

    scene_path = args.asset.with_name(args.asset.stem + "_drop_scene.usda")
    try:
        plan = build_drop_scene(args.asset, scene_path, args.drop_height)
    except SceneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"[drop-test] wrote {scene_path}")
    if args.build_only:
        print(json.dumps(plan["expected"], indent=2))
        return 0

    image_path = args.asset.with_name(args.asset.stem + "_drop_test.png")
    try:
        result = run_in_isaac(scene_path, plan, args.gui, args.max_seconds, image_path)
    except SceneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
