"""Tests for Gaussian splat import (3DGS .ply -> USD ParticleField) and CAD surface sampling."""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdVol

from step2usd.cli import main
from step2usd.reader import read_step
from step2usd.splat import (
    SH_C0,
    SplatData,
    SplatError,
    SplatWriteOptions,
    clean,
    convert_splat,
    read_ply,
    read_splat_usd,
    splats_from_scene,
    write_ply,
    write_splat_usd,
)

REPO = Path(__file__).resolve().parents[1]
STEP = REPO / "examples" / "bracket_assembly.step"
SAMPLE_PLY = REPO / "examples" / "bracket_splat.ply"


def random_splats(count: int, degree: int, seed: int = 1) -> SplatData:
    rng = np.random.default_rng(seed)
    rotations = rng.normal(size=(count, 4))
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    return SplatData(
        positions=rng.uniform(-2, 2, (count, 3)).astype(np.float32),
        rotations=rotations.astype(np.float32),
        scales=rng.uniform(0.01, 0.3, (count, 3)).astype(np.float32),
        opacities=rng.uniform(0.05, 0.95, count).astype(np.float32),
        sh=rng.normal(size=(count, (degree + 1) ** 2, 3)).astype(np.float32),
    )


def raw_ply(properties: list[str], rows: list[list[float]], fmt="binary_little_endian") -> bytes:
    """Build a PLY byte string by hand, independently of the module's own writer."""
    header = ["ply", f"format {fmt} 1.0", "comment hand built", f"element vertex {len(rows)}"]
    header += [f"property float {name}" for name in properties] + ["end_header", ""]
    body = b"".join(struct.pack(f"<{len(row)}f", *row) for row in rows)
    return "\n".join(header).encode("ascii") + body


STANDARD = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(9)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


# --- reading the PLY layout ---------------------------------------------------


def test_reads_the_standard_layout_and_applies_activations(tmp_path):
    row = (
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]  # position, unused normal
        + [0.1, 0.2, 0.3]  # f_dc: red, green, blue
        + [10, 11, 12, 20, 21, 22, 30, 31, 32]  # f_rest: three reds, three greens, three blues
        + [0.0]  # opacity logit -> 0.5
        + [math.log(0.5), math.log(2.0), 0.0]  # log scales
        + [2.0, 0.0, 0.0, 0.0]  # w, x, y, z, not normalised
    )
    path = tmp_path / "one.ply"
    path.write_bytes(raw_ply(STANDARD, [row]))
    data = read_ply(path)

    assert len(data) == 1 and data.degree == 1
    assert data.positions[0].tolist() == [1.0, 2.0, 3.0]
    assert data.opacities[0] == pytest.approx(0.5)
    assert data.scales[0].tolist() == pytest.approx([0.5, 2.0, 1.0])
    assert data.rotations[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    # Coefficients are regrouped per Gaussian: each one is an (r, g, b) triple.
    assert np.allclose(data.sh[0], [[0.1, 0.2, 0.3], [10, 20, 30], [11, 21, 31], [12, 22, 32]])
    assert data.colors()[0].tolist() == pytest.approx([0.5 + SH_C0 * c for c in (0.1, 0.2, 0.3)])


def test_property_order_in_the_file_does_not_matter(tmp_path):
    names = ["rot_3", "rot_2", "rot_1", "rot_0", "scale_2", "scale_1", "scale_0", "opacity",
             "f_dc_2", "f_dc_1", "f_dc_0", "z", "y", "x"]
    row = [0, 0, 0, 1, 0, 0, 0, 5, 0.3, 0.2, 0.1, 3, 2, 1]
    path = tmp_path / "shuffled.ply"
    path.write_bytes(raw_ply(names, [row]))
    data = read_ply(path)
    assert data.degree == 0
    assert data.positions[0].tolist() == [1.0, 2.0, 3.0]
    assert data.sh[0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert data.rotations[0].tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_ply_round_trip(tmp_path, degree):
    data = random_splats(200, degree)
    back = read_ply(write_ply(data, tmp_path / "rt.ply"))
    assert back.degree == degree
    assert np.allclose(back.positions, data.positions, atol=1e-6)
    assert np.allclose(back.rotations, data.rotations, atol=1e-6)
    assert np.allclose(back.scales, data.scales, rtol=1e-5)
    assert np.allclose(back.opacities, data.opacities, atol=1e-6)
    assert np.allclose(back.sh, data.sh, atol=1e-6)


def test_rejects_files_that_are_not_splats(tmp_path):
    with pytest.raises(SplatError, match="no such file"):
        read_ply(tmp_path / "missing.ply")

    (tmp_path / "text.ply").write_text("hello")
    with pytest.raises(SplatError, match="not a PLY"):
        read_ply(tmp_path / "text.ply")

    (tmp_path / "mesh.ply").write_bytes(raw_ply(["x", "y", "z"], [[0, 0, 0]]))
    with pytest.raises(SplatError, match="not a Gaussian splat"):
        read_ply(tmp_path / "mesh.ply")

    (tmp_path / "ascii.ply").write_bytes(raw_ply(STANDARD, [], fmt="ascii"))
    with pytest.raises(SplatError, match="binary_little_endian"):
        read_ply(tmp_path / "ascii.ply")

    whole = raw_ply(STANDARD, [[0.0] * len(STANDARD)] * 3)
    (tmp_path / "cut.ply").write_bytes(whole[:-10])
    with pytest.raises(SplatError, match="truncated"):
        read_ply(tmp_path / "cut.ply")

    odd = [n for n in STANDARD if n != "f_rest_8"]
    (tmp_path / "odd.ply").write_bytes(raw_ply(odd, [[0.0] * len(odd)]))
    with pytest.raises(SplatError, match="whole SH degree"):
        read_ply(tmp_path / "odd.ply")


# --- writing USD --------------------------------------------------------------


def test_writes_a_particle_field_prim_with_usd_conventions(tmp_path):
    data = random_splats(50, degree=2)
    write_splat_usd(data, tmp_path / "s.usda", SplatWriteOptions(up_axis="Y", meters_per_unit=0.5))
    stage = Usd.Stage.Open(str(tmp_path / "s.usda"))

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.5
    prim = stage.GetPrimAtPath("/Capture/Splat")
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat"
    assert stage.GetDefaultPrim().GetPath() == "/Capture"

    splat = UsdVol.ParticleField3DGaussianSplat(prim)
    assert len(splat.GetPositionsAttr().Get()) == 50
    assert splat.GetRadianceSphericalHarmonicsDegreeAttr().Get() == 2
    assert len(splat.GetRadianceSphericalHarmonicsCoefficientsAttr().Get()) == 50 * 9
    # USD wants linear scales and opacities in 0..1, not the PLY's log and logit values.
    assert np.allclose(np.array(splat.GetScalesAttr().Get()), data.scales)
    opacities = np.array(splat.GetOpacitiesAttr().Get())
    assert opacities.min() >= 0.0 and opacities.max() <= 1.0
    # Coefficients are grouped per Gaussian: the first nine belong to Gaussian 0.
    coefficients = np.array(splat.GetRadianceSphericalHarmonicsCoefficientsAttr().Get())
    assert np.allclose(coefficients[:9], data.sh[0])

    # Quaternions keep their meaning through USD's own quaternion type.
    quat = splat.GetOrientationsAttr().Get()[7]
    w, x, y, z = data.rotations[7]
    assert quat.GetReal() == pytest.approx(w)
    assert list(quat.GetImaginary()) == pytest.approx([x, y, z])

    lo, hi = (np.array(v) for v in splat.GetExtentAttr().Get())
    reach = 3 * data.scales.max(axis=1, keepdims=True)
    assert (data.positions - reach >= lo - 1e-5).all() and (data.positions + reach <= hi + 1e-5).all()


def test_usd_round_trip_and_rewrite_to_the_same_path(tmp_path):
    for degree in (0, 3):
        data = random_splats(120, degree, seed=degree)
        back = read_splat_usd(write_splat_usd(data, tmp_path / "s.usdc"))
        for name in ("positions", "rotations", "scales", "opacities", "sh"):
            assert np.array_equal(getattr(back, name), getattr(data, name)), name


def test_rotate_x_turns_the_root_and_leaves_the_data_alone(tmp_path):
    data = random_splats(30, degree=0)
    write_splat_usd(data, tmp_path / "s.usda", SplatWriteOptions(rotate_x=180.0))
    stage = Usd.Stage.Open(str(tmp_path / "s.usda"))
    assert np.array_equal(read_splat_usd(tmp_path / "s.usda").positions, data.positions)
    prim = stage.GetPrimAtPath("/Capture/Splat")
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    moved = world.Transform(Gf.Vec3d(*data.positions[0].tolist()))
    x, y, z = data.positions[0]
    assert list(moved) == pytest.approx([x, -y, -z], abs=1e-5)


def test_clean_drops_unusable_gaussians():
    data = random_splats(10, degree=0)
    data.positions[2, 1] = np.nan
    data.scales[5, 0] = np.inf
    data.opacities[7] = 0.001
    cleaned, dropped = clean(data, min_opacity=0.02)
    assert dropped == {"non_finite": 2, "below_min_opacity": 1}
    assert len(cleaned) == 7
    assert np.isfinite(cleaned.positions).all()
    with pytest.raises(SplatError):
        write_splat_usd(data.select(np.zeros(10, bool)), Path("unused.usda"))


# --- CAD surfaces to Gaussians ------------------------------------------------


@pytest.fixture(scope="module")
def sampled():
    return splats_from_scene(read_step(STEP), count=6000)


def test_sampled_gaussians_lie_on_the_cad_surfaces(sampled):
    scene = read_step(STEP)
    lo, hi = (b * 0.001 for b in scene.bounds_mm)
    assert len(sampled) == 6000 and sampled.degree == 0
    assert (sampled.positions >= lo - 1e-6).all() and (sampled.positions <= hi + 1e-6).all()
    # The model reaches every side of its bounding box, so the samples should too.
    assert np.allclose(sampled.positions.min(axis=0), lo, atol=2e-3)
    assert np.allclose(sampled.positions.max(axis=0), hi, atol=2e-3)
    # The plate's top face (z = 10 mm) was painted red in the CAD model. Look at a
    # patch of it that no other part touches: every sample there is red, and
    # nothing away from that height is.
    x, y, z = sampled.positions.T
    at_top = np.isclose(z, 0.010, atol=1e-6)
    patch = at_top & (x > 0.045) & (x < 0.075) & (y > 0.050) & (y < 0.075)
    colors = sampled.colors()
    reddish = (colors[:, 0] > 2 * colors[:, 1]) & (colors[:, 0] > 2 * colors[:, 2])
    assert patch.sum() > 50
    assert reddish[patch].all()
    assert not reddish[~at_top].any()


def test_sampled_gaussians_are_flat_and_face_outwards(sampled, tmp_path):
    # Thin along local Z, round in the surface plane.
    assert np.allclose(sampled.scales[:, 0], sampled.scales[:, 1])
    assert (sampled.scales[:, 2] < 0.2 * sampled.scales[:, 0]).all()

    # Check orientation through USD's quaternion maths, on what is actually written.
    write_splat_usd(sampled, tmp_path / "s.usda")
    stage = Usd.Stage.Open(str(tmp_path / "s.usda"))
    splat = UsdVol.ParticleField3DGaussianSplat(stage.GetPrimAtPath("/Capture/Splat"))
    positions = np.array(splat.GetPositionsAttr().Get())
    quats = splat.GetOrientationsAttr().Get()
    x, y, z = positions.T
    clear = (x > 0.045) & (x < 0.075) & (y > 0.050) & (y < 0.075)  # no other part sits here
    top = np.flatnonzero(np.isclose(z, 0.010, atol=1e-6) & clear)[:20]
    assert len(top) == 20
    for i in top:  # plate top faces +Z
        assert list(quats[int(i)].Transform(Gf.Vec3f(0, 0, 1))) == pytest.approx([0, 0, 1], abs=1e-5)
    left = np.flatnonzero(np.isclose(positions[:, 0], 0.0, atol=1e-6))[:20]
    assert len(left) > 0
    for i in left:  # plate's x = 0 side faces -X
        assert list(quats[int(i)].Transform(Gf.Vec3f(0, 0, 1))) == pytest.approx([-1, 0, 0], abs=1e-5)


def test_sampling_is_even_and_repeatable(sampled):
    again = splats_from_scene(read_step(STEP), count=6000)
    assert np.array_equal(again.positions, sampled.positions)

    # On the big flat top face, count samples per 10 mm tile: none should be empty or crowded.
    top = sampled.positions[np.isclose(sampled.positions[:, 2], 0.010, atol=1e-6)]
    inner = top[(top[:, 0] > 0.03) & (top[:, 0] < 0.09) & (top[:, 1] > 0.045) & (top[:, 1] < 0.075)]
    tiles, _, _ = np.histogram2d(inner[:, 0], inner[:, 1], bins=(6, 3), range=((0.03, 0.09), (0.045, 0.075)))
    assert tiles.min() > 0.5 * tiles.mean() and tiles.max() < 1.6 * tiles.mean()


# --- command line and sample data ---------------------------------------------


def test_cli_sample_then_convert(tmp_path, capsys):
    ply, usd, report_path = tmp_path / "b.ply", tmp_path / "b.usdc", tmp_path / "r.json"
    assert main(["sample-splat", str(STEP), "-o", str(ply), "--count", "2000"]) == 0
    assert main(["splat", str(ply), "-o", str(usd), "--min-opacity", "0.02", "--report", str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "2000 Gaussians" in out and "read back from USD: identical" in out
    report = json.loads(report_path.read_text())
    assert report["gaussians_written"] == 2000 and report["sh_degree"] == 0
    assert report["round_trip"]["ok"]
    # The splat occupies the same space as the CAD model it was sampled from (metres).
    size = np.array(report["bounds"][1]) - np.array(report["bounds"][0])
    assert np.allclose(size, [0.120, 0.080, 0.075], atol=0.03)


def test_cli_reports_bad_input(tmp_path, capsys):
    (tmp_path / "mesh.ply").write_bytes(raw_ply(["x", "y", "z"], [[0, 0, 0]]))
    assert main(["splat", str(tmp_path / "mesh.ply")]) == 2
    assert "not a Gaussian splat" in capsys.readouterr().err


def test_committed_sample_converts(tmp_path):
    report = convert_splat(SAMPLE_PLY, tmp_path / "sample.usdc")
    assert report["gaussians_written"] == 12000
    assert report["round_trip"]["ok"]
