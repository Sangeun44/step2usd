"""Conversion report: what was read, what was written, and whether the two agree."""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from .model import Scene
from .writer import MM_TO_M, WriteOptions, WriteResult


def verify_bounds(scene: Scene, result: WriteResult, options: WriteOptions, tolerance_mm: float) -> dict:
    """Compare the USD stage's bounding box with the exact one from the CAD model.

    The two are computed independently: one from the B-rep by the CAD kernel,
    the other by USD from the meshes, transforms and instances that were
    written. A wrong unit scale, a transposed matrix or a dropped instance
    shows up here as a mismatch.
    """
    if scene.bounds_mm is None:
        return {"ok": False, "reason": "the CAD model has no bounds"}

    lo, hi = (b * options.scale for b in scene.bounds_mm)
    if options.up_axis == "Y":  # the writer maps CAD (x, y, z) to stage (x, z, -y)
        lo, hi = np.array([lo[0], lo[2], -hi[1]]), np.array([hi[0], hi[2], -lo[1]])

    stage = Usd.Stage.Open(str(result.path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if box.IsEmpty():
        return {"ok": False, "reason": "the USD stage has no geometry"}
    usd_lo, usd_hi = np.array(box.GetMin()), np.array(box.GetMax())

    tolerance = tolerance_mm * options.scale
    deviation = float(max(np.abs(usd_lo - lo).max(), np.abs(usd_hi - hi).max()))
    return {
        "ok": deviation <= tolerance,
        "max_deviation": deviation,
        "tolerance": tolerance,
        "cad_bounds": [lo.tolist(), hi.tolist()],
        "usd_bounds": [usd_lo.tolist(), usd_hi.tolist()],
        "units": options.units,
    }


def build_report(
    scene: Scene, result: WriteResult, options: WriteOptions, linear_deflection: float
) -> dict:
    counts = scene.occurrences()
    nodes = list(scene.root.walk())
    warnings: list[str] = []
    parts = []
    total_volume_mm3 = 0.0

    for part in scene.parts:
        n = counts[part.key]
        triangles = len(part.mesh.triangles)
        entry = {
            "name": part.name,
            "occurrences": n,
            "instanced": options.instancing and n > 1,
            "faces": part.face_count,
            "triangles": triangles,
            "solid": part.is_solid,
        }
        if part.volume_mm3 is not None:
            total_volume_mm3 += part.volume_mm3 * n
            entry["volume_cm3"] = part.volume_mm3 / 1000.0
            if options.density is not None:
                entry["mass_kg"] = options.density * part.volume_mm3 * MM_TO_M**3
        parts.append(entry)

        if part.mesh.is_empty:
            warnings.append(f"'{part.name}' produced no triangles and was written without geometry")
        elif part.failed_faces:
            warnings.append(f"'{part.name}': {part.failed_faces} of {part.face_count} faces failed to mesh")
        if not part.is_valid:
            warnings.append(f"'{part.name}' failed the CAD kernel's validity check")
        if not part.is_solid:
            warnings.append(f"'{part.name}' is not a solid, so it has no volume or mass")

    for path in result.mirrored:
        warnings.append(f"{path} has a mirroring transform; check its face winding in your renderer")

    unique_triangles = sum(len(p.mesh.triangles) for p in scene.parts)
    placed_triangles = sum(len(p.mesh.triangles) * counts[p.key] for p in scene.parts)
    report = {
        "source": scene.source,
        "output": result.path.name,
        "units": options.units,
        "up_axis": options.up_axis,
        "linear_deflection_mm": linear_deflection,
        "counts": {
            "assemblies": sum(node.is_assembly for node in nodes),
            "part_occurrences": sum(counts.values()),
            "unique_parts": len(scene.parts),
            "prototypes": result.prototypes,
            "instances": result.instances,
            "triangles_stored": unique_triangles,
            "triangles_placed": placed_triangles,
        },
        "volume_cm3": total_volume_mm3 / 1000.0,
        "parts": parts,
        # CAD names that are not valid prim names; the original is kept as displayName.
        "renamed": [{"cad_name": cad, "prim_name": prim} for cad, prim in result.renamed],
        # Curved surfaces sit up to one deflection inside their triangles' chords.
        "bounds_check": verify_bounds(scene, result, options, tolerance_mm=2 * linear_deflection + 1e-3),
        "warnings": warnings,
    }
    if options.density is not None:
        report["density_kg_m3"] = options.density
        report["mass_kg"] = options.density * total_volume_mm3 * MM_TO_M**3
    return report
