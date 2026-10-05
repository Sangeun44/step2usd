"""Command line: step2usd {convert,inspect,splat,sample-splat}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import Node, Scene
from .reader import StepReadError, read_step
from .report import build_report
from .splat import SplatWriteOptions, convert_splat, splats_from_scene, write_ply
from .writer import WriteOptions, write_usd


def convert(
    source: Path,
    output: Path,
    options: WriteOptions | None = None,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> dict:
    """Convert a STEP file to USD and return the report."""
    options = options or WriteOptions()
    options.metadata = {
        "linearDeflectionMm": linear_deflection,
        "angularDeflectionRad": angular_deflection,
        **options.metadata,
    }
    scene = read_step(source, linear_deflection, angular_deflection)
    result = write_usd(scene, output, options)
    return build_report(scene, result, options, linear_deflection)


def format_tree(scene: Scene) -> str:
    counts = scene.occurrences()
    lines: list[str] = []

    def visit(node: Node, depth: int) -> None:
        indent = "  " * depth
        if node.is_assembly:
            lines.append(f"{indent}{node.name}/")
        else:
            part = node.part
            shared = f", 1 of {counts[part.key]}" if counts[part.key] > 1 else ""
            name = node.name if node.name == part.name else f"{node.name} -> {part.name}"
            lines.append(f"{indent}{name}  [{part.face_count} faces{shared}]")
        for child in node.children:
            visit(child, depth + 1)

    visit(scene.root, 0)
    return "\n".join(lines)


def _print_summary(report: dict) -> None:
    c = report["counts"]
    print(f"{report['source']} -> {report['output']}  ({report['units']}, {report['up_axis']} up)")
    print(
        f"  {c['assemblies']} assemblies, {c['part_occurrences']} placed parts from "
        f"{c['unique_parts']} unique parts ({c['instances']} instanced)"
    )
    print(f"  {c['triangles_stored']} triangles stored, {c['triangles_placed']} once instances are placed")
    print(f"  volume {report['volume_cm3']:.2f} cm^3", end="")
    print(f", mass {report['mass_kg']:.3f} kg" if "mass_kg" in report else "")
    check = report["bounds_check"]
    if check["ok"]:
        print(f"  bounds match the CAD model within {check['max_deviation']:.2e} {check['units']}")
    else:
        detail = check.get("reason") or (
            f"off by {check['max_deviation']:.3g} {check['units']} (tolerance {check['tolerance']:.3g})"
        )
        print(f"  BOUNDS MISMATCH: {detail}")
    if report["renamed"]:
        print(f"  {len(report['renamed'])} names adjusted to valid prim names (originals kept as displayName)")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")


def _cmd_convert(args) -> int:
    options = WriteOptions(
        units=args.units,
        up_axis=args.up_axis,
        instancing=not args.no_instancing,
        density=args.density,
        face_ids=args.face_ids,
    )
    output = args.output or args.input.with_suffix(".usda")
    report = convert(args.input, output, options, args.linear_deflection, args.angular_deflection)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)
    return 0 if report["bounds_check"]["ok"] else 1


def _cmd_inspect(args) -> int:
    # Coarse tessellation: inspect only needs the structure.
    print(format_tree(read_step(args.input, linear_deflection=1.0)))
    return 0


def _cmd_splat(args) -> int:
    options = SplatWriteOptions(
        up_axis=args.up_axis, meters_per_unit=args.meters_per_unit, rotate_x=args.rotate_x
    )
    output = args.output or args.input.with_suffix(".usdc")
    report = convert_splat(args.input, output, options, args.min_opacity)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    dropped = report["dropped"]
    print(f"{report['source']} -> {report['output']}  ({report['up_axis']} up)")
    print(f"  {report['gaussians_written']} Gaussians, spherical-harmonics degree {report['sh_degree']}")
    if dropped["non_finite"] or dropped["below_min_opacity"]:
        print(
            f"  dropped {dropped['non_finite']} with non-finite values, "
            f"{dropped['below_min_opacity']} below the opacity threshold"
        )
    lo, hi = report["bounds"]
    size = " x ".join(f"{b - a:.3g}" for a, b in zip(lo, hi))
    print(f"  bounds {size} units, {report['opacity']['nearly_transparent']} nearly transparent")
    ok = report["round_trip"]["ok"]
    print("  read back from USD: identical" if ok else "  READ-BACK MISMATCH: see the report")
    return 0 if ok else 1


def _cmd_sample_splat(args) -> int:
    scene = read_step(args.input, args.linear_deflection)
    output = args.output or args.input.with_suffix(".ply")
    write_ply(splats_from_scene(scene, args.count), output)
    print(f"{scene.source} -> {output.name}  ({args.count} Gaussians sampled from the CAD surfaces, metres)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="step2usd", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert a STEP file to USD")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, help=".usda, .usdc or .usd (default: next to the input)")
    p.add_argument("--units", choices=("m", "mm"), default="m", help="stage units (default: m)")
    p.add_argument("--up-axis", choices=("Z", "Y"), default="Z", help="stage up axis (default: Z)")
    p.add_argument("--linear-deflection", type=float, default=0.1, metavar="MM",
                   help="max gap between a surface and its triangles, in mm (default: 0.1)")
    p.add_argument("--angular-deflection", type=float, default=0.5, metavar="RAD",
                   help="max angle between neighbouring triangles, in radians (default: 0.5)")
    p.add_argument("--density", type=float, metavar="KG_M3",
                   help="author colliders and exact masses using this density, e.g. 7850 for steel")
    p.add_argument("--no-instancing", action="store_true", help="write every placement as its own mesh")
    p.add_argument("--face-ids", action="store_true", help="keep each triangle's CAD face index as a primvar")
    p.add_argument("--report", type=Path, help="write the JSON report here")
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("inspect", help="print the assembly tree of a STEP file")
    p.add_argument("input", type=Path)
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("splat", help="convert a 3D Gaussian Splatting .ply to a USD ParticleField")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, help=".usdc, .usda or .usd (default: .usdc next to the input)")
    p.add_argument("--up-axis", choices=("Z", "Y"), default="Z", help="stage up axis (default: Z)")
    p.add_argument("--meters-per-unit", type=float, default=1.0,
                   help="size of one capture unit in metres, if known (default: 1.0)")
    p.add_argument("--rotate-x", type=float, default=0.0, metavar="DEG",
                   help="turn the capture about X on the root prim, e.g. 180 for a COLMAP-oriented scene")
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="drop Gaussians fainter than this, e.g. 0.02 (default: keep all)")
    p.add_argument("--report", type=Path, help="write the JSON report here")
    p.set_defaults(func=_cmd_splat)

    p = sub.add_parser("sample-splat", help="sample a STEP model's surfaces into a synthetic splat .ply")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, help=".ply (default: next to the input)")
    p.add_argument("--count", type=int, default=12000, help="number of Gaussians (default: 12000)")
    p.add_argument("--linear-deflection", type=float, default=0.1, metavar="MM")
    p.set_defaults(func=_cmd_sample_splat)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (StepReadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
