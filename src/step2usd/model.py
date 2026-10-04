"""The neutral scene model that sits between the CAD reader and the USD writer.

Nothing here imports OpenCascade or USD. Lengths are millimetres, the unit the
STEP reader normalises every file to; the writer rescales on the way out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Color = tuple[float, float, float]  # linear RGB, 0..1


@dataclass
class MeshData:
    points: np.ndarray  # (N, 3) float64, mm
    normals: np.ndarray  # (N, 3) float64, unit length, one per point
    triangles: np.ndarray  # (M, 3) int32 indices into points
    face_ids: np.ndarray  # (M,) int32, index of the B-rep face each triangle came from

    @property
    def is_empty(self) -> bool:
        return len(self.triangles) == 0


@dataclass
class Part:
    """One unique piece of geometry. May be placed many times in the tree."""

    key: str  # stable id from the CAD document
    name: str
    mesh: MeshData
    color: Color | None = None
    face_colors: dict[int, Color] = field(default_factory=dict)  # B-rep face index -> colour
    face_count: int = 0
    failed_faces: int = 0  # faces the tessellator could not mesh
    is_solid: bool = False
    is_valid: bool = True  # B-rep passed the kernel's validity check
    volume_mm3: float | None = None  # exact, from the B-rep; None when not a solid
    center_of_mass_mm: tuple[float, float, float] | None = None


@dataclass
class Node:
    """One occurrence in the product structure: an assembly or a placed part."""

    name: str
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # local, column-vector, mm
    part: Part | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_assembly(self) -> bool:
        return self.part is None

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Scene:
    source: str  # file name the scene was read from
    root: Node
    parts: list[Part]  # unique parts, in first-seen order
    # Exact bounding box of the whole model from the B-rep, mm: (min xyz, max xyz).
    bounds_mm: tuple[np.ndarray, np.ndarray] | None = None

    def occurrences(self) -> dict[str, int]:
        """How many times each part is placed, keyed by Part.key."""
        counts: dict[str, int] = {part.key: 0 for part in self.parts}
        for node in self.root.walk():
            if node.part is not None:
                counts[node.part.key] += 1
        return counts
