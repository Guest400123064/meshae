from __future__ import annotations

from typing import Literal

b = None
n_edge = None
n_face = None
n_vrtx = None

MeshAEFeatNameType = Literal["area", "norm", "acos", "vrtx"]
MeshAEDatumKeyType = Literal["vertices", "faces", "edges", "face_masks", "edge_masks"]
