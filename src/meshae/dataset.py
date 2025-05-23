from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import trimesh
from beartype import beartype
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from meshae.utils import (
    compute_face_edges,
    normalize_mesh,
    sort_faces,
)

if TYPE_CHECKING:
    from torchtyping import TensorType

    from meshae.typing import MeshAEDatumKeyType


class MeshAEDataset(Dataset):
    r"""Mesh auto-encoder training dataset.

    Each object is loaded from disk on-demand. Loaded mesh will be normalized to center around the
    origin and bounded by a bounding box with the long diagonal normalized to 1. Meshes in formats
    other than ``.glb`` are ignored.

    Parameters
    ----------
    path : str
        String path to the data folder. The folder is assumed to be flattened, i.e., all ``.glb``
        objects should residue in at the same depth.
    sort_by : str, default="zyx"
        The order of axis used to sort mesh faces. ``"zyx"`` indicates that the first vertex of a
        processed face always have the lowest vertical coordinate. Then, faces from the same mesh
        are sorted by the coordinates of the first vertices from all faces.
    """

    def __init__(self, path: str, sort_by: str = "zyx") -> None:
        super().__init__()

        self.path = Path(path)
        self.sort_by = sort_by
        self.objects = list(self.path.glob("*.glb"))
        if len(self.objects) == 0:
            msg = f"No '.glb' object found under directory <{path}>."
            raise RuntimeError(msg)

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> dict[MeshAEDatumKeyType, TensorType]:
        mesh, _, _ = normalize_mesh(
            trimesh.load(self.objects[idx], file_type="glb", force="mesh", process=False)
        )
        faces = sort_faces(mesh, by=self.sort_by, return_tensor=True)
        datum = {
            "faces": faces,
            "edges": compute_face_edges(faces),
            "vertices": torch.from_numpy(mesh.vertices),
        }
        return datum


class MeshAECollateFn:
    r"""A configurable collate function.

    Parameters
    ----------
    vertex_padding_value : float, default=0.0
        Padding value for vertex coordinate values.
    face_padding_value : int, default=0
        Padding vertex index for batch of faces. Setting to any valid index that does not
        exceed the maximum number of vertices should be sufficient. Retrieved vertices will
        be masked out.
    edge_padding_value : int, default=0
        Padding face index for batch of face edges. Setting to any valid index that does
        not exceed the maximum number of faces should be sufficient. Retrieved faces will
        be masked out.
    """

    @beartype
    def __init__(
        self,
        vertex_padding_value: float = 0.0,
        face_padding_value: int = 0,
        edge_padding_value: int = 0,
    ) -> None:
        self.vertex_padding_value = vertex_padding_value
        self.face_padding_value = face_padding_value
        self.edge_padding_value = edge_padding_value

    def __call__(
        self,
        data: list[dict[MeshAEDatumKeyType, TensorType]],
    ) -> dict[MeshAEDatumKeyType, TensorType]:
        r"""Custom collate function for vertices, faces, and face edges.

        Padding values are configured through the initialization parameters. For faces and
        face edges, boolean masks will be created to help distinguish actual inputs from
        padding values. No mask generated for vertices since there's no mixing operation
        applied to vertices.

        Parameters
        ----------
        data : list[dict[MeshAEDatumKeyType, TensorType]]
            A batch of datum dictionaries.

        Returns
        -------
        batch : dict[MeshAEDatumKeyType, TensorType]
            Batch of inputs collated into a single dictionary.
        """

        def _collate(key, padding_value, create_mask=False):
            batch, s_len = [], []
            for datum in data:
                batch.append(datum[key])
                s_len.append(datum[key].size(0))

            batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
            if create_mask:
                array = torch.arange(max(s_len), device=batch.device).unsqueeze(0)
                masks = torch.cat([array < len_ for len_ in s_len], dim=0)
                return batch, masks

            return batch

        faces, face_masks = _collate(
            "faces",
            padding_value=self.face_padding_value,
            create_mask=True,
        )
        edges, edge_masks = _collate(
            "edges",
            padding_value=self.edge_padding_value,
            create_mask=True,
        )
        batch = {
            "vertices": _collate("vertices", padding_value=self.vertex_padding_value),
            "faces": faces,
            "edges": edges,
            "face_masks": face_masks,
            "edge_masks": edge_masks,
        }
        return batch
