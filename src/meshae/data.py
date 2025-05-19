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
    compute_normalized_mesh,
    compute_sorted_faces,
)

if TYPE_CHECKING:
    from torchtyping import TensorType

    from meshae.typing import MeshAEDatumKeyType


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

        Padding values are configured through the dataset initialization parameters. For
        faces and face edges, boolean masks will be created to help distinguish actual
        inputs from padding values. No mask generated for vertices since there's no mixing
        operation applied to vertices.

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


class MeshAEDataset(Dataset):
    r""" """

    def __init__(
        self,
        path: Path,
        *,
        sort_face_by: str = "zxy",
        include_self: bool = False,
    ) -> None:
        super().__init__()

        self.path = Path(path)
        self.objects = list(self.path.glob("*.glb"))
        if len(self.objects) == 0:
            msg = f"No '.glb' object found under directory <{str(path)}>."
            raise RuntimeError(msg)

        self.sort_face_by = sort_face_by
        self.include_self = include_self

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> dict:
        return self.load_and_process(self.objects[idx])

    def load_and_process(self, path: Path) -> dict[MeshAEDatumKeyType, TensorType]:
        r"""Load and normalize a single mesh object from given path.

        Parameters
        ----------
        path : pathlib.Path
            Path to the source mesh object.

        Returns
        -------
        faces : TensorType["n_face", 3, int]
            Face tensor.
        edges : TensorType["n_edge", 2, int]
            Face edge tensor.
        vertices : TensorType["n_vrtx", 3, 3, float]
            Vertex tensor.
        """
        mesh = trimesh.load(path, file_type="glb", force="mesh", process=False)
        mesh, _, _ = compute_normalized_mesh(mesh)

        vertices = torch.from_numpy(mesh.vertices)
        faces = compute_sorted_faces(mesh, by=self.sort_face_by, return_tensor=True)
        edges = compute_face_edges(faces, include_self=self.include_self)

        return {"faces": faces, "edges": edges, "vertices": vertices}
