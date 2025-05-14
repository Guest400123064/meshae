from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import trimesh
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from meshae.utils import (
    compute_face_edges,
    compute_normalized_mesh,
    compute_sorted_faces,
)

if TYPE_CHECKING:
    from torchtyping import TensorType

MeshAEDatumKeyType = Literal["vertex", "face", "edge", "face_masks", "edge_masks"]


class MeshAEDataset(Dataset):
    r""" """

    def __init__(
        self,
        path: Path,
        *,
        sort_face_by: str = "zxy",
        neighbor_if_share_one_vertex: bool = False,
        include_self: bool = False,
        vertex_padding_value: float = 0.0,
        face_padding_value: int = -1,
        edge_padding_value: int = -1,
    ) -> None:
        super().__init__()

        self.path = Path(path)
        self.objects = list(self.path.glob("*.glb"))
        if len(self.objects) == 0:
            msg = f"No '.glb' object found under directory <{str(path)}>."
            raise RuntimeError(msg)

        self.sort_face_by = sort_face_by
        self.neighbor_if_share_one_vertex = neighbor_if_share_one_vertex
        self.include_self = include_self

        self.vertex_padding_value = vertex_padding_value
        self.face_padding_value = face_padding_value
        self.edge_padding_value = edge_padding_value

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
        edges = compute_face_edges(
            faces,
            neighbor_if_share_one_vertex=self.neighbor_if_share_one_vertex,
            include_self=self.include_self,
        )
        return {"faces": faces, "edges": edges, "vertices": vertices}

    def collate_fn(
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
