from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange, repeat

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torchtyping import TensorType

b = None
n_edge = None
n_face = None
n_vrtx = None


def quantize(
    tensor: TensorType[..., float],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., int]:
    r"""Discretize continuous inputs into bin IDs."""

    high, low = high_low
    if high <= low:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    tensor = ((tensor - low) / (high - low)) * num_bins - 0.5
    return tensor.round().long().clamp(min=0, max=(num_bins - 1))


def dequantize(
    tensor: TensorType[..., int],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., float]:
    r"""Reconstruct continuous inputs from discrete bin IDs."""

    high, low = high_low
    if high <= low:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    tensor = (tensor.float() + 0.5) / num_bins
    return tensor * (high - low) + low


def gaussian_blur1d(
    tensor: TensorType[..., float],
    *,
    sigma: float = 1.0,
) -> TensorType[..., float]:
    r"""Apply 1D Gaussian blur to the last dimension of input tensor."""

    width = int(math.ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(
        -half_width,
        half_width + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    gaussian = torch.exp(-(distance**2) / (2 * sigma**2))
    gaussian = F.normalize(gaussian, 1, dim=-1)
    channels = tensor.size(-1)

    tensor = F.conv1d(
        rearrange(tensor, "... n c -> ... c n"),
        repeat(gaussian, "n -> c 1 n", c=channels),
        padding=half_width,
        groups=channels,
    )
    return rearrange(tensor, "... c n -> ... n c")


def compute_face_edges(
    faces: TensorType["n_face", 3, int],
    neighbor_if_share_one_vertex: bool = False,
    include_self: bool = False,
) -> TensorType["b", "n_edge", 2, int]:
    r"""Compute face edges from a set of mesh faces.

    TODO: If possible, we can create a pure NumPy version of this function.

    Parameters
    ----------
    faces : TensorType["n_face", 3, int]
        Mesh face sequences with each face represented by three vertex ids.
    neighbor_if_share_one_vertex : bool, default=False
        Set to ``True`` to consider faces only sharing a single vertex as neighbors.
    include_self : bool, default=False
        Whether include self as one of the neighbors.

    Returns
    -------
    edges : TensorType["n_edge", 2, int]
        Edges capturing the topological relationship between mesh faces.
    """
    T, device = faces.size(0), faces.device

    threshold = 2 - neighbor_if_share_one_vertex
    all_edges = torch.stack(
        torch.meshgrid(
            torch.arange(T, device=device),
            torch.arange(T, device=device),
            indexing="ij",
        ),
        dim=-1,
    )

    vrtx_shared = rearrange(faces, "t v -> t 1 v 1") == rearrange(
        faces, "t v -> 1 t 1 v"
    )
    vrtx_shared = vrtx_shared.any(-1).sum(-1)
    is_neighbor = vrtx_shared >= threshold

    if not include_self:
        is_neighbor &= vrtx_shared != 3

    return all_edges[is_neighbor]


def compute_sorted_faces(
    mesh: trimesh.Trimesh,
    by: str = "zxy",
    return_tensor: bool = True,
) -> NDArray[np.int64] | TensorType["n_face", 3, int]:
    r"""Sort faces and face vertices according to the ``by`` argument.

    This transformation to faces is performed in two stages. In the first stage, the three vertices
    within each individual face are rotated such that the vertex with the first ``by`` coordinate is
    shifted to the first position. Afterwards, the entire set of faces are sorted according to the
    coordinate of the first vertex.

    Parameters
    ----------
    mesh : NDArray[np.float32]
        The mesh object to be normalized.
    by : str, default="zxy"
        A string literal denoting the lexical sort priority. Default to sorting by the
        "z" axis, followed by "x" and "y".
    return_tensor : bool, default=True
        Set to ``True`` to return tensor object of ``TensorType["n_face", 3, int]``

    Returns
    -------
    sorted_faces : NDArray[np.int64] | TensorType["n_face", 3, int]
        Sorted tri-mesh faces.
    """

    def _in_face_sort() -> NDArray[np.int64]:
        r"""Rotate vertex indices within each face based on vertex coordinate.

        For each face containing three vertices, the vertex index with lowest ``by`` coordinate
        is identified and rotate to the first position. This operation preserves the mesh face
        orientation, i.e., the clockwise/counter-clockwise order.
        """
        vrtx = mesh.vertices[mesh.faces]
        keys = {"x": vrtx[..., 0], "y": vrtx[..., 1], "z": vrtx[..., 2]}
        keys = tuple(keys[ax] for ax in reversed(by))

        indices = np.lexsort(keys)[..., 0]
        argsort = np.empty(mesh.faces.shape, dtype=mesh.faces.dtype)
        for i in range(3):
            argsort[indices == i] = [j % 3 for j in range(i, i + 3)]

        return np.take_along_axis(mesh.faces, argsort, axis=1)

    def _xx_face_sort(faces: NDArray[np.int64]) -> NDArray[np.int64]:
        r"""Sort the entire set of faces based on the coordinate of the first vertices."""

        vrtx = mesh.vertices[faces[:, 0]]
        keys = {"x": vrtx[..., 0], "y": vrtx[..., 1], "z": vrtx[..., 2]}
        keys = tuple(keys[ax] for ax in reversed(by))

        return faces[np.lexsort(keys)]

    sorted_faces = _xx_face_sort(_in_face_sort())
    if return_tensor:
        return torch.from_numpy(sorted_faces)

    return sorted_faces


def compute_normalized_mesh(
    mesh: trimesh.Trimesh,
    scale: float = 0.95,
) -> tuple[trimesh.Trimesh, NDArray[np.float32], float]:
    r"""Normalize the mesh object into a bounding box with the long diagonal being 1.

    This design from PivotMesh [1]_ is different from the MeshAnything V1 paper [2]_ in
    that MeshAnything normalizes the mesh object by the long edge, instead of the diagonal.

    This function creates a new mesh object.

    Parameters
    ----------
    mesh : Trimesh
        The mesh object to be normalized.
    scale : float, default to 0.95
        A small scaler applied to the normalized mesh object that further shrinks
        the mesh size to avoid collision with bounding box. Should be a scaler value
        ranging from 0 to 1.

    Returns
    -------
    mesh : trimesh.Trimesh
        Normalized mesh.
    center : NDArray[np.float32]
        Mid-point of the long diagonal used to center the mesh.
    scale : float
        Final scaling factor.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    .. [2] `"MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers", Chen et al., 2022
            <https://github.com/wang-ps/mesh2sdf/tree/master>`_
    """
    mesh = mesh.copy()
    bbmin, bbmax = mesh.bounds

    center = -0.5 * (bbmax + bbmin)
    scale = scale / np.linalg.norm(bbmax - bbmin)

    mesh.apply_translation(center)
    mesh.apply_scale(scale)

    return mesh, center, scale
