from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from trimesh import Trimesh

if TYPE_CHECKING:
    from numpy.typing import NDArray


def normalize_mesh(
    mesh: Trimesh, scale: float = 0.95,
) -> tuple[Trimesh, NDArray[np.float32], float]:
    r"""Normalize the mesh object into a bounding box spanning from -0.5 to 0.5.

    The scale is aligned with the experiment design mentioned in Section 5.1 of
    MeshAnything V1 paper [1]_. This function creates a new mesh object.

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
        
    tuple[Trimesh, NDArray[np.float32], float]
        ``(normalized_mesh, center, net_scale)`` in that order. ``center``
        and ``net_scale`` is used to restore the transformation. 

    References
    ----------
    .. [1] `"MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers", Chen et al., 2022
            <https://github.com/wang-ps/mesh2sdf/tree/master>`_
    """
    mesh = mesh.copy()
    bbmin, bbmax = mesh.bounds

    ctr = -0.5 * (bbmax + bbmin)
    scl = scale / (bbmax - bbmin).max()

    mesh.apply_translation(ctr)
    mesh.apply_scale(scl)

    return mesh, ctr, scl


def normalize_vertices(
    vertices: NDArray[np.float32], scale: float = 0.95,
) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
    r"""Normalize the mesh vertices into a bounding box spanning from -1 to 1.

    Parameters
    ----------
    vertices : NDArray[np.float32]
        Mesh vertices of type ``trimesh.caching.TrackedArray``, shape ``(n, 3)``.
    scale : float, default to 0.95
        Minor scaling applied to the centered and bounded vertices to not
        collide with the bounding box; should be a value from 0 to 1.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32], float]
        ``(normalized_vertices, center, net_scale)`` in that order. ``center``
        and ``net_scale`` is used to restore the transformation.
    """
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)

    c = 0.5 * (bbmax + bbmin)
    s = 2.0 / (bbmax - bbmin).max() * scale

    return (vertices - c) * s, c, s


def argsort_face_vertices(vertices: NDArray[np.float32], by: str = "zxy") -> NDArray[np.int8]:
    r"""Sort face vertices such that the first vertices has lowest coordinate by ``by``.

    This operation preserves the mesh face orientation, i.e., the clockwise/counter-clockwise
    order. It returns a batch of reordering indices of shape ``(B, 3)``, where ``B`` denotes
    the number of faces in the ``vertices`` argument.

    Parameters
    ----------
    vertices : NDArray[np.float32]
        Batch of face vertices with expected shape ``(B, 3, 3)`` where ``B`` denotes
        the number of faces.
    by : str, default to "zxy"
        A string literal denoting the lexical sort priority. Default to sorting by the
        "z" axis, followed by "x" and "y".

    Returns
    -------
    NDArray[np.int8]
        Batch of face argsort indices with shape ``(B, 3)``. One may want to leverage
        ``np.take_along_axis(faces, argsort_idx)`` to rearrange vertices in each face.
    """
    key = {
        "x": vertices[..., 0], "y": vertices[..., 1], "z": vertices[..., 2],
    }
    ret = np.zeros(vertices.shape[:2], dtype=np.int8)
    idx = np.lexsort(tuple(key[a] for a in reversed(by)))[..., 0]
    for i in range(3):
        ret[idx == i] = [j % 3 for j in range(i, i + 3)]

    return ret
