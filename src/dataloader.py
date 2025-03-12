from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes


def load_normalized_mesh_from_obj(path, normalize=True) -> Meshes:
    r"""Load and normalize a mesh.

    Load a ``pytorch3d.structures.Meshes`` object from a given
    path. The loaded mesh is centered around the mean and scaled
    to ``(-1, 1)``.
    """
    v, f, _ = load_obj(path)
    if normalize:
        c = v.mean(dim=0)
        s = max(v.abs().max(dim=0).values)
        v = (v - c) / s

    return Meshes(verts=[v], faces=[f.verts_idx])
