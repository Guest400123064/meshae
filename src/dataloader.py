from pathlib import Path
from typing import Any

import torch
from ezgatr.interfaces import point
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from PIL import Image


class Pix3DObject:
    r"""Helper class for creating point cloud samples.

    Parameters
    ----------
    root: pathlib.Path
        Path to the root folder of the raw Pix3D dataset.
    meta: dict[str, Any]
        Metadata from ``pix3d.json``.
    load_on_init: bool
        Whether load the object with default settings on initialization.
        Please refer to ``Pix3DObject.load_obj()`` method for the default
        settings.
    """

    def __init__(
        self, root: Path, meta: dict[str, Any], load_on_init: bool = True,
    ) -> None:
        self.root = root
        self.meta = meta
        self.mesh = self.load_obj()
        self.image = self.load_img()

    def sample_points(
        self, num_samples: int = 1024, to_pga: bool = True,
    ) -> torch.Tensor:
        r"""Sample a point cloud with size ``num_samples``.

        The sampled point cloud is returned in a batch with a single record.
        If ``to_pga`` is set to ``True``, a channel dimension with size 1 is
        automatically ``unsqueeze``-ed to the second to the last dimension.

        Parameters
        ----------
        num_samples: int
            Number of points in the returned point cloud.
        to_pga: bool
            If returning point cloud encoded as PGA multi-vectors.

        Returns
        -------
        torch.Tensor
            Sampled point cloud in 3D or PGA multi-vectors. Returning tensor
            has shape ``(B, N, 3)`` or ``(B, N, 1, 16)``.
        """
        if not self.is_initialized:
            raise RuntimeError("Call ``Pix3DObject.load_obj`` to load mesh first.")

        ret = sample_points_from_meshes(self.mesh, num_samples)
        if to_pga:
            return point.encode(ret).unsqueeze(-2)
        return ret

    def load_img(self,):
        return Image.open(self.root / self.meta["img"])

    def load_obj(self, normalize: bool = True):
        r"""Load the object into mesh and assign to ``self.mesh``.

        Load a ``pytorch3d.structures.Meshes`` object from a given
        path. The loaded .

        Parameters
        ----------
        normalize: bool
            Loaded mesh is centered around the mean and scaled to
            ``(-1, 1)`` if set to ``True``.

        Returns
        -------
        Pix3DObject
            Returning a reference to self.
        """
        verts, faces, _ = load_obj(self.root / self.meta["model"])
        if normalize:
            center = verts.mean(dim=0)
            scaler = max(verts.abs().max(dim=0).values)
            verts = (verts - center) / scaler

        return Meshes(verts=[verts], faces=[faces.verts_idx])
