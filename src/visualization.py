import torch
import matplotlib.pyplot as plt


def plot_normalized_point_cloud(points: torch.Tensor):
    r"""Visualize normalized point cloud.

    Assuming that the points are centered to the origin and scaled
    to ``(-1, 1)``.
    """
    x, y, z = (
        points.clone()
        .detach()
        .cpu()
        .squeeze()
        .unbind(1)
    )
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(x, y, z)
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xlim=(-1, 1),
        ylim=(-1, 1),
        zlim=(-1, 1),
    )
    ax.set_box_aspect((1, 1, 1))

    return ax
