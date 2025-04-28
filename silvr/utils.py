import json
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor

from nerfstudio.utils import colormaps


def apply_depth_colormap(
    depth: Float[Tensor, "*bs 1"],
    accumulation: Optional[Float[Tensor, "*bs 1"]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    cmap="turbo",
) -> Float[Tensor, "*bs rgb=3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        cmap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """

    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    if cmap == "bwr":
        # import pdb
        # pdb.set_trace()
        colored_image = 255 * cm.get_cmap(cmap)(depth[..., 0].cpu().numpy())[..., :3]

        colored_image = torch.from_numpy(colored_image).to(depth.device)

    else:
        colored_image = colormaps.apply_colormap(depth, cmap=cmap)

        if accumulation is not None:
            colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def load_dataparser_transform(path):
    """Load the transform and scale from the dataparser_transforms.json file.
    p_nerf = T_nerf_metric @ p_metric = scale @ transform @ p_metric
    """
    with open(path, "r") as f:
        data_transforms = json.load(f)
        transform = data_transforms["transform"]
        transform = np.array(transform)
        transform = np.vstack([transform, np.array([0, 0, 0, 1])])
        scale = data_transforms["scale"]
    return transform, scale


def load_transformation_matrix(path):
    transform, scale = load_dataparser_transform(path)
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, :3] *= scale
    T_nerf_metric = scaling_matrix @ transform
    return T_nerf_metric
