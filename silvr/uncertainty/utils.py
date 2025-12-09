"""
Code adapted from BayesRays https://github.com/BayesRays/BayesRays/tree/main
"""

import torch

from nerfstudio.data.scene_box import SceneBox


def normalize_point_coords(points, aabb, distortion):
    """coordinate normalization process according to density_feild.py in nerfstudio"""
    if distortion is not None:
        pos = distortion(points)
        pos = (pos + 2.0) / 4.0
    else:
        pos = SceneBox.get_normalized_positions(points, aabb)
    selector = ((pos > 0.0) & (pos < 1.0)).all(dim=-1)  # points outside aabb are filtered out
    pos = pos * selector[..., None]
    return pos, selector


def find_grid_indices(points, aabb, distortion, lod, device, zero_out=True, new_impl=False):
    pos, selector = normalize_point_coords(points, aabb, distortion)  # ? distortion change gradients?
    pos, selector = pos.view(-1, 3), selector[..., None].view(-1, 1)
    uncertainty_lod = 2**lod
    coords = (pos * uncertainty_lod).unsqueeze(0)
    if new_impl:  # ? TODO: Is this this better and correct?
        uncertainty_lod = 2**lod + 1
    inds = torch.zeros((8, pos.shape[0]), dtype=torch.int32, device=device)
    coefs = torch.zeros((8, pos.shape[0]), device=device)
    corners = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [2, 0, 1, 0],
            [3, 0, 1, 1],
            [4, 1, 0, 0],
            [5, 1, 0, 1],
            [6, 1, 1, 0],
            [7, 1, 1, 1],
        ],
        device=device,
    )
    corners = corners.unsqueeze(1)
    inds[corners[:, :, 0].squeeze(1)] = (
        (torch.floor(coords[..., 0]) + corners[:, :, 1]) * uncertainty_lod * uncertainty_lod
        + (torch.floor(coords[..., 1]) + corners[:, :, 2]) * uncertainty_lod
        + (torch.floor(coords[..., 2]) + corners[:, :, 3])
    ).int()
    coefs[corners[:, :, 0].squeeze(1)] = (
        torch.abs(coords[..., 0] - (torch.floor(coords[..., 0]) + (1 - corners[:, :, 1])))
        * torch.abs(coords[..., 1] - (torch.floor(coords[..., 1]) + (1 - corners[:, :, 2])))
        * torch.abs(coords[..., 2] - (torch.floor(coords[..., 2]) + (1 - corners[:, :, 3])))
    )  # trilinear interpolation
    if zero_out:
        coefs[corners[:, :, 0].squeeze(1)] *= selector[..., 0].unsqueeze(
            0
        )  # zero out the contribution of points outside aabb box

    return inds, coefs
