from enum import Enum
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.model_components.losses import EPS, ds_nerf_depth_loss, urban_radiance_field_depth_loss


class LidarDepthLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = 1
    URF = 2
    DS_NERF_NEW = 3


def ds_nerf_depth_loss_new(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    lengths: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
    valid_depth_mask: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
    sky_mask: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
    # depth_scale_factor: Optional[Float[Tensor, "0"]] = None,
) -> Float[Tensor, "*batch 1"]:
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).
    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    # sky_depth = 250  # meters
    # scaled_sky_depth = sky_depth * depth_scale_factor
    # sigma = sigma * depth_scale_factor
    # depth_mask = (termination_depth > 0) & (termination_depth < scaled_sky_depth)
    # sky_mask = termination_depth >= scaled_sky_depth  # ignore large depth i.e. sky

    ds_loss = (
        -torch.log(weights + EPS) * torch.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    )
    sky_loss = weights**2
    masked_sum_ds_loss = ds_loss.sum(-2) * valid_depth_mask
    masked_sum_sky_loss = sky_loss.sum(-2) * sky_mask
    loss = masked_sum_ds_loss + masked_sum_sky_loss
    return torch.mean(loss)


def lidar_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    ray_samples: RaySamples,
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    sigma: Float[Tensor, "0"],
    directions_norm: Float[Tensor, "*batch 1"],
    is_euclidean: bool,
    depth_loss_type: LidarDepthLossType,
    valid_depth_mask: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
    sky_mask: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
    # depth_scale_factor: Optional[Float[Tensor, "0"]] = None,
) -> Float[Tensor, "0"]:
    """Implementation of depth losses.

    Args:
        weights: Weights predicted for each sample.
        ray_samples: Samples along rays corresponding to weights.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        sigma: Uncertainty around depth value.
        directions_norm: Norms of ray direction vectors in the camera frame.
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        depth_loss_type: Type of depth loss to apply.

    Returns:
        Depth loss scalar.
    """
    if not is_euclidean:
        termination_depth = termination_depth * directions_norm
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    if depth_loss_type == LidarDepthLossType.DS_NERF:
        lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        return ds_nerf_depth_loss(weights, termination_depth, steps, lengths, sigma)
    elif depth_loss_type == LidarDepthLossType.DS_NERF_NEW:
        lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        return ds_nerf_depth_loss_new(weights, termination_depth, steps, lengths, sigma, valid_depth_mask, sky_mask)
    if depth_loss_type == LidarDepthLossType.URF:
        return urban_radiance_field_depth_loss(weights, termination_depth, predicted_depth, steps, sigma)

    raise NotImplementedError("Provided depth loss type not implemented.")


def monosdf_normal_loss(
    normal_pred: Float[Tensor, "num_samples 3"],
    normal_gt: Float[Tensor, "num_samples 3"],
    valid_gt_mask: Optional[Float[Tensor, "num_samples"]] = None,
) -> Float[Tensor, "0"]:
    """
    Normal consistency loss proposed in monosdf - https://niujinshuchong.github.io/monosdf/
    Enforces consistency between the volume rendered normal and the predicted monocular normal.
    With both angluar and L1 loss. Eq 14 https://arxiv.org/pdf/2206.00665.pdf
    Args:
        normal_pred: volume rendered normal
        normal_gt: monocular normal
    """
    # Note that the default normal mask can be prone to rounding errors: Some 0 normal vectors ended up with a non-zero value
    # depth mask is more reliable
    valid_gt_mask = (normal_gt != 0).all(dim=-1) if valid_gt_mask is None else valid_gt_mask
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)[valid_gt_mask]
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)[valid_gt_mask]
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos
