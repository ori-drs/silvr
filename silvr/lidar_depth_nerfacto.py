from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Type

import open3d as o3d
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils import colormaps
from silvr.loss import LidarDepthLossType, lidar_depth_loss
from silvr.utils import apply_depth_colormap

silvr_path = Path(__file__).parent.parent


@dataclass
class LidarDepthNerfactoModelConfig(DepthNerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LidarDepthNerfactoModel)
    lidar_depth_loss_type: LidarDepthLossType = LidarDepthLossType.DS_NERF_NEW
    save_pcd: bool = False
    save_pcd_folder: str = "pcd"
    depth_encoding: float = 1 / 256.0  # depth value x depth_encoding = depth in meters
    dataparser_scale: float = None
    sky_depth_meter: float = 250  # hardcoded sky depth in meters

    @property
    def sky_depth_nerf(self):
        """sky depth in nerf scale after dataparser"""
        assert self.dataparser_scale is not None
        return self.sky_depth_meter * self.dataparser_scale

    def __post_init__(self):
        (silvr_path / self.save_pcd_folder).mkdir(exist_ok=True)


class LidarDepthNerfactoModel(DepthNerfactoModel):
    """nerfacto + lidar depth"""

    config: LidarDepthNerfactoModelConfig
    save_count = 0

    def populate_modules(self):
        super().populate_modules()
        self.renderer_depth = DepthRenderer(method="expected")

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # not compute normal by default
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }
        outputs["depth_metric_uint16"] = depth / self.config.dataparser_scale / self.config.depth_encoding
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY]
        outputs["ray_origins"] = ray_bundle.origins
        outputs["ray_directions"] = ray_bundle.directions

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = NerfactoModel.get_metrics_dict(self, outputs, batch)
        # metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            sigma *= self.config.dataparser_scale  # scale uncertaitny to dataparser scale
            termination_depth = batch["depth_image"].to(self.device)
            for i in range(len(outputs["weights_list"])):
                valid_depth_mask = (termination_depth > 0) & (termination_depth < self.config.sky_depth_nerf)
                sky_mask = termination_depth >= self.config.sky_depth_nerf
                metrics_dict["depth_loss"] += lidar_depth_loss(
                    weights=outputs["weights_list"][i],
                    ray_samples=outputs["ray_samples_list"][i],
                    termination_depth=termination_depth,
                    predicted_depth=outputs["depth"],
                    sigma=sigma,
                    directions_norm=outputs["directions_norm"],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.lidar_depth_loss_type,
                    valid_depth_mask=valid_depth_mask,
                    sky_mask=sky_mask,
                    # depth_scale_factor=self.config.dataparser_scale,
                ) / len(outputs["weights_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # loss_dict["rgb_loss"] *= 0
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, batch)

        # visualise depth error
        ground_truth_depth = batch["depth_image"]
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]
        max_depth = 70.0 * self.config.dataparser_scale
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=float(torch.min(ground_truth_depth).cpu()),
            far_plane=float(max_depth),
        )
        images["depth"] = predicted_depth_colormap

        error = outputs["depth"] - ground_truth_depth
        percentange_error = error / ground_truth_depth
        error_colormap = apply_depth_colormap(
            error, accumulation=outputs["accumulation"], near_plane=-0.1, far_plane=0.1, cmap="bwr"
        )
        percentange_error_colormap = apply_depth_colormap(
            percentange_error,
            accumulation=outputs["accumulation"],
            near_plane=-0.2,
            far_plane=0.2,
            cmap="bwr",
        )
        depth_mask = ground_truth_depth[..., 0] > 0
        error_colormap[~depth_mask] = torch.DoubleTensor([0, 0, 0]).to(ground_truth_depth.device)
        percentange_error_colormap[~depth_mask] = torch.DoubleTensor([0, 0, 0]).to(ground_truth_depth.device)
        images["depth_error"] = error_colormap
        # images['depth_percentage_error'] = percentange_error_colormap

        if self.config.save_pcd:
            accum_mask = (outputs["accumulation"] > 0.98).squeeze(-1)
            self.save_depth_as_pcd(
                f"{self.save_count}_nerf",
                outputs["depth"],
                outputs["ray_origins"],
                outputs["ray_directions"],
                outputs["rgb"],
                accum_mask,
            )
            sky_mask = (ground_truth_depth < self.config.sky_depth_meter * self.config.dataparser_scale).squeeze(-1)
            self.save_depth_as_pcd(
                f"{self.save_count}_gt",
                ground_truth_depth,
                outputs["ray_origins"],
                outputs["ray_directions"],
                outputs["rgb"],
                sky_mask,
            )
            self.save_count += 1

        return metrics, images

    def save_depth_as_pcd(self, save_name, depth, ray_origins, ray_directions, rgb, mask):
        pcd = o3d.geometry.PointCloud()
        points = ray_origins + ray_directions * depth

        points = points[mask]
        rgb = rgb[mask]

        pcd.points = o3d.utility.Vector3dVector(points.float().reshape(-1, 3).cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.float().reshape(-1, 3).cpu().numpy())
        o3d.io.write_point_cloud(f"{silvr_path}/{self.config.save_pcd_folder}/{save_name}.pcd", pcd)
