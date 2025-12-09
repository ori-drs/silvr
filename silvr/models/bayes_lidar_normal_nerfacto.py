from dataclasses import dataclass, field
from typing import Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from silvr.models.bayes_nerfacto import BayesNerfactoModel, BayesNerfactoModelConfig
from silvr.models.lidar_normal_nerfacto import LidarNormalNerfactoModel, LidarNormalNerfactoModelConfig


@dataclass
class BayesLidarNormalNerfactoModelConfig(LidarNormalNerfactoModelConfig, BayesNerfactoModelConfig):
    _target: Type = field(default_factory=lambda: BayesLidarNormalNerfactoModel)


class BayesLidarNormalNerfactoModel(LidarNormalNerfactoModel, BayesNerfactoModel):
    config: BayesLidarNormalNerfactoModelConfig

    def get_outputs_with_deformation_field(self, ray_bundle, deform_field):
        with torch.autocast(self.device.type, enabled=True):
            return super().get_outputs_with_deformation_field(ray_bundle, deform_field)

    def get_outputs(self, ray_bundle: RayBundle):
        if not hasattr(self, "uncertainties"):
            outputs = super().get_outputs(ray_bundle)
            return outputs
        # If have computed hessian, render uncertainty
        else:
            with torch.autocast(self.device.type, enabled=True):
                return self.get_outputs_with_uncertainty(ray_bundle)

    def get_outputs_with_uncertainty(self, ray_bundle: RayBundle):
        density_fns_new = []
        max_un_points = (
            self.config.filter_point_thresh * self.config.max_log_uncertainty
            if self.config.use_legacy_log_unc
            else self.config.filter_point_thresh * self.config.max_uncertainty
        )
        if self.config.filter_out_point:
            for i in self.density_fns:
                density_fns_new.append(lambda x, i=i: i(x) * (self.get_un_points(x) <= max_un_points))
        else:
            density_fns_new = self.density_fns
        # resample with filtered point density
        ray_samples, _, _ = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
        field_outputs = self.field(ray_samples, compute_normals=True)
        points = ray_samples.frustums.get_positions()
        un_points = self.get_un_points(points)

        if self.config.filter_out_point:
            density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= max_un_points)
        else:
            density = field_outputs[FieldHeadNames.DENSITY]
        weights = ray_samples.get_weights(density)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        outputs = {"rgb": rgb, "depth": depth, "accumulation": accumulation}

        outputs["depth_metric_uint16"] = depth / self.config.dataparser_scale / self.config.depth_encoding
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY]
        outputs["ray_origins"] = ray_bundle.origins
        outputs["ray_directions"] = ray_bundle.directions

        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        outputs["normals"] = normals

        uncertainty = torch.sum(weights * un_points, dim=-2)
        if self.config.use_legacy_log_unc:
            uncertainty += (1 - torch.sum(weights, dim=-2)) * self.config.min_log_uncertainty  # alpha blending
        else:
            uncertainty += (1 - torch.sum(weights, dim=-2)) * self.config.max_uncertainty  # alpha blending
            uncertainty = torch.log10(uncertainty + 1e-12)

        # normalize into acceptable range for rendering
        uncertainty = torch.clip(uncertainty, self.config.min_log_uncertainty, self.config.max_log_uncertainty)
        uncertainty = (uncertainty - self.config.min_log_uncertainty) / (
            self.config.max_log_uncertainty - self.config.min_log_uncertainty
        )
        # uncertainty += (1 - torch.sum(weights, dim=-2)) * 1000000 # self.config.min_uncertainty  # alpha blending
        # # normalize into acceptable range for rendering
        # # uncertainty = torch.clip(uncertainty, self.config.min_uncertainty, self.config.max_uncertainty)
        # max_uncertainty = 100
        # uncertainty = torch.clip(uncertainty, 0, max_uncertainty)  # self.config.max_uncertainty
        # uncertainty = uncertainty / max_uncertainty

        outputs["uncertainty"] = uncertainty
        return outputs
