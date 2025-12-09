from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss, scale_gradients_by_distance_squared
from silvr.models.silvr_nerfacto import SilvrNerfactoConfig, SilvrNerfactoModel
from silvr.uncertainty.utils import find_grid_indices, normalize_point_coords


@dataclass
class BayesNerfactoModelConfig(SilvrNerfactoConfig):
    _target: Type = field(default_factory=lambda: BayesNerfactoModel)
    filter_out_point: bool = True
    filter_point_thresh: float = 0.1
    # white_bg: bool = True
    # black_bg: bool = False
    ray_num: int = 4096000
    """Approximate number of rays (train batch size x query iterations)"""
    max_log_uncertainty: float = 6
    """approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian"""
    min_log_uncertainty: float = -3
    """approximate lower bound of that function (cutting off at hessian = 1000)"""
    use_legacy_log_unc: bool = False
    """compute log unc before rendering; From BayesRays but I think is wrong"""


class BayesNerfactoModel(SilvrNerfactoModel):
    """nerfacto + model uncertainty"""

    config: BayesNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

    def load_uncertainty(
        self,
        unc_path: Path,
        filter_out_point: bool = True,
        filter_point_thresh: float = 0.1,  # remove high unc points in the field before rendering
        background_color: Literal["random", "last_sample", "black", "white"] = "white",
    ):
        assert unc_path.exists(), f"Uncertainty file {unc_path} does not exist."
        assert unc_path.suffix == ".npy", f"Uncertainty file {unc_path} must be a .npy file."
        self.uncertainties = np.load(str(unc_path))
        self.uncertainties = torch.tensor(self.uncertainties).to(self.device)
        self.lod = np.log2(round(self.uncertainties.shape[0] ** (1 / 3)) - 1)

        self.config.max_uncertainty = 10**self.config.max_log_uncertainty
        self.config.min_uncertainty = 10**self.config.min_log_uncertainty
        self.config.filter_out_point = filter_out_point
        self.config.filter_point_thresh = filter_point_thresh
        self.config.background_color = background_color
        self.renderer_rgb.background_color = self.config.background_color

    def get_un_points(self, points):
        aabb = self.scene_box.aabb.to(points.device)
        ## samples outside aabb will have 0 coeff and hence 0 uncertainty. To avoid problems with these samples we set zero_out=False
        inds, coeffs = find_grid_indices(
            points, aabb, self.field.spatial_distortion, self.lod, points.device, zero_out=False
        )
        # TODO!also not use log! then check out transient embedding
        # Var(aX+bY) = a^2 Var(X) + b^2 Var(Y) + 2ab Cov(X,Y)
        cfs_2 = (coeffs**2) / torch.sum((coeffs**2), dim=0, keepdim=True)
        uns = self.uncertainties[inds.long()]  # [8,N]
        un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)
        if self.config.use_legacy_log_unc:
            # for stability in volume rendering we use log uncertainty
            un_points = torch.log10(un_points + 1e-12)
            un_points = un_points.view((points.shape[0], points.shape[1], 1))
        else:
            un_points = un_points.view((points.shape[0], points.shape[1], 1))
        return un_points

    def get_outputs(self, ray_bundle: RayBundle):
        if not hasattr(self, "uncertainties"):
            outputs, _, _, _ = super().get_outputs(ray_bundle)
            return outputs
        # If have computed hessian, render uncertainty
        else:
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
        field_outputs = self.field(ray_samples, compute_normals=self.config.compute_normals)
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

    def get_outputs_with_deformation_field(self, ray_bundle, deform_field):
        """reimplementation of get_output function from models because of lack of proper interface to ray_samples"""
        # apply the camera optimizer pose tweaks
        if self.collider is not None:  # from forward pass in base_model.py
            ray_bundle = self.collider(ray_bundle)
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        # Add offsets to the ray samples to obtain gradients w.r.t. the deformation field.
        points = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points, self.scene_box.aabb, self.field.spatial_distortion)
        offsets = deform_field(pos).clone().detach()
        offsets.requires_grad = True
        ray_samples.frustums.set_offsets(offsets)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

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
        return outputs, points, offsets
