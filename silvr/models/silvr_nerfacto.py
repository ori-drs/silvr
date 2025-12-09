from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

silvr_path = Path(__file__).parent.parent


@dataclass
class SilvrNerfactoConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: SilvrNerfactoModel)
    compute_normals: bool = False


class SilvrNerfactoModel(NerfactoModel):
    """nerfacto + model uncertainty"""

    config: SilvrNerfactoConfig

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.compute_normals)
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

        # if self.config.predict_normals:
        #     normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        #     pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        #     outputs["normals"] = self.normals_shader(normals)
        #     outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # if self.training and self.config.predict_normals:
        #     outputs["rendered_orientation_loss"] = orientation_loss(
        #         weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        #     )

        #     outputs["rendered_pred_normal_loss"] = pred_normal_loss(
        #         weights.detach(),
        #         field_outputs[FieldHeadNames.NORMALS].detach(),
        #         field_outputs[FieldHeadNames.PRED_NORMALS],
        #     )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs, field_outputs, weights, ray_samples
