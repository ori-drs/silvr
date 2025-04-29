from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import HashEncoding, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import MLP, NerfactoField
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss
from nerfstudio.utils.external import tcnn
from silvr.lidar_depth_nerfacto import LidarDepthNerfactoModel, LidarDepthNerfactoModelConfig
from silvr.loss import monosdf_normal_loss


class SiLVRMLPWithHashEncoding(MLPWithHashEncoding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tcnn_hash_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=HashEncoding.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                interpolation=kwargs.get("interpolation", None),
            ),
            # dtype=torch.float32,
        )
        mlp = MLP(
            in_dim=tcnn_hash_encoding.n_output_dims,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation="torch",
        )
        self.model = nn.Sequential(tcnn_hash_encoding, mlp)


class HybridNerfactoField(NerfactoField):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mlp_base = SiLVRMLPWithHashEncoding(
            num_levels=kwargs.get("num_levels"),
            min_res=kwargs.get("base_res"),
            max_res=kwargs.get("max_res"),
            log2_hashmap_size=kwargs.get("log2_hashmap_size"),
            features_per_level=kwargs.get("features_per_level"),
            hash_init_scale=kwargs.get("hash_init_scale"),
            # num_layers=kwargs.get("num_layers"), # note that num_layers is not in args so using default
            layer_width=kwargs.get("hidden_dim"),
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=kwargs.get("implementation"),
        )

    def get_normals(self) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of normals.

        Args:
            density: Tensor of densities.
        """
        assert self._sample_locations is not None, "Sample locations must be set before calling get_normals."
        assert self._density_before_activation is not None, "Density must be set before calling get_normals."
        assert (
            self._sample_locations.shape[:-1] == self._density_before_activation.shape[:-1]
        ), "Sample locations and density must have the same shape besides the last dimension."

        normals = torch.autograd.grad(
            self._density_before_activation,
            self._sample_locations,
            grad_outputs=torch.ones_like(self._density_before_activation),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        normals = -torch.nn.functional.normalize(normals, dim=-1)

        return normals


@dataclass
class LidarNormalNerfactoModelConfig(LidarDepthNerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LidarNormalNerfactoModel)
    normal_loss_mult: float = 1e-3


class LidarNormalNerfactoModel(LidarDepthNerfactoModel):
    """nerfacto + lidar depth + nerf normal"""

    config: LidarNormalNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = HybridNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=True)
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

        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        outputs["normals"] = normals  #! TODO shader?
        # if self.config.predict_normals:
        #     normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        #     pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        #     outputs["normals"] = self.normals_shader(normals)
        #     outputs["pred_normals"] = self.normals_shader(pred_normals)
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

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                # if output_name == "density" or output_name == "normals":
                output = output.detach().to("cpu")
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["normal_loss"] = 0.0
            normal_gt = batch["normal_image"].to(self.device)
            normal_nerf = outputs["normals"]
            valid_depth_mask = ((batch["depth_image"] > 0) & (batch["depth_image"] < self.config.sky_depth_nerf))[
                ..., 0
            ]
            metrics_dict["normal_loss"] = monosdf_normal_loss(normal_gt, normal_nerf, valid_depth_mask)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and "normal_loss" in metrics_dict
            loss_dict["normal_loss"] = self.config.normal_loss_mult * metrics_dict["normal_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        for key, value in outputs.items():
            outputs[key] = value.to(self.device)
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        rendered_normal = self.normals_shader(outputs["normals"])
        rendered_normal = rendered_normal * outputs["accumulation"] + (1 - outputs["accumulation"])
        gt_normal = self.normals_shader(batch["normal_image"]) if "normal_image" in batch else None
        images["rendered_normal"] = (
            torch.cat([gt_normal, rendered_normal], dim=1) if gt_normal is not None else rendered_normal
        )
        del images["prop_depth_0"]
        del images["prop_depth_1"]
        return metrics, images
