#!/usr/bin/env python
"""
Code adapted from BayesRays https://github.com/BayesRays/BayesRays/tree/main
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pkg_resources
import torch
import tqdm
import tyro

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.eval_utils import eval_setup
from silvr.models.bayes_lidar_normal_nerfacto import BayesLidarNormalNerfactoModel
from silvr.models.bayes_nerfacto import BayesNerfactoModel
from silvr.uncertainty.utils import find_grid_indices

logger = logging.getLogger(__name__)


def set_seeds(seed=1000, torch_deterministic_check=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_deterministic_check:
        torch.use_deterministic_algorithms(True, warn_only=True)  # print warning if not deterministic


@dataclass
class ComputeUncertainty:
    """Load a checkpoint, compute uncertainty, and save it to a npy file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path_rgb: Path = Path("unc_rgb.npy")
    output_path_depth: Path = Path("unc_depth.npy")
    # Uncertainty level of detail (log2 of it)
    lod: int = 8
    # number of iterations on the trainset
    iters: int = -1  # -1 means default iterations from datamanager

    def find_uncertainty_old(self, points, deform_points, rgb, distortion):
        inds, coeffs = find_grid_indices(points, self.aabb, distortion, self.lod, self.device)
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        r = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        colors[1].backward(retain_graph=True)
        g = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        colors[2].backward()
        b = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        dmy = (torch.arange(points.shape[0])[..., None]).repeat((1, points.shape[1])).flatten().to(self.device)
        first = True
        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)
                all_r = coeffs[corner].unsqueeze(-1) * r
                all_g = coeffs[corner].unsqueeze(-1) * g
                all_b = coeffs[corner].unsqueeze(-1) * b
                first = False
            else:
                all_ind = torch.cat(
                    (all_ind, torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)), dim=0
                )
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1) * r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1) * g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1) * b), dim=0)
        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[..., 0])  # for first element of deformation field
        grad_g_1 = torch.bincount(inds_all, weights=all_g[..., 0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[..., 0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[..., 1])  # for second element of deformation field
        grad_g_2 = torch.bincount(inds_all, weights=all_g[..., 1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[..., 1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[..., 2])  # for third element of deformation field
        grad_g_3 = torch.bincount(inds_all, weights=all_g[..., 2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[..., 2])
        grad_1 = grad_r_1**2 + grad_g_1**2 + grad_b_1**2
        grad_2 = grad_r_2**2 + grad_g_2**2 + grad_b_2**2
        grad_3 = grad_r_3**2 + grad_g_3**2 + grad_b_3**2
        # will consider the trace of each submatrix for each deformation
        # vector as indicator of hessian wrt the whole vector

        grads_all = torch.cat((keys_all[:, 1].unsqueeze(-1), (grad_1 + grad_2 + grad_3).unsqueeze(-1)), dim=-1)
        hessian = torch.zeros(((2**self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)

        return hessian

    def find_uncertainty_rgb(self, points, deform_points, rgb, distortion):
        grid_inds, grid_coeffs = find_grid_indices(points, self.aabb, distortion, self.lod, self.device)
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        grad_R = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        colors[1].backward(retain_graph=True)
        grad_G = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        colors[2].backward()
        grad_B = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        ray_inds = (torch.arange(points.shape[0])[..., None]).repeat((1, points.shape[1])).flatten().to(self.device)
        ray_grid = torch.cat(
            [torch.cat((ray_inds.unsqueeze(-1), grid_inds[corner].unsqueeze(-1)), dim=-1) for corner in range(8)], dim=0
        )

        grad_vertices_R = torch.cat([grid_coeffs[corner].unsqueeze(-1) * grad_R for corner in range(8)], dim=0)
        grad_vertices_G = torch.cat([grid_coeffs[corner].unsqueeze(-1) * grad_G for corner in range(8)], dim=0)
        grad_vertices_B = torch.cat([grid_coeffs[corner].unsqueeze(-1) * grad_B for corner in range(8)], dim=0)

        ray_grid_keys, ray_grid_inds = torch.unique(ray_grid, dim=0, return_inverse=True)

        # Squaring before summing - treating each sample along a ray as independent observation, but seems wrong
        # grad_sq_vertices_RGB = grad_vertices_R**2 + grad_vertices_G**2 + grad_vertices_B**2
        # grad_sq_x = torch.bincount(ray_grid_inds, weights=grad_sq_vertices_RGB[..., 0])
        # grad_sq_y = torch.bincount(ray_grid_inds, weights=grad_sq_vertices_RGB[..., 1])
        # grad_sq_z = torch.bincount(ray_grid_inds, weights=grad_sq_vertices_RGB[..., 2])
        grad_R_x = torch.bincount(ray_grid_inds, weights=grad_vertices_R[..., 0])
        grad_G_x = torch.bincount(ray_grid_inds, weights=grad_vertices_G[..., 0])
        grad_B_x = torch.bincount(ray_grid_inds, weights=grad_vertices_B[..., 0])
        grad_R_y = torch.bincount(ray_grid_inds, weights=grad_vertices_R[..., 1])
        grad_G_y = torch.bincount(ray_grid_inds, weights=grad_vertices_G[..., 1])
        grad_B_y = torch.bincount(ray_grid_inds, weights=grad_vertices_B[..., 1])
        grad_R_z = torch.bincount(ray_grid_inds, weights=grad_vertices_R[..., 2])
        grad_G_z = torch.bincount(ray_grid_inds, weights=grad_vertices_G[..., 2])
        grad_B_z = torch.bincount(ray_grid_inds, weights=grad_vertices_B[..., 2])
        grad_sq_x = grad_R_x**2 + grad_G_x**2 + grad_B_x**2
        grad_sq_y = grad_R_y**2 + grad_G_y**2 + grad_B_y**2
        grad_sq_z = grad_R_z**2 + grad_G_z**2 + grad_B_z**2

        grid_grads_sq = torch.cat(
            (ray_grid_keys[:, 1].unsqueeze(-1), (grad_sq_x + grad_sq_y + grad_sq_z).unsqueeze(-1)), dim=-1
        )
        hessian = torch.zeros(((2**self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grid_grads_sq[:, 0]).long(), grid_grads_sq[:, 1], True)

        return hessian

    def find_uncertainty_depth(self, points, deform_points, depth, distortion):
        grid_inds, grid_coeffs = find_grid_indices(points, self.aabb, distortion, self.lod, self.device)
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        depths = torch.sum(depth, dim=0)
        depths.backward(retain_graph=True)
        grad_depth = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()
        ray_inds = (torch.arange(points.shape[0])[..., None]).repeat((1, points.shape[1])).flatten().to(self.device)

        ray_grid = torch.cat(
            [torch.cat([ray_inds.unsqueeze(-1), grid_inds[corner].unsqueeze(-1)], dim=-1) for corner in range(8)], dim=0
        )
        grad_voxel_corners = torch.cat([grid_coeffs[corner].unsqueeze(-1) * grad_depth for corner in range(8)], dim=0)
        ray_grid_keys, ray_grid_inds = torch.unique(ray_grid, dim=0, return_inverse=True)

        # squred gradients and then summed; Treats each sample along a ray as independent observation
        # grad_sq_x = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 0] ** 2)
        # grad_sq_y = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 1] ** 2)
        # grad_sq_z = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 2] ** 2)

        # Adds gradient of the same ray and same voxel before squaring. Seems wrong to me
        grad_x = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 0])
        grad_y = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 1])
        grad_z = torch.bincount(ray_grid_inds, weights=grad_voxel_corners[..., 2])
        grad_sq_x = grad_x**2
        grad_sq_y = grad_y**2
        grad_sq_z = grad_z**2

        grid_grads_sq = torch.cat(
            (ray_grid_keys[:, 1].unsqueeze(-1), (grad_sq_x + grad_sq_y + grad_sq_z).unsqueeze(-1)), dim=-1
        )  # Summing the squared gradients of xyz leads to the spatial uncertainty
        hessian = torch.zeros(((2**self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grid_grads_sq[:, 0]).long(), grid_grads_sq[:, 1], True)

        return hessian

    def compute_variance(self, hessian_likelihood, ray_number, Gaussian_prior):
        hessian_posterior = hessian_likelihood / ray_number + Gaussian_prior
        variance = 1 / hessian_posterior
        return variance

    def main(self) -> None:
        """Main function."""

        assert pkg_resources.get_distribution("nerfstudio").version >= "0.3.1"
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)

        self.output_path_rgb.parent.mkdir(parents=True, exist_ok=True)

        self.device = pipeline.device
        self.aabb = pipeline.model.scene_box.aabb.to(self.device)
        self.hessian_rgb = torch.zeros(((2**self.lod) + 1) ** 3).to(self.device)
        self.hessian_depth = torch.zeros(((2**self.lod) + 1) ** 3).to(self.device)
        self.Gaussian_prior = 1e-4 / ((2**self.lod) ** 3)  # regulariser, prevent zero Hessian
        self.deform_field = HashEncoding(
            num_levels=1,
            min_res=2**self.lod,
            max_res=2**self.lod,
            log2_hashmap_size=self.lod * 3 + 1,  # simple regular grid (hash table size > grid size)
            features_per_level=3,
            hash_init_scale=0.0,
            implementation="torch",
            interpolation="Linear",
        )
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2**self.lod]).to(self.device)

        pipeline.eval()
        iterations = pipeline.datamanager.train_dataset.__len__() if self.iters == -1 else self.iters
        if self.iters != -1 and self.iters < pipeline.datamanager.train_dataset.__len__():
            logger.warning(f"Number of iterations {self.iters} is less than to iterate training images")
        total_ray_num_rgb = 0
        total_ray_num_depth = 0

        assert isinstance(pipeline.model, BayesNerfactoModel), "Only NerfactoModel is supported for now."
        set_seeds(1000)

        logger.info(f"Computing uncertainty for {iterations} iterations")
        for step in tqdm.trange(iterations, desc="Computing model uncertainty", unit="step"):
            ray_bundle, batch = pipeline.datamanager.next_train(step)

            outputs, points, offsets = pipeline.model.get_outputs_with_deformation_field(ray_bundle, self.deform_field)
            # rgb uncertainty
            hessian_rgb = self.find_uncertainty_rgb(
                points, offsets, outputs["rgb"], pipeline.model.field.spatial_distortion
            )
            hessian = hessian_rgb
            total_ray_num_rgb += ray_bundle.shape[0]
            self.hessian_rgb += hessian.clone().detach()

            # depth uncertainty
            if isinstance(pipeline.model, BayesLidarNormalNerfactoModel):
                outputs, points, offsets = pipeline.model.get_outputs_with_deformation_field(
                    ray_bundle, self.deform_field
                )
                valid_depth_mask = batch["depth_image"] > 0
                hessian_depth = self.find_uncertainty_depth(
                    points,
                    offsets,
                    outputs["expected_depth"][valid_depth_mask],
                    pipeline.model.field.spatial_distortion,
                )
                hessian = hessian_depth
                total_ray_num_depth += valid_depth_mask.sum()
                self.hessian_depth += hessian.clone().detach()

        variance_rgb = self.compute_variance(self.hessian_rgb, total_ray_num_rgb, self.Gaussian_prior)
        with open(str(self.output_path_rgb), "wb") as f:
            np.save(f, variance_rgb.cpu().numpy())
            logger.info(f"Saved RGB uncertainty to {self.output_path_rgb}")
        if isinstance(pipeline.model, BayesLidarNormalNerfactoModel):
            variance_depth = self.compute_variance(self.hessian_depth, total_ray_num_depth, self.Gaussian_prior)
            with open(str(self.output_path_depth), "wb") as f:
                np.save(f, variance_depth.cpu().numpy())
                logger.info(f"Saved depth uncertainty to {self.output_path_depth}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeUncertainty).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeUncertainty)  # noqa
