from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from silvr.lidar_dataset import LidarDepthDataset, LidarDepthNormalDataset
from silvr.models.bayes_lidar_normal_nerfacto import BayesLidarNormalNerfactoModelConfig
from silvr.models.bayes_nerfacto import BayesNerfactoModelConfig
from silvr.models.lidar_depth_nerfacto import LidarDepthNerfactoModelConfig
from silvr.models.lidar_normal_nerfacto import LidarNormalNerfactoModelConfig
from silvr.silvr_dataparser import LidarDepthNormalDataParserConfig
from silvr.silvr_pipeline import SiLVRPipelineConfig

depth_encoding = 1 / 256.0  # depth value x depth_encoding = depth in meters
Lidar_depth_nerfacto = MethodSpecification(
    TrainerConfig(
        method_name="lidar-depth-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=SiLVRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[LidarDepthDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=NerfstudioDataParserConfig(depth_unit_scale_factor=depth_encoding),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=LidarDepthNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                depth_encoding=depth_encoding,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SiLVR: Lidar Depth nerfacto",
)

Lidar_normal_nerfacto = MethodSpecification(
    TrainerConfig(
        method_name="lidar-normal-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=SiLVRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[LidarDepthNormalDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=LidarDepthNormalDataParserConfig(depth_unit_scale_factor=depth_encoding),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=LidarNormalNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                depth_encoding=depth_encoding,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SiLVR: Lidar Depth Normal nerfacto",
)

Bayes_nerfacto = MethodSpecification(
    TrainerConfig(
        method_name="bayes-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=SiLVRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=BayesNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SiLVR: Bayes nerfacto",
)


Bayes_Lidar_normal_nerfacto = MethodSpecification(
    TrainerConfig(
        method_name="bayes-lidar-normal-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=SiLVRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[LidarDepthNormalDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=LidarDepthNormalDataParserConfig(depth_unit_scale_factor=depth_encoding),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=BayesLidarNormalNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                depth_encoding=depth_encoding,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SiLVR: Bayes Lidar Depth Normal nerfacto",
)

Bayes_Lidar_normal_nerfacto_big = MethodSpecification(
    TrainerConfig(
        method_name="bayes-lidar-normal-nerfacto-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=SiLVRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[LidarDepthNormalDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=LidarDepthNormalDataParserConfig(depth_unit_scale_factor=depth_encoding),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=BayesLidarNormalNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                depth_encoding=depth_encoding,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SiLVR: Bayes Lidar Depth Normal nerfacto",
)
