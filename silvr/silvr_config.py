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
from silvr.lidar_depth_nerfacto import LidarDepthNerfactoModelConfig
from silvr.lidar_normal_nerfacto import LidarNormalNerfactoModelConfig
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

Lidar_nerfacto = MethodSpecification(
    TrainerConfig(
        method_name="lidar-nerfacto",
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
