import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from nerfstudio.scripts.train import entrypoint
from silvr.cloud_exporter import ExportPointCloudSiLVR


def parse_args():
    parser = argparse.ArgumentParser(description="Train SiLVR")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/docker_dev/silvr/config/2024-03-13-roq-01.yaml",
        help="Path to the config file",
    )
    return parser.parse_args()


@dataclass
class BaseTrainingConfig:
    method: str = "nerfacto"
    data: str = "/home/yifu/workspace/nerfstudio_drs/data_dvc_drs/carla_town02_loop/transforms.json"
    vis: str = "wandb"
    max_train_images: int = 400
    max_eval_images: int = -1
    cam_optimiser_mode: str = "off"
    cam_opt_lf: float = 6e-4
    max_num_iterations: int = 30001
    steps_per_eval_image: int = 500
    steps_per_eval_all_images: int = 70000
    output_dir: Path = Path(__file__).absolute().parent / "outputs"


@dataclass
class LidarNerfTrainingConfig:
    lidar_depth_loss_type: str = "DS_NERF_NEW"
    is_euclidean_depth: bool = False
    depth_loss_mult: float = 3e-1
    normal_loss_mult: float = 1e-3
    depth_sigma: float = 0.01
    should_decay_sigma: bool = False
    starting_depth_sigma: float = 0.1
    sigma_decay_rate: float = 0.99985
    save_pcd: bool = False
    # use_transient_embedding: bool = False


@dataclass
class SubmapTrainingConfig:
    run_submap: bool = False
    data_main_folder: str = "/home/yifu/data/silvr/hbac_maths"
    submap_folder: str = "/home/yifu/data/silvr/hbac_maths/submaps_vocab_tree_matcher_1024_True_50_1e-06_50_50"


@dataclass
class PostProcessConfig:
    export_cloud: bool = False


@dataclass
class TrainingConfig:
    base: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    lidar_nerf: LidarNerfTrainingConfig = field(default_factory=LidarNerfTrainingConfig)
    submap: SubmapTrainingConfig = field(default_factory=SubmapTrainingConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        self.base = BaseTrainingConfig(**yaml_data["base"])
        self.lidar_nerf = LidarNerfTrainingConfig(**yaml_data["lidar_nerf"])
        self.submap = SubmapTrainingConfig(**yaml_data["submap"])
        self.post_process = PostProcessConfig(**yaml_data["post_process"])

    def merge_config(self, base_config, lidar_nerf_config):
        config = asdict(base_config)
        if config["method"] in ["lidar-nerfacto", "lidar-depth-nerfacto"]:
            config.update(asdict(lidar_nerf_config))
        return config

    def update_argv(self, config):
        assert sys.argv[0].endswith("train.py") and len(sys.argv) == 1, "No args should be provided."
        for k, v in config.items():
            if k == "method":
                sys.argv.append(f"{v}")
            else:
                sys.argv.append(f"--{k}")
                sys.argv.append(f"{v}")
        print(" ".join(sys.argv))

    def clean_argv(self):
        sys.argv = [sys.argv[0]]

    def update_short_form(self, config):
        short_form = {
            # base config
            "max_train_images": "pipeline.datamanager.train-num-images-to-sample-from",
            "max_eval_images": "pipeline.datamanager.eval-num-images-to-sample-from",
            "cam_optimiser_mode": "pipeline.model.camera-optimizer.mode",
            "cam_opt_lf": "optimizers.camera-opt.optimizer.lr",
            # lidar nerf config
            "lidar_depth_loss_type": "pipeline.model.lidar-depth-loss-type",
            "depth_loss_mult": "pipeline.model.depth-loss-mult",
            "normal_loss_mult": "pipeline.model.normal-loss-mult",
            "depth_sigma": "pipeline.model.depth-sigma",
            "should_decay_sigma": "pipeline.model.should-decay-sigma",
            "starting_depth_sigma": "pipeline.model.starting-depth-sigma",
            "sigma_decay_rate": "pipeline.model.sigma-decay-rate",
            "save_pcd": "pipeline.model.save-pcd",
            "use_transient_embedding": "pipeline.model.use-transient-embedding",
            "is_euclidean_depth": "pipeline.model.is-euclidean-depth",
        }
        for k_old_key, v_new_key in short_form.items():
            if k_old_key in config:
                config[v_new_key] = config.pop(k_old_key)
        return config

    def set_args(self):
        config = self.merge_config(self.base, self.lidar_nerf)
        config = self.update_short_form(config)
        self.update_argv(config)


def run_silvr(config):
    config.set_args()
    entrypoint()
    config.clean_argv()


def run_silvr_submap(config, export_cloud=False, export_cloud_folder="exported_clouds"):
    """create symlinks from the submap_folder to data_main_folder.
    This is because nerfstudio expects the transforms.json to be in the same folder as the images.
    In our case, the submap jsons are not necessarily in the same folder as the images.
    Therefore, we create symlinks to the submap jsons in the data_main_folder, and remove them after training.
    Note that in rendering time, we will need to recreate the symlinks.

    Args:
    data_main_folder: the folder containing images, transforms.json
    submap_folder: the folder containing submaps pose as json files
    """
    trajs = sorted(Path(config.submap.submap_folder).glob("*.json"))
    assert len(trajs) > 0, f"No submaps found in {config.submap.submap_folder}."
    print("Clean up old symlinks")
    data_main_folder = Path(config.submap.data_main_folder)
    for i, traj in enumerate(trajs):
        new_json_path = data_main_folder / traj.name
        if new_json_path.is_symlink():
            new_json_path.unlink()
        if new_json_path.exists():
            raise RuntimeError(f"{new_json_path} already exists. Back up and delete it first.")

    for i, traj in enumerate(trajs):
        new_json_path = data_main_folder / traj.name
        new_json_path.symlink_to(traj)
        config.base.data = str(new_json_path)

        run_silvr(config)
        if config.post_process.export_cloud:
            data_folder = config.base.data if Path(config.base.data).is_dir() else Path(config.base.data).parent
            output_log_dir = config.base.output_dir / data_folder.name / config.base.method
            lastest_output_folder = sorted([x for x in output_log_dir.glob("*") if x.is_dir()])[-1]
            nerf_config = lastest_output_folder / "config.yml"
            saved_cloud_name = f"submap_{i}_{lastest_output_folder.name}.ply"
            export_cloud_folder = Path(export_cloud_folder)
            export_cloud_silvr = ExportPointCloudSiLVR(
                nerf_config, export_cloud_folder, normal_method="open3d", cloud_name=saved_cloud_name
            )
            export_cloud_silvr.run()
        new_json_path.unlink()


if __name__ == "__main__":
    args = parse_args()
    config = TrainingConfig(args.config)
    if not config.submap.run_submap:
        run_silvr(config)
    else:
        run_silvr_submap(config, export_cloud=False)
