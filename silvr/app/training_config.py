import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


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
    output_dir: Path = Path(__file__).absolute().parent.parent.parent / "outputs"


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
class PostProcessConfig:
    save_folder_name: str = "post_process"
    compute_nvs_metrics: bool = False
    compute_uncertainty: bool = False
    unc_iterations: int = -1  # number of iterations when computing Hessian; -1 means detault from datamanager
    unc_grid_lod: int = 8  # Uncertainty grid map resolution in terms of level of detail. Resolution= 2^lod
    render_uncertainty: bool = False
    render_max_uncertainty_point_legacy: float = 1  # 0-1, for render, lower is less points that are more accurate
    render_downscale: float = 2  # for rendering's resolution. 1 is not downscaled, 2 is half resolution
    render_cam_path_with_submap: bool = False
    camera_path_file: str = ""  # optional camera path for rendering uncertainty
    camera_path_model_folder_path: str = ""  # optional trained model folder path for camera path
    cloud_max_uncertainty_point_legacy: float = 1  # 0-1, for exported cloud. filter points before rendering.
    cloud_max_uncertainty_ray: float = 0.3  # 0-1, filter a rendered ray. For exporting cloud
    export_cloud: bool = False
    export_cloud_suffix: str = ""
    export_num_points: int = 1000000
    evaluate_cloud: bool = True
    ground_truth_3d_map_path: str = ""
    T_gt_nerf_path: str = ""


@dataclass
class OnlyPostProcessConfig:
    turn_on: bool = False
    output_folders: List[str] = field(default_factory=list)


@dataclass
class SubmapTrainingConfig:
    run_submap: bool = False
    data_main_folder: str = "/home/yifu/data/silvr/hbac_maths"
    submap_folder: str = "/home/yifu/data/silvr/hbac_maths/submaps_vocab_tree_matcher_1024_True_50_1e-06_50_50"
    export_cloud: bool = False
    compute_uncertainty: bool = True
    render_uncertainty: bool = True


@dataclass
class TrainingConfig:
    base: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    lidar_nerf: LidarNerfTrainingConfig = field(default_factory=LidarNerfTrainingConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    submap: SubmapTrainingConfig = field(default_factory=SubmapTrainingConfig)

    def __init__(self, yaml_data):
        self.base = BaseTrainingConfig(**yaml_data["base"])
        self.lidar_nerf = LidarNerfTrainingConfig(**yaml_data["lidar_nerf"])
        self.post_process = PostProcessConfig(**yaml_data["post_process"])
        self.only_post_process = OnlyPostProcessConfig(**yaml_data["only_post_process"])
        self.submap = SubmapTrainingConfig(**yaml_data["submap"])

        if self.post_process.compute_uncertainty or self.post_process.render_uncertainty:
            assert self.base.method in [
                "bayes-nerfacto",
                "bayes-lidar-normal-nerfacto",
                "bayes-lidar-normal-nerfacto-big",
            ], "Uncertainty can only be computed for bayes-nerfacto method."

    def merge_config(self, base_config, lidar_nerf_config):
        config = asdict(base_config)
        if config["method"] in [
            "lidar-normal-nerfacto",
            "lidar-depth-nerfacto",
            "bayes-lidar-normal-nerfacto",
            "bayes-lidar-normal-nerfacto-big",
        ]:
            config.update(asdict(lidar_nerf_config))
        return config

    def update_argv(self, config):
        supported_files = ["main.py", "batch.py"]
        assert Path(sys.argv[0]).name in supported_files, f"Unsupported file: {sys.argv[0]}"
        if Path(sys.argv[0]).name == "main.py":
            # remove the other arg parsers. TODO: refactor this code
            sys.argv = [sys.argv[0]]
        assert len(sys.argv) == 1, f"Extra args found: {sys.argv}"
        for k, v in config.items():
            if k == "method":
                sys.argv.append(f"{v}")
            else:
                sys.argv.append(f"--{k}")
                sys.argv.append(f"{v}")
        logger.info(" ".join(sys.argv))

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
