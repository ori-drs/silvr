import logging
from pathlib import Path

import yaml

from silvr.app.evaluate_unc import evaluate_unc_ply
from silvr.cloud_exporter import ExportPointCloudSiLVR
from silvr.silvr_renderer import merge_images
from silvr.submap.manager import SubmapManager
from silvr.submap.utils import compute_uncertainty, create_submap_symlink, remove_submap_symlink, render_uncertainty
from silvr.uncertainty.compute_unc import set_seeds
from silvr.utils.eval import get_nvs_metrics

logger = logging.getLogger(__name__)


def run_post_processing_only(config):
    # Need this function for cleaning submap symlinks
    if config.submap.run_submap:
        remove_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)
        _ = create_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)

    if config.post_process.render_cam_path_with_submap:
        logger.info("▶️  Creating submap manager for camera path rendering with submaps")
        assert config.submap.run_submap

        input_traj_list = []
        render_folder_list = []
        for i, output_folder in enumerate(config.only_post_process.output_folders):
            render_folder_list.append(
                Path(output_folder) / "post_process" / "renders_cam_path_submap"
            )  # TODO: already defined in post process
            output_config = yaml.load((Path(output_folder) / "config.yml").read_text(), Loader=yaml.Loader)
            input_traj_list.append(output_config.data)
        submap_manager = SubmapManager(input_traj_list, config.only_post_process.output_folders, render_folder_list)
    else:
        submap_manager = None

    for i, output_folder in enumerate(config.only_post_process.output_folders):
        logger.info(
            f"🚀 ({i + 1}/{len(config.only_post_process.output_folders)}) Running post processing on {Path(output_folder).name} "
        )
        run_post_processing(config, output_folder, submap_manager, i)

    if config.post_process.render_cam_path_with_submap:
        logger.info("▶️  Merging images from submaps")
        merge_images(
            config.post_process.camera_path_file,
            Path(config.post_process.camera_path_model_folder_path) / "dataparser_transforms.json",
            submap_manager,
            output_dir="merged_renders",
            image_format="jpg",
        )

    if config.submap.run_submap:
        remove_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)


def run_post_processing(config, output_folder=None, submap_manager=None, current_submap_idx=None):
    if output_folder is None:
        data_folder = config.base.data if Path(config.base.data).is_dir() else Path(config.base.data).parent
        output_log_dir = Path(config.base.output_dir) / data_folder.name / config.base.method
        lastest_output_folder = sorted([x for x in output_log_dir.glob("*") if x.is_dir()])[-1]
        output_folder = lastest_output_folder
    output_folder = Path(output_folder)
    save_folder = output_folder / config.post_process.save_folder_name
    save_folder.mkdir(parents=True, exist_ok=True)

    if config.post_process.compute_nvs_metrics:
        get_nvs_metrics(
            config_path=output_folder / "config.yml",
            output_path=save_folder / "nvs_metrics.json",
        )

    if config.post_process.compute_uncertainty:
        compute_uncertainty(
            nerf_config_path=output_folder / "config.yml",
            output_unc_path_rgb=save_folder / "unc" / "unc_rgb.npy",
            output_unc_path_depth=save_folder / "unc" / "unc_depth.npy",
            lod=config.post_process.unc_grid_lod,
            iterations=config.post_process.unc_iterations,
        )
    if config.post_process.render_uncertainty:
        renderer_type = "interpolation"
        render_uncertainty(
            nerf_config_path=output_folder / "config.yml",
            unc_path=save_folder / "unc" / "unc_rgb.npy",
            render_folder_path=save_folder / "renders_uncertainty_rgb",
            filter_out_point=True,
            filter_point_thresh=config.post_process.render_max_uncertainty_point_legacy,
            render_downscale=config.post_process.render_downscale,
            renderer_type=renderer_type,
            camera_path_file=config.post_process.camera_path_file,
            camera_path_model_folder_path=config.post_process.camera_path_model_folder_path,
        )
        if config.base.method in ["bayes-lidar-normal-nerfacto", "bayes-lidar-normal-nerfacto-big"]:
            render_uncertainty(
                nerf_config_path=output_folder / "config.yml",
                unc_path=save_folder / "unc" / "unc_depth.npy",
                render_folder_path=save_folder / "renders_uncertainty_depth",
                filter_out_point=True,
                filter_point_thresh=config.post_process.render_max_uncertainty_point_legacy,
                render_downscale=config.post_process.render_downscale,
                renderer_type=renderer_type,
                camera_path_file=config.post_process.camera_path_file,
                camera_path_model_folder_path=config.post_process.camera_path_model_folder_path,
            )
    if config.post_process.render_cam_path_with_submap:
        assert submap_manager is not None, "Submap manager is required for rendering with submaps."
        renderer_type = "camera_path"
        render_uncertainty(
            nerf_config_path=output_folder / "config.yml",
            unc_path=save_folder / "unc" / "unc_rgb.npy",
            render_folder_path=save_folder / "renders_cam_path_submap",
            filter_out_point=True,
            filter_point_thresh=config.post_process.render_max_uncertainty_point_legacy,
            render_downscale=config.post_process.render_downscale,
            renderer_type=renderer_type,
            camera_path_file=config.post_process.camera_path_file,
            camera_path_model_folder_path=config.post_process.camera_path_model_folder_path,
            submap_manager=submap_manager,
            current_submap_idx=current_submap_idx,
            filter_traj=True,
        )

    exported_cloud_folder_name = save_folder / "exported_clouds"
    if config.post_process.export_cloud:
        saved_cloud_name_unc_rgb = f"{output_folder.name}_rgb{config.post_process.export_cloud_suffix}"
        set_seeds(torch_deterministic_check=False)  # TODO! do not use deterministic for running batch. temp approach.
        export_cloud_silvr = ExportPointCloudSiLVR(
            output_folder / "config.yml",
            exported_cloud_folder_name,
            normal_method="open3d",
            cloud_name=saved_cloud_name_unc_rgb,
            unc_file=save_folder / "unc" / "unc_rgb.npy",
            filter_out_point=True,
            filter_point_thresh=config.post_process.cloud_max_uncertainty_point_legacy,
            filter_ray_thresh=config.post_process.cloud_max_uncertainty_ray,
            num_points=config.post_process.export_num_points,
        )
        export_cloud_silvr.run()
        if config.base.method == "bayes-lidar-normal-nerfacto":
            export_cloud_silvr.unc_file = save_folder / "unc" / "unc_depth.npy"
            saved_cloud_name_unc_depth = f"{output_folder.name}_depth{config.post_process.export_cloud_suffix}"
            export_cloud_silvr.cloud_name = saved_cloud_name_unc_depth
            export_cloud_silvr.run()
    if config.post_process.evaluate_cloud:
        if not config.post_process.ground_truth_3d_map_path:
            logger.warning("Ground truth path is not set. Skipping evaluation.")
            return
        evaluate_unc_ply(
            ns_output_cloud_folder=exported_cloud_folder_name,
            gt_cloud_file=config.post_process.ground_truth_3d_map_path,
            T_gt_nerf_file=config.post_process.T_gt_nerf_path,
        )
