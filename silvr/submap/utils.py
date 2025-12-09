import logging
import shutil
from pathlib import Path

from silvr.silvr_renderer import RenderCameraPath, RenderInterpolated, rename_images, transform_camera_path
from silvr.uncertainty.compute_unc import ComputeUncertainty

logger = logging.getLogger(__name__)


def get_trajs_from_submap_folder(submap_folder):
    trajs = sorted(Path(submap_folder).glob("*.json"))
    assert len(trajs) > 0, f"No submaps found in {submap_folder}."
    return trajs


def create_submap_symlink(submap_folder, data_main_folder):
    trajs = get_trajs_from_submap_folder(submap_folder)
    for i, traj in enumerate(trajs):
        new_json_path = Path(data_main_folder) / traj.name
        relative_path = traj.relative_to(Path(submap_folder).parent)
        new_json_path.symlink_to(relative_path)
        logger.info(f"Created symlink from {traj} to {new_json_path}")
    trajs_full_path = [Path(data_main_folder) / traj.name for traj in trajs]
    return trajs_full_path


def remove_submap_symlink(submap_folder, data_main_folder):
    trajs = get_trajs_from_submap_folder(submap_folder)
    assert len(trajs) > 0, f"No submaps found in {submap_folder}."
    logger.info("Clean up old symlinks")
    for i, traj in enumerate(trajs):
        new_json_path = Path(data_main_folder) / traj.name
        if new_json_path.is_symlink():
            new_json_path.unlink()
        if new_json_path.exists():
            raise RuntimeError(f"Symlink {new_json_path} to be created already exists. Back up and delete it first.")


def compute_uncertainty(nerf_config_path, output_unc_path_rgb, output_unc_path_depth, lod, iterations):
    unc_compute = ComputeUncertainty(
        load_config=Path(nerf_config_path),
        output_path_rgb=output_unc_path_rgb,
        output_path_depth=output_unc_path_depth,
        lod=lod,
        iters=iterations,
    )
    unc_compute.main()


def render_uncertainty(
    nerf_config_path,
    unc_path,
    render_folder_path,
    filter_out_point=False,
    filter_point_thresh=0.1,
    render_downscale=2,
    renderer_type="interpolation",
    camera_path_file="",
    camera_path_model_folder_path="",
    submap_manager=None,
    current_submap_idx=None,
    filter_traj=False,
):
    if renderer_type == "interpolation":
        interpolation_renderer = RenderInterpolated(
            load_config=Path(nerf_config_path),
            unc_path=unc_path,
            interpolation_steps=1,
            frame_rate=1,
            output_format="images",
            output_path=render_folder_path,
            downscale_factor=render_downscale,
            rendered_output_names=["rgb", "depth", "uncertainty"],
            filter_out_point=filter_out_point,
            filter_point_thresh=filter_point_thresh,
        )
        interpolation_renderer.main()
    elif renderer_type == "camera_path":
        if camera_path_file == "":
            logger.error("camera_path_file must be provided for camera_path renderer_type.")
            raise ValueError("camera_path_file is None for camera_path renderer_type.")
        # Transform camera path from its original nerf model's frame into current nerf model's frame
        camera_path_model_dataparser_transform_path = Path(camera_path_model_folder_path) / "dataparser_transforms.json"
        nerf_dataparser_transform_path = Path(nerf_config_path).parent / "dataparser_transforms.json"
        assert submap_manager.submaps[current_submap_idx].render_folder_path == render_folder_path
        if filter_traj and any(render_folder_path.glob("*.jpg")):
            logger.warning(f"⚠️  Render folder {render_folder_path} is not empty. Removing folder")
            shutil.rmtree(render_folder_path)
            Path(render_folder_path).mkdir(parents=True, exist_ok=True)
        camera_path_new_file = transform_camera_path(
            camera_path_file,
            camera_path_model_dataparser_transform_path,
            nerf_dataparser_transform_path,
            filter_traj=filter_traj,
            submap_manager=submap_manager,
            current_submap_idx=current_submap_idx,
        )

        camera_path_renderer = RenderCameraPath(
            load_config=Path(nerf_config_path),
            camera_path_filename=Path(camera_path_new_file),
            unc_path=unc_path,
            output_format="images",
            output_path=render_folder_path,
            downscale_factor=render_downscale,
            rendered_output_names=["rgb", "normals"],
            filter_out_point=filter_out_point,
            filter_point_thresh=filter_point_thresh,
        )
        camera_path_renderer.main()
        if filter_traj:
            rename_images(render_folder_path, image_format="jpg")
    else:
        logger.error(f"Unknown renderer_type: {renderer_type}")
        raise ValueError(f"Unknown renderer_type: {renderer_type}")
