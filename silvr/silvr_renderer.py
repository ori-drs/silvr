import json
import shutil
import sys
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import mediapy as media
import numpy as np
import torch
import viser.transforms as tf
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.render import (
    BaseRender,
    CropData,
    entrypoint,
    get_crop_from_json,
    get_path_from_json,
    insert_spherical_metadata_into_file,
)
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from silvr.utils import load_transformation_matrix


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png", "png_unit16"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    render_nearest_camera=False,
    check_occlusions: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        render_nearest_camera: Whether to render the nearest training camera to the rendered camera.
        check_occlusions: If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        if render_nearest_camera:
            assert pipeline.datamanager.train_dataset is not None
            train_dataset = pipeline.datamanager.train_dataset
            train_cameras = train_dataset.cameras.to(pipeline.device)
        else:
            train_dataset = None
            train_cameras = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if render_nearest_camera:
                    assert pipeline.datamanager.train_dataset is not None
                    assert train_dataset is not None
                    assert train_cameras is not None
                    cam_pos = cameras[camera_idx].camera_to_worlds[:, 3].cpu()
                    cam_quat = tf.SO3.from_matrix(cameras[camera_idx].camera_to_worlds[:3, :3].numpy(force=True)).wxyz

                    for i in range(len(train_cameras)):
                        train_cam_pos = train_cameras[i].camera_to_worlds[:, 3].cpu()
                        # Make sure the line of sight from rendered cam to training cam is not blocked by any object
                        bundle = RayBundle(
                            origins=cam_pos.view(1, 3),
                            directions=((cam_pos - train_cam_pos) / (cam_pos - train_cam_pos).norm()).view(1, 3),
                            pixel_area=torch.tensor(1).view(1, 1),
                            nears=torch.tensor(0.05).view(1, 1),
                            fars=torch.tensor(100).view(1, 1),
                            camera_indices=torch.tensor(0).view(1, 1),
                            metadata={},
                        ).to(pipeline.device)
                        outputs = pipeline.model.get_outputs(bundle)

                        q = tf.SO3.from_matrix(train_cameras[i].camera_to_worlds[:3, :3].numpy(force=True)).wxyz
                        # calculate distance between two quaternions
                        rot_dist = 1 - np.dot(q, cam_quat) ** 2
                        pos_dist = torch.norm(train_cam_pos - cam_pos)
                        dist = 0.3 * rot_dist + 0.7 * pos_dist

                        if true_max_dist == -1 or dist < true_max_dist:
                            true_max_dist = dist
                            true_max_idx = i

                        if outputs["depth"][0] < torch.norm(cam_pos - train_cam_pos).item():
                            continue

                        if check_occlusions and (max_dist == -1 or dist < max_dist):
                            max_dist = dist
                            max_idx = i

                    if max_idx == -1:
                        max_idx = true_max_idx
                with torch.autocast(pipeline.device.type, enabled=True):  # TODO check if else is needed
                    if crop_data is not None:
                        with renderers.background_color_override_context(
                            crop_data.background_color.to(pipeline.device)
                        ), torch.no_grad():
                            outputs = pipeline.model.get_outputs_for_camera(
                                cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                            )
                    else:
                        with torch.no_grad():
                            outputs = pipeline.model.get_outputs_for_camera(
                                cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                            )

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        if rendered_output_name == "depth_metric_uint16":
                            output_image = output_image.cpu().numpy().astype(np.uint16)
                        else:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=depth_near_plane,
                                    far_plane=depth_far_plane,
                                    colormap_options=colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                    elif rendered_output_name == "normals":
                        output_image = pipeline.model.normals_shader(outputs["normals"]).cpu().numpy()
                        accum = outputs["accumulation"].cpu().numpy()
                        output_image = output_image * accum + (1 - accum)
                    elif rendered_output_name == "accumulation":
                        output_image = (outputs["accumulation"].cpu().numpy() * 255).astype(np.uint8)
                    else:
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                # Add closest training image to the right of the rendered image
                if render_nearest_camera:
                    assert train_dataset is not None
                    assert train_cameras is not None
                    img = train_dataset.get_image_float32(max_idx)
                    height = cameras.image_height[0]
                    # maintain the resolution of the img to calculate the width from the height
                    width = int(img.shape[1] * (height / img.shape[0]))
                    resized_image = torch.nn.functional.interpolate(
                        img.permute(2, 0, 1)[None], size=(int(height), int(width))
                    )[0].permute(1, 2, 0)
                    resized_image = (
                        colormaps.apply_colormap(
                            image=resized_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                    render_image.append(resized_image)

                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    if rendered_output_name == "depth_metric_uint16":
                        cv2.imwrite(str(output_image_dir / f"{camera_idx:05d}.png"), render_image)
                    elif rendered_output_name == "accumulation":
                        cv2.imwrite(str(output_image_dir / f"{camera_idx:05d}.png"), render_image)
                    else:
                        if image_format == "png":
                            media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                        if image_format == "jpeg":
                            media.write_image(
                                output_image_dir / f"{camera_idx:05d}.jpg",
                                render_image,
                                fmt="jpeg",
                                quality=jpeg_quality,
                            )
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "images"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                check_occlusions=self.check_occlusions,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path) + "//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path) + "//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path) + "//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path) + "//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


def update_camera_path(camera_paths_file, T_new_old, new_camera_path_file=None):
    """Transform camera path from old coordinate system to new coordinate system
    new_traj = T_new_old @ old_traj

    Args:
        camera_paths_file (str): path to the camera path file
        T_new_old (np.ndarray): 4x4 transformation matrix
        new_camera_path_file (str, optional): path to the new camera path file. Add "_new" to the original file name if not provided.
    """
    camera_paths_file = Path(camera_paths_file)
    with open(camera_paths_file, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
    for camera in camera_path["camera_path"]:
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        if c2w[3] != [0, 0, 0, 1]:
            c2w[3] = [0, 0, 0, 1]
        c2w_new = T_new_old @ c2w
        camera["camera_to_world"] = c2w_new.reshape(-1).tolist()
    if new_camera_path_file is None:
        new_camera_path_file = camera_paths_file.parent / (camera_paths_file.stem + "_new.json")
    with open(new_camera_path_file, "w", encoding="utf-8") as f:
        json.dump(camera_path, f, indent=4)


def merge_images(
    camera_path_model,
    submap_manager,
    output_dir,
    image_format="png",
    interpolate=False,
    interpolate_threshold=2,
    accum_folder=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(camera_path_model.camera_path, "r", encoding="utf-8") as f:
        camera_paths = json.load(f)

    for i, camera in enumerate(camera_paths["camera_path"]):
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        T_nerf_metric = load_transformation_matrix(camera_path_model.training_dataparser_transforms_path)
        c2w_metric = np.linalg.inv(T_nerf_metric) @ c2w
        submap_idx, dist_1 = submap_manager.get_i_nearest_submap(c2w_metric[:2, 3], i=0, return_dist=True)
        second_nearest_submap, dist_2 = submap_manager.get_i_nearest_submap(c2w_metric[:2, 3], i=1, return_dist=True)
        if accum_folder is not None:
            accum = cv2.imread(str(accum_folder / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
            assert len(accum.shape) == 2
            accum = accum[..., None] / 255
        if dist_2 < interpolate_threshold and interpolate:
            print(f"Interpolating {i:05d}.{image_format}")
            img_1 = cv2.imread(str(submap_manager.submaps[submap_idx].render_folder_path / f"{i:05d}.{image_format}"))
            img_2 = cv2.imread(
                str(submap_manager.submaps[second_nearest_submap].render_folder_path / f"{i:05d}.{image_format}")
            )
            alpha = (interpolate_threshold - dist_1 + 0.1) / (2 * interpolate_threshold - dist_2 - dist_1 + 0.2)
            img = cv2.addWeighted(img_1, alpha, img_2, 1 - alpha, 0)
            if accum_folder is not None:
                img = img * accum + 255 * (1 - accum)
            cv2.imwrite(str(output_dir / f"{i:05d}.{image_format}"), img)
        else:
            shutil.copy(
                submap_manager.submaps[submap_idx].render_folder_path / f"{i:05d}.{image_format}",
                output_dir / f"{i:05d}.{image_format}",
            )
            if accum_folder is not None:
                img = cv2.imread(str(output_dir / f"{i:05d}.{image_format}"))
                img = img * accum + 255 * (1 - accum)
                cv2.imwrite(str(output_dir / f"{i:05d}.{image_format}"), img)


def render(config):
    config.update_argv()
    entrypoint()
    config.clean_argv()


def filter_traj_outside_submap(
    camera_paths_file, submap_manager, current_submap_idx, indice_file_name="filtered_indices.json", tolerance=5
):
    camera_paths_file = Path(camera_paths_file)
    with open(camera_paths_file, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
    filtered_camera_path = deepcopy(camera_path)
    remaining_indices = [i for i in range(len(camera_path["camera_path"]))]
    submap_poses = submap_manager.submaps[current_submap_idx].all_poses
    for i, camera in enumerate(camera_path["camera_path"]):
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)
        assert np.allclose(c2w[3], [0, 0, 0, 1])
        dist = np.linalg.norm(np.array(list(submap_poses.values())) - c2w[:2, 3], axis=1).min()
        if current_submap_idx != submap_manager.get_i_nearest_submap(c2w[:2, 3]) and dist > tolerance:
            filtered_camera_path["camera_path"].remove(camera)
            remaining_indices.remove(i)
    with open(camera_paths_file, "w", encoding="utf-8") as f:
        json.dump(filtered_camera_path, f, indent=4)
    save_filtered_indices_path = submap_manager.submaps[current_submap_idx].render_folder_path / indice_file_name
    save_filtered_indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_filtered_indices_path, "w", encoding="utf-8") as f:
        json.dump(remaining_indices, f, indent=4)


def rename_images(render_folder, indice_file_name="filtered_indices.json", image_format="png"):
    render_folder = Path(render_folder)
    images = list(render_folder.glob(f"*.{image_format}"))
    images.sort()
    tmp_folder = render_folder.parent / (render_folder.stem + "_tmp")
    indice_file_path = render_folder / indice_file_name
    with open(indice_file_path, "r", encoding="utf-8") as f:
        remaining_indices = json.load(f)
    tmp_folder.mkdir(parents=True, exist_ok=True)
    assert len(remaining_indices) == len(images)
    for i, idx in enumerate(remaining_indices):
        shutil.move(images[i], tmp_folder / f"{idx:05d}.{image_format}")
    shutil.move(indice_file_path, tmp_folder / indice_file_name)
    shutil.rmtree(render_folder, ignore_errors=True)
    tmp_folder.rename(render_folder)


def render_submaps(
    camera_path_model, submap_manager, render_output_names=["rgb"], image_format="png", filter_traj=False
):
    camera_path = camera_path_model.camera_path
    for i, trained_nerf_submap in enumerate(submap_manager.submaps):
        """
        # p_current_model = scale_old @ T_old @ p_metric
        # p_campath_model = scale_new @ T_new @ p_metric
        # p_current_model = T_currentmodel_campathmodel @ p_campath_model
        # T_currentmodel_campathmodel = scale_new @ T_new @ T_old^-1 @ scale_old^-1
        """
        T_currentmodel_metric = load_transformation_matrix(trained_nerf_submap.training_dataparser_transforms_path)
        T_campathmodel_metric = load_transformation_matrix(camera_path_model.training_dataparser_transforms_path)
        T_metric_campathmodel = np.linalg.inv(T_campathmodel_metric)
        new_camera_path_file = camera_path.parent / (camera_path.stem + "_new.json")
        update_camera_path(camera_path, T_metric_campathmodel, new_camera_path_file)
        if filter_traj:
            filter_traj_outside_submap(new_camera_path_file, submap_manager, i)
        update_camera_path(new_camera_path_file, T_currentmodel_metric, new_camera_path_file)
        render_camera_path = RenderCameraPath(Path(trained_nerf_submap.training_config_path))
        render_camera_path.camera_path_filename = new_camera_path_file
        render_camera_path.rendered_output_names = render_output_names
        render_camera_path.image_format = image_format
        render_camera_path.output_path = trained_nerf_submap.render_folder_path
        # create symlink of submap json to the main data folder TODO
        render_camera_path.main()
        if filter_traj:
            rename_images(trained_nerf_submap.render_folder_path)


def render_from_metric_traj(render_camera_path, training_dataparser_transforms_path):
    camera_path = Path(render_camera_path.camera_path_filename)
    T_currentmodel_metric = load_transformation_matrix(training_dataparser_transforms_path)
    new_camera_path_file = camera_path.parent / (camera_path.stem + "_nerf_scale.json")
    update_camera_path(camera_path, T_currentmodel_metric, new_camera_path_file)
    render_camera_path.camera_path_filename = new_camera_path_file
    render_camera_path.main()
