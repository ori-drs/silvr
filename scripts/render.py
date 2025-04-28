from pathlib import Path

from silvr.silvr_renderer import RenderCameraPath, merge_images, render_from_metric_traj, render_submaps
from silvr.submap import CameraTrajNeRF, SubmapManager

if __name__ == "__main__":
    nerfstudio_config = "/home/docker_dev/silvr/outputs/ori/lidar-depth-nerfacto/2024-02-29_232702/config.yml"
    render_camera_path = RenderCameraPath(Path(nerfstudio_config))
    render_camera_path.camera_path_filename = (
        "/home/docker_dev/data/ori/camera_paths/converted_from_tum_cam_traj_nerfstudio.json"
    )
    render_camera_path.rendered_output_names = ["rgb"]
    render_camera_path.image_format = "png"
    render_camera_path.output_path = Path("renders/rgb")

    # render(config)

    render_from_metric_traj(
        render_camera_path,
        "/home/docker_dev/silvr/outputs/ori/lidar-depth-nerfacto/2024-02-29_232702/dataparser_transforms.json",
    )

    # submapping
    camera_path_model = CameraTrajNeRF(
        trained_model_folder_path="/home/yifu/workspace/silvr/outputs/hbac_maths/lidar-nerfacto/2024-01-24_144359",
        input_trajectory_path="",
        camera_path="/home/yifu/data/silvr/hbac_maths/camera_paths/2024-01-24-14-44-01.json",
    )

    submap_list = {
        # input_traj: trained_nerf
        "submap_1": "2024-01-23_191816",
        "submap_2": "2024-01-23_192849",
        "submap_3": "2024-01-23_193958",
        "submap_4": "2024-01-23_195150",
        "submap_5": "2024-01-23_200436",
        "submap_6": "2024-01-23_201833",
        # "submap_7": "2024-01-23_203250",
    }
    trained_model_list = [
        Path("/home/yifu/workspace/silvr/outputs/hbac_maths/lidar-nerfacto") / submap_list[x] for x in submap_list
    ]
    input_traj_list = [Path("/home/yifu/data/silvr/hbac_maths") / f"{x}.json" for x in submap_list]
    render_folder = Path("/home/yifu/workspace/silvr/renders")
    render_folder_list = [render_folder / submap_list[x] for x in submap_list]
    submap_manager = SubmapManager(input_traj_list, trained_model_list, render_folder_list)

    render_submaps(
        camera_path_model,
        submap_manager,
        render_output_names=render_camera_path.rendered_output_names,
        image_format=render_camera_path.image_format,
        filter_traj=True,
    )
    merged_folder = render_folder / "merged"
    merge_images(camera_path_model, submap_manager, merged_folder, image_format=render_camera_path.image_format)
