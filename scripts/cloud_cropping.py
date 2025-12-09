from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from silvr.utils.eval import compute_p2p_distance, get_recon_metrics, save_error_cloud


class EvalCloudCropper:
    def __init__(
        self,
        input_cloud: np.ndarray,
        gt_cloud: np.ndarray,
        output_folder: str,
        input_crop_voxel_grid: Optional[o3d.geometry.VoxelGrid] = None,
        gt_crop_voxel_grid: Optional[o3d.geometry.VoxelGrid] = None,
        voxel_size: float = 0.5,
    ):
        assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3  # N x 3
        self.input_cloud = input_cloud
        self.gt_cloud = gt_cloud
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.voxel_size = voxel_size

    def compute_crop_cloud_proposal(self, max_error: float = 1.0):
        lidar_error_cloud = self.compute_high_error_cloud(self.input_cloud, self.gt_cloud, max_error=max_error)
        lidar_error_cloud.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(str(self.output_folder / "lidar_error_cloud.pcd"), lidar_error_cloud)
        gt_error_cloud = self.compute_high_error_cloud(self.gt_cloud, self.input_cloud, max_error=max_error)
        gt_error_cloud.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(str(self.output_folder / "gt_error_cloud.pcd"), gt_error_cloud)
        return lidar_error_cloud, gt_error_cloud

    def compute_high_error_cloud(self, input_cloud: np.ndarray, gt_cloud: np.ndarray, max_error: float = 1.0):
        input_error_map = compute_p2p_distance(input_cloud, gt_cloud)
        input_error_mask = input_error_map > max_error
        input_error_cloud = input_cloud[input_error_mask]
        input_error_cloud_o3d = o3d.geometry.PointCloud()
        input_error_cloud_o3d.points = o3d.utility.Vector3dVector(input_error_cloud)
        # lidar_error_1m_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return input_error_cloud_o3d

    def compute_crop_voxel_grid(self, cloud: np.ndarray, voxel_size: float = 0.5):
        if isinstance(cloud, np.ndarray):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        assert isinstance(cloud, o3d.geometry.PointCloud)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size)
        return voxel_grid

    def crop_cloud_by_voxel_grid(self, cloud: np.ndarray, voxel_grid: o3d.geometry.VoxelGrid):
        def points_in_voxels(points, voxel_grid):
            queries = o3d.utility.Vector3dVector(points)
            return voxel_grid.check_if_included(queries)

        if isinstance(cloud, np.ndarray):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        assert isinstance(cloud, o3d.geometry.PointCloud)
        points_to_crop = points_in_voxels(cloud.points, voxel_grid)
        points_to_keep = [not x for x in points_to_crop]
        remaining_cloud = cloud.select_by_index(np.where(points_to_keep)[0])
        return remaining_cloud


def convert_voxel_grid_to_point_cloud(voxel_grid):
    points = []
    for voxel in voxel_grid.get_voxels():
        voxel_centre = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        points.append(voxel_centre)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


if __name__ == "__main__":
    folder_path = Path("/home/yifu/workspace/T-RO_2025/Observatory_quater")
    lidar_map_path = folder_path / "ROQ_lidar_map_cleaned.pcd"
    leica_map_path = folder_path / "roq_5cm.pcd"

    T_leica_lidar = np.loadtxt(str(folder_path / "T_RTC_vilens.txt"))

    lidar_map = o3d.io.read_point_cloud(str(lidar_map_path))
    # lidar_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    lidar_map.transform(T_leica_lidar)
    lidar_map = np.array(lidar_map.points)
    leica_map = o3d.io.read_point_cloud(str(leica_map_path))
    leica_map = np.array(leica_map.points)

    output_folder_path = folder_path / "test"
    eval_cloud_cropper = EvalCloudCropper(lidar_map, leica_map, output_folder_path)
    create_proposed_error_cloud = True
    crop_cloud = True
    evaluation = True

    lidar_error_cloud_path = output_folder_path / "lidar_error_cloud.pcd"
    gt_error_cloud_path = output_folder_path / "gt_error_cloud.pcd"
    cropped_lidar_cloud_save_path = output_folder_path / "lidar_cropped.pcd"
    cropped_gt_cloud_save_path = output_folder_path / "gt_cropped.pcd"

    if create_proposed_error_cloud:
        lidar_error_cloud, gt_error_cloud = eval_cloud_cropper.compute_crop_cloud_proposal(max_error=0.2)
        o3d.io.write_point_cloud(str(lidar_error_cloud_path), lidar_error_cloud)
        o3d.io.write_point_cloud(str(gt_error_cloud_path), gt_error_cloud)

    # load error cloud and convert to voxel grid, then crop the map
    if crop_cloud:
        lidar_error_cloud = o3d.io.read_point_cloud(str(lidar_error_cloud_path))
        lidar_crop_voxel_grid = eval_cloud_cropper.compute_crop_voxel_grid(lidar_error_cloud)
        lidar_cropped_cloud = eval_cloud_cropper.crop_cloud_by_voxel_grid(lidar_map, lidar_crop_voxel_grid)
        o3d.io.write_point_cloud(str(cropped_lidar_cloud_save_path), lidar_cropped_cloud)

        gt_error_cloud = o3d.io.read_point_cloud(str(gt_error_cloud_path))
        gt_crop_voxel_grid = eval_cloud_cropper.compute_crop_voxel_grid(gt_error_cloud)
        gt_cropped_cloud = eval_cloud_cropper.crop_cloud_by_voxel_grid(leica_map, gt_crop_voxel_grid)
        o3d.io.write_point_cloud(str(cropped_gt_cloud_save_path), gt_cropped_cloud)

    # apply cropping
    if evaluation:
        lidar_cropped_cloud_np = np.array(o3d.io.read_point_cloud(str(cropped_lidar_cloud_save_path)).points)
        gt_cropped_cloud_np = np.array(o3d.io.read_point_cloud(str(cropped_gt_cloud_save_path)).points)
        print(get_recon_metrics(lidar_cropped_cloud_np, gt_cropped_cloud_np))
        lidar_error_cmap_cloud_save_path = output_folder_path / "lidar_error_cmap_cloud.ply"
        save_error_cloud(lidar_cropped_cloud_np, gt_cropped_cloud_np, lidar_error_cmap_cloud_save_path)
        gt_error_cmap_cloud_save_path = output_folder_path / "gt_error_cmap_cloud.ply"
        save_error_cloud(gt_cropped_cloud_np, lidar_cropped_cloud_np, gt_error_cmap_cloud_save_path)
