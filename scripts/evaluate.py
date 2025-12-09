from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import rcParams
from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds
from oxford_spires_utils.se3 import is_se3_matrix
from oxford_spires_utils.utils import convert_e57_folder_to_pcd_folder, transform_pcd_folder
from scipy.spatial import cKDTree
from spires_cpp import convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints

from silvr.utils.eval import compute_p2p_distance, get_recon_metrics, save_error_cloud
from silvr.utils.eval_unc import compute_ause, eval_ause, get_recon_unc_metrics
from silvr.utils.io import read_unc_ply, write_unc_ply

rcParams.update(
    {
        "font.size": 18,  # Global font size
        "axes.titlesize": 16,  # Axes title size
        # "axes.labelsize": 14,  # Axes label size
        "xtick.labelsize": 12,  # X-tick label size
        "ytick.labelsize": 12,  # Y-tick label size
        "legend.fontsize": 16,  # Legend font size
    }
)


def get_recon_metrics_new_10cm(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    precision_threshold=0.05,
    recall_threshold=0.05,
    compute_precision=True,
    compute_recall=True,
    save_error_cloud_dir=None,
    csv_path=None,
    max_distance=np.inf,
):
    results = get_recon_metrics_new(
        input_cloud,
        gt_cloud,
        precision_threshold,
        recall_threshold,
        compute_precision,
        compute_recall,
        save_error_cloud_dir,
        csv_path,
        max_distance,
    )
    results_10cm = get_recon_metrics_new(
        input_cloud,
        gt_cloud,
        precision_threshold=0.1,
        recall_threshold=0.1,
        compute_precision=compute_precision,
        compute_recall=compute_recall,
        save_error_cloud_dir=save_error_cloud_dir,
        csv_path=csv_path,
        max_distance=max_distance,
    )
    for metric in ["precision", "recall", "f1_score"]:
        results[metric + "_10cm"] = results_10cm[metric]
    return results


def process_cloud_folder(
    cloud_folder,
    cloud_suffix="pcd",
    output_folder=None,
    octomap_resolution=0.1,
    cloud_downsample_voxel_size=0.01,
    transform_matrix=None,
):
    assert cloud_suffix in ["pcd", "e57"], f"cloud_suffix should be 'pcd' or 'e57', got {cloud_suffix}"
    output_folder = Path(output_folder) if output_folder is not None else (Path(cloud_folder).parent / "output")
    output_folder.mkdir(exist_ok=True)
    cloud_folder = Path(cloud_folder)
    octree_path = output_folder / f"{cloud_folder.stem}_octree.ot"
    cloud_merged_path = output_folder / f"{cloud_folder.stem}_cloud_merged.pcd"

    if cloud_suffix == "e57":
        pcd_folder = cloud_folder.parent / "individual_pcd" if output_folder is None else Path(output_folder)
        pcd_folder.mkdir(exist_ok=True)
        convert_e57_folder_to_pcd_folder(cloud_folder, pcd_folder)
    else:
        pcd_folder = Path(cloud_folder)
    if transform_matrix is not None:
        assert is_se3_matrix(transform_matrix)[0], is_se3_matrix(transform_matrix)[1]
        new_pcd_folder = pcd_folder.parent / f"{pcd_folder.stem}_transformed"
        transform_pcd_folder(pcd_folder, new_pcd_folder, transform_matrix)
        pcd_folder = new_pcd_folder

    processPCDFolder(str(pcd_folder), octomap_resolution, str(octree_path))
    gt_cloud_occ_path = str(octree_path.with_name(f"{octree_path.stem}_occ.pcd"))
    gt_cloud_free_path = str(octree_path.with_name(f"{octree_path.stem}_free.pcd"))
    convertOctreeToPointCloud(str(octree_path), str(gt_cloud_free_path), str(gt_cloud_occ_path))

    _ = merge_downsample_vilens_slam_clouds(pcd_folder, cloud_downsample_voxel_size, cloud_merged_path)


def evaluate_cloud(input_cloud_path, gt_cloud_path):
    gt_cloud = o3d.io.read_point_cloud(str(gt_cloud_path))
    input_cloud = o3d.io.read_point_cloud(str(input_cloud_path))
    save_error_cloud_path = str(Path(input_cloud_path).with_name(f"{Path(input_cloud_path).stem}_error.ply"))
    input_cloud_np = np.array(input_cloud.points)
    gt_cloud_np = np.array(gt_cloud.points)
    save_error_cloud(input_cloud_np, gt_cloud_np, save_error_cloud_path)


def filter_error(lidar_recon_cloud, gt_cloud, target_recon_cloud, voxel_size=0.05, error_threshold=0.5):
    """
    This function filters changes/dynamic objects in the target recon cloud
    It first create a voxel grid using the error cloud (between lidar & gt),
    then filter the target cloud using this voxel grid

    @param recon_cloud: open3d.geometry.PointCloud, LiDAR reconstructed point cloud
    @param gt_cloud: open3d.geometry.PointCloud, ground truth point cloud
    @param target_recon_cloud: open3d.geometry.PointCloud, target (SiLVR) point cloud to be filtered
    """
    assert isinstance(lidar_recon_cloud, o3d.geometry.PointCloud), (
        f"error_cloud should be an open3d.geometry.PointCloud, got {type(lidar_recon_cloud)}"
    )
    assert isinstance(gt_cloud, o3d.geometry.PointCloud), (
        f"target_cloud should be an open3d.geometry.PointCloud, got {type(gt_cloud)}"
    )
    assert isinstance(target_recon_cloud, o3d.geometry.PointCloud), (
        f"target_recon_cloud should be an open3d.geometry.PointCloud, got {type(target_recon_cloud)}"
    )

    lidar_cloud_np = np.array(lidar_recon_cloud.points)
    gt_cloud_np = np.array(gt_cloud.points)
    target_recon_cloud_np = np.array(target_recon_cloud.points)

    # create error cloud between lidar and gt to remove changes
    dist_acc = compute_p2p_distance(lidar_cloud_np, gt_cloud_np)
    error_cloud_np = np.array(lidar_cloud_np[dist_acc > error_threshold])
    error_cloud = o3d.geometry.PointCloud()
    error_cloud.points = o3d.utility.Vector3dVector(error_cloud_np)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(error_cloud, voxel_size=voxel_size)
    voxel_centers = np.array(
        [voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    )
    kdtree = cKDTree(voxel_centers)

    # filter target recon cloud with the voxel grid
    dist, indices = kdtree.query(target_recon_cloud_np, distance_upper_bound=voxel_size)
    remaining_mask = dist > voxel_size
    remaining_cloud_indices = np.where(remaining_mask)[0]
    filtered_cloud_indices = np.where(~remaining_mask)[0]

    target_cloud_remaining = target_recon_cloud.select_by_index(remaining_cloud_indices)
    target_cloud_filtered = target_recon_cloud.select_by_index(filtered_cloud_indices)
    return error_cloud, target_cloud_remaining, target_cloud_filtered, remaining_mask


def combine_rgb_depth_unc(rgb_unc_np, depth_unc_np):
    assert isinstance(rgb_unc_np, np.ndarray) and isinstance(depth_unc_np, np.ndarray), "input should be np.ndarray"
    assert rgb_unc_np.shape[0] == depth_unc_np.shape[0], "shape mismatch"
    return rgb_unc_np


def silvr_filtering(
    recon_cloud_file,
    gt_cloud_file,
    gt_octomap_file=None,
    lidar_cloud_file=None,
    recon_unc_ply_file=None,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else Path(recon_cloud_file).parent
    gt_cloud = o3d.io.read_point_cloud(gt_cloud_file)
    gt_cloud_np = np.array(gt_cloud.points)

    original_recon_cloud = o3d.io.read_point_cloud(recon_cloud_file)
    original_recon_cloud_np = np.array(original_recon_cloud.points)
    recon_remaining_np = original_recon_cloud_np

    if recon_unc_ply_file is not None:
        original_recon_cloud_np_, unc = read_unc_ply(recon_unc_ply_file, return_uncertainty=True)
        assert original_recon_cloud_np.shape[0] == original_recon_cloud_np_.shape[0], "recon cloud mismatch"
        # assert np.allclose(original_recon_cloud_np, original_recon_cloud_np_), "recon cloud mismatch"
        remaining_unc = unc
    # Stage 1: occupancy filtering
    if gt_octomap_file is not None:
        occ_filter_output_dir = output_dir / "occ_filter"
        occ_filter_output_dir.mkdir(exist_ok=True)
        occ_filtered_recon_cloud_file = str(occ_filter_output_dir / f"{Path(recon_cloud_file).stem}_1_occ_filtered.pcd")
        removeUnknownPoints(recon_cloud_file, gt_octomap_file, occ_filtered_recon_cloud_file)
        target_recon_cloud = o3d.io.read_point_cloud(occ_filtered_recon_cloud_file)
        target_recon_cloud_np = np.array(target_recon_cloud.points)
        # save the filtered cloud
        dist = original_recon_cloud.compute_point_cloud_distance(target_recon_cloud)
        diff_mask = np.array(dist) > 0.001
        diff_recon_cloud_np = original_recon_cloud_np[diff_mask]
        if recon_unc_ply_file is not None:
            remaining_unc = unc[~diff_mask]
            assert remaining_unc.shape[0] == target_recon_cloud_np.shape[0], "uncertainty mismatch"
        # save_error_cloud(diff_recon_cloud_np, gt_cloud_np, occ_filtered_recon_cloud_file.replace(".pcd", "_diff.pcd"))
        get_recon_metrics(
            diff_recon_cloud_np,
            gt_cloud_np,
            compute_recall=False,
            save_error_precision_cloud_path=occ_filtered_recon_cloud_file.replace(".pcd", "_diff_p.pcd"),
            csv_path=occ_filtered_recon_cloud_file.replace(".pcd", "_diff.csv"),
        )
        # save_error_cloud(target_recon_cloud_np, gt_cloud_np, occ_filtered_recon_cloud_file.replace(".pcd", "_filtered.pcd"))
        get_recon_metrics(
            target_recon_cloud_np,
            gt_cloud_np,
            compute_recall=False,
            save_error_precision_cloud_path=occ_filtered_recon_cloud_file.replace(".pcd", "_remaining_p.pcd"),
            csv_path=occ_filtered_recon_cloud_file.replace(".pcd", "_remaining.csv"),
        )

    # Stage 2: error filtering
    if lidar_cloud_file is not None:
        lidar_cloud_file = str(lidar_cloud_file) if lidar_cloud_file is not None else occ_filtered_recon_cloud_file
        lidar_recon_cloud = o3d.io.read_point_cloud(lidar_cloud_file)
        error_cloud, recon_remaining, recon_filrtered, mask = filter_error(
            lidar_recon_cloud, gt_cloud, target_recon_cloud, 0.2
        )
        recon_remaining_np = np.array(recon_remaining.points)
        recon_filrtered_np = np.array(recon_filrtered.points)
        if recon_unc_ply_file is not None:
            remaining_unc = remaining_unc[mask]
        error_filter_output_dir = output_dir / "error_filter"
        error_filter_output_dir.mkdir(exist_ok=True)
        error_filter_cloud_file = str(error_filter_output_dir / f"{Path(recon_cloud_file).stem}_error.pcd")
        # save_error_cloud(recon_remaining_np, gt_cloud_np, error_filter_cloud_file.replace(".pcd", "_remaining_error.pcd"))
        # save_error_cloud(recon_filrtered_np, gt_cloud_np, error_filter_cloud_file.replace(".pcd", "_filtered_error.pcd"))
        get_recon_metrics(
            recon_filrtered_np,
            gt_cloud_np,
            compute_recall=False,
            save_error_precision_cloud_path=error_filter_cloud_file.replace(".pcd", "_filtered_p.pcd"),
            csv_path=error_filter_cloud_file.replace(".pcd", "_filtered.csv"),
        )
        o3d.io.write_point_cloud(str(Path(recon_cloud_file).parent / "lidar_error_for_filtering.ply"), error_cloud)
        get_recon_metrics(
            recon_remaining_np,
            gt_cloud_np,
            compute_recall=False,
            save_error_precision_cloud_path=error_filter_cloud_file.replace(".pcd", "_remaining_p.pcd"),
            csv_path=error_filter_cloud_file.replace(".pcd", "_remaining.csv"),
        )
        if recon_unc_ply_file is not None:
            write_unc_ply(
                error_filter_cloud_file.replace(".pcd", "_unc_remainng.ply"),
                recon_remaining_np,
                uncertainties=remaining_unc,
            )
    unc_range_list = [[0.0, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    final_cloud_file = str(output_dir / f"{Path(recon_cloud_file).stem}_unc_remaining_error.pcd")
    evaluate_recon_uncertainty(recon_remaining_np, gt_cloud_np, remaining_unc, final_cloud_file, unc_range_list)

    # no_error_cloud_file = str(error_filter_output_dir / f"{Path(recon_cloud_file).stem}.pcd")
    # o3d.io.write_point_cloud(no_error_cloud_file.replace(".pcd", "_remaining.pcd"), recon_remaining)
    # o3d.io.write_point_cloud(no_error_cloud_file.replace(".pcd", "_filtered.pcd"), recon_filrtered)


def evaluate_recon_uncertainty(recon_np, gt_np, unc_np, error_cloud_path, unc_range_list=[[0.0, 1.0]]):
    distances_acc, _ = get_recon_unc_metrics(
        recon_np,
        gt_np,
        unc_np,
        skip_compl=True,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=str(error_cloud_path),
        csv_path=Path(error_cloud_path).with_suffix(".csv"),
    )
    json_save_path = error_cloud_path.replace(".pcd", "_metrics.json")
    spar_save_path = error_cloud_path.replace(".pcd", "_sparsification.png")
    err_plot_save_path = error_cloud_path.replace(".pcd", "_error.png")
    eval_ause(unc_np, distances_acc, "rmse", json_save_path, spar_save_path, err_plot_save_path)


def silvr_filtering_v2(
    recon_cloud_file, gt_cloud_file, rgb_unc_ply_file, depth_unc_ply_file, loss_unc_ply_file, output_dir=None
):
    output_dir = Path(output_dir) if output_dir is not None else Path(recon_cloud_file).parent
    gt_cloud = o3d.io.read_point_cloud(gt_cloud_file)
    gt_cloud_np = np.array(gt_cloud.points)

    recon_cloud = o3d.io.read_point_cloud(recon_cloud_file)
    recon_cloud_np = np.array(recon_cloud.points)

    rgb_cloud_np, rgb_unc_np = read_unc_ply(rgb_unc_ply_file, return_uncertainty=True)
    assert np.allclose(rgb_cloud_np, recon_cloud_np), "recon cloud mismatch"

    depth_cloud_np, depth_unc_np = read_unc_ply(depth_unc_ply_file, return_uncertainty=True)
    # assert np.allclose(rgb_cloud_np, depth_cloud_np), "rgb depth unc ply mismatch"
    loss_cloud_np, loss_unc_np = read_unc_ply(loss_unc_ply_file, return_uncertainty=True)

    unc_range_list = [[0.0, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    rgb_error_cloud_file = str(output_dir / f"{Path(rgb_unc_ply_file).stem}_unc_remaining_error.pcd")
    depth_error_cloud_file = str(output_dir / f"{Path(depth_unc_ply_file).stem}_unc_remaining_error.pcd")
    loss_error_cloud_file = str(output_dir / f"{Path(loss_unc_ply_file).stem}_unc_remaining_error.pcd")

    distances_acc, _ = get_recon_unc_metrics(
        rgb_cloud_np,
        gt_cloud_np,
        rgb_unc_np,
        skip_compl=True,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=str(rgb_error_cloud_file),
        csv_path=Path(rgb_error_cloud_file).with_suffix(".csv"),
    )
    ause_err_rgb, ause_err_by_var_rgb, ause_rgb, ause_err_slice_interval_list_rgb, ratio_removed_rgb = compute_ause(
        rgb_unc_np, distances_acc, "rmse"
    )
    # save the lowest x% of points based on rgb uncertainty
    bottom_percentage = 0.3
    bottom_percentage_idx = int(rgb_unc_np.shape[0] * bottom_percentage)
    rgb_unc_sorted_idxs = np.argsort(rgb_unc_np)
    rgb_cloud_np_sorted = rgb_cloud_np[rgb_unc_sorted_idxs][:bottom_percentage_idx]
    dist_rgb_sorted = distances_acc[rgb_unc_sorted_idxs][:bottom_percentage_idx]
    save_error_cloud(
        rgb_cloud_np_sorted,
        rgb_error_cloud_file.replace(".pcd", f"_bottom_{bottom_percentage}.pcd"),
        distances=dist_rgb_sorted,
    )

    top_percentage = 0.998
    top_percentage_idx = int(rgb_unc_np.shape[0] * top_percentage)
    rgb_cloud_np_sorted = rgb_cloud_np[rgb_unc_sorted_idxs][top_percentage_idx:]
    dist_rgb_sorted = distances_acc[rgb_unc_sorted_idxs][top_percentage_idx:]
    save_error_cloud(
        rgb_cloud_np_sorted,
        rgb_error_cloud_file.replace(".pcd", f"_top_{top_percentage}.pcd"),
        distances=dist_rgb_sorted,
    )

    distances_acc, _ = get_recon_unc_metrics(
        depth_cloud_np,
        gt_cloud_np,
        depth_unc_np,
        skip_compl=True,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=str(depth_error_cloud_file),
        csv_path=Path(depth_error_cloud_file).with_suffix(".csv"),
    )
    ause_err_depth, ause_err_by_var_depth, ause_depth, ause_err_slice_interval_list_depth, ratio_removed_depth = (
        compute_ause(depth_unc_np, distances_acc, "rmse")
    )

    depth_unc_sorted_idxs = np.argsort(depth_unc_np)
    depth_cloud_np_sorted = depth_cloud_np[depth_unc_sorted_idxs][:bottom_percentage_idx]
    dist_depth_sorted = distances_acc[depth_unc_sorted_idxs][:bottom_percentage_idx]
    save_error_cloud(
        depth_cloud_np_sorted,
        depth_error_cloud_file.replace(".pcd", f"_bottom_{bottom_percentage}.pcd"),
        distances=dist_depth_sorted,
    )

    depth_cloud_np_sorted = depth_cloud_np[depth_unc_sorted_idxs][top_percentage_idx:]
    dist_depth_sorted = distances_acc[depth_unc_sorted_idxs][top_percentage_idx:]
    save_error_cloud(
        depth_cloud_np_sorted,
        depth_error_cloud_file.replace(".pcd", f"_top_{top_percentage}.pcd"),
        distances=dist_depth_sorted,
    )

    distances_acc, _ = get_recon_unc_metrics(
        loss_cloud_np,
        gt_cloud_np,
        loss_unc_np,
        skip_compl=True,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=str(loss_error_cloud_file),
        csv_path=Path(loss_error_cloud_file).with_suffix(".csv"),
    )
    ause_err_loss, ause_err_by_var_loss, ause_loss, ause_err_slice_interval_list_loss, ratio_removed_loss = (
        compute_ause(loss_unc_np, distances_acc, "rmse")
    )

    loss_unc_sorted_idxs = np.argsort(loss_unc_np)
    loss_cloud_np_sorted = loss_cloud_np[loss_unc_sorted_idxs][:bottom_percentage_idx]
    dist_loss_sorted = distances_acc[loss_unc_sorted_idxs][:bottom_percentage_idx]
    save_error_cloud(
        loss_cloud_np_sorted,
        loss_error_cloud_file.replace(".pcd", f"_bottom_{bottom_percentage}.pcd"),
        distances=dist_loss_sorted,
    )

    loss_cloud_np_sorted = loss_cloud_np[loss_unc_sorted_idxs][top_percentage_idx:]
    dist_loss_sorted = distances_acc[loss_unc_sorted_idxs][top_percentage_idx:]
    save_error_cloud(
        loss_cloud_np_sorted,
        loss_error_cloud_file.replace(".pcd", f"_top_{top_percentage}.pcd"),
        distances=dist_loss_sorted,
    )

    assert np.allclose(ratio_removed_rgb, ratio_removed_depth), "ratio removed mismatch"
    # assert np.allclose(ause_err_rgb, ause_err_depth), "ause err mismatch"
    # sparsification plot
    plot_line_width = 3
    plt.figure(figsize=(6.4, 5.0))
    # plt.title("Sparsification plot")
    plt.plot(ratio_removed_rgb, ause_err_rgb, "--", linewidth=plot_line_width)
    plt.plot(ratio_removed_rgb, ause_err_by_var_rgb, "-r", linewidth=plot_line_width)
    plt.plot(ratio_removed_rgb, ause_err_by_var_depth, "-b", linewidth=plot_line_width)
    # plt.plot(ratio_removed_rgb, ause_err_by_var_loss, "-g", linewidth=plot_line_width)
    plt.legend(["Ideal Error", "Error by RGB uncertainty", "Error by depth uncertainty"])
    plt.xlabel("Fraction of pixels removed")
    plt.ylabel("Point-to-point error")
    plt.savefig(rgb_error_cloud_file.replace(".pcd", "_combined_sparsification.png"))
    # error by variance
    plt.clf()
    plt.figure()
    # plt.title("Error by variance")
    plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_rgb[::-1], "-r", linewidth=plot_line_width)
    plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_depth[::-1], "-b", linewidth=plot_line_width)
    # plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_loss[::-1], "-g", linewidth=plot_line_width)
    plt.legend(["Error by RGB uncertainty", "Error by depth uncertainty"])
    plt.xlabel("relative uncertainty")
    # plt.ylabel("Point-to-point error")
    plt.savefig(rgb_error_cloud_file[:-4] + "_interval.png")


def convert_ply_to_pcd(unc_ply_file):
    original_recon_cloud_np_, unc = read_unc_ply(unc_ply_file, return_uncertainty=True)
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(original_recon_cloud_np_)
    o3d.io.write_point_cloud(unc_ply_file.replace(".ply", ".pcd"), o3d_cloud)


def merge_csv(csv_base_dir, only_first_submap=True):
    merged_csv_path = Path(csv_base_dir) / "merged.csv"
    if merged_csv_path.exists():
        merged_csv_path.unlink()
    csv_files = sorted(list(Path(csv_base_dir).rglob("*.csv")))

    last_folder = ""
    for csv_file in csv_files:
        csv_file = str(csv_file)
        if csv_file.split("/")[-2] == "depth_unc":
            continue
        with open(csv_file, "r") as f:
            lines = f.readlines()
            depth_csv_file = csv_file.replace("rgb", "depth")
            if Path(depth_csv_file).exists():
                with open(depth_csv_file, "r") as f2:
                    lines2 = f2.readlines()
                new_lines = [l1.strip() + ", ," + l2.strip() + "\n" for l1, l2 in zip(lines, lines2)]
            else:
                new_lines = lines
            folder = csv_file.split("/")[-4]
            method = csv_file.split("/")[-3]
            fname = csv_file.split("/")[-1]
            setting = f"{folder},{method},{fname}"
            print(setting)

            if only_first_submap and folder == last_folder:
                continue
            with open(merged_csv_path, "a") as f2:
                # write filename
                f2.write(setting)
                f2.write("\n")
                f2.writelines(new_lines)
                f2.write("\n")
            last_folder = folder


if __name__ == "__main__":
    gt_cloud_path = "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/rtc_gt_colmap_frame.ply"
    input_cloud_path = "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/nerf_rgb_merged.ply"
    # evaluate_cloud(input_cloud_path, gt_cloud_path)
    gt_e57_folder = "/media/yifu/Samsung_T7/data/oxford_spires_dataset/ground_truth_cloud/bodleian/individual_cloud_e57"
    # process_cloud_folder(cloud_folder=gt_e57_folder, cloud_suffix="e57")

    lidar_pcd_dir = "/home/yifu/data/nerf_data_pipeline/roq_1/raw/sparse_cloud"
    T_gt_lidar_path = "/home/yifu/data/nerf_data_pipeline/roq_1/raw/T_gt_lidar.txt"
    # transform_matrix = np.loadtxt(T_gt_lidar_path)
    # process_cloud_folder(cloud_folder=lidar_pcd_dir, cloud_suffix="pcd", transform_matrix=transform_matrix)

    ################## bod-2 ###################
    # recon_cloud_pcd_file_list = [
    #     "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/submaps_visibility/2024-11-25_153731/exported_clouds/transformed_cloud.pcd",
    #     "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/submaps_visibility/2024-11-25_154220/exported_clouds/transformed_cloud.pcd",
    #     "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/submaps_visibility/2024-11-25_154717/exported_clouds/transformed_cloud.pcd",
    #     "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/submaps_visibility/2024-11-25_155243/exported_clouds/transformed_cloud.pcd",
    #     "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/submaps_visibility/2024-11-25_160409/exported_clouds/transformed_cloud.pcd",
    # ]

    # gt_cloud_file = "/home/yifu/workspace/T-RO_2025/bod_2_abandon/bod_gt_merged_5cm.pcd"
    # gt_octomap_file = "/home/yifu/workspace/T-RO_2025/bod_2_abandon/gt_octree.ot"
    # lidar_cloud_file = "/home/yifu/workspace/T-RO_2025/bod_2_abandon/silvr/lidar_cropped_submap_0.pcd"

    ################## carla ###################
    run_combine_unc = False
    combined_ply_list = []
    if run_combine_unc:
        rgb_depth_ply_pair_list = [
            {
                "rgb": "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal/silvr/rgb_unc/2025-01-01_132924_rgb.ply",
                "depth": "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal/silvr/depth_unc/2025-01-01_132924_depth.ply",
            },
            {
                "rgb": "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal/silvr/rgb_unc/2025-01-01_133414_rgb.ply",
                "depth": "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal/silvr/depth_unc/2025-01-01_133414_depth.ply",
            },
        ]
        for rgb_depth_ply_pair in rgb_depth_ply_pair_list:
            rgb_recon_np, rgb_unc = read_unc_ply(rgb_depth_ply_pair["rgb"], return_uncertainty=True)
            depth_recon_np, depth_unc = read_unc_ply(rgb_depth_ply_pair["depth"], return_uncertainty=True)
            assert np.allclose(rgb_recon_np, depth_recon_np), "recon cloud mismatch"
            combined_unc = combine_rgb_depth_unc(rgb_unc, depth_unc)
            rgb_unc_ply_file = Path(rgb_depth_ply_pair["rgb"])
            combined_unc_file = rgb_unc_ply_file.parent.parent / "combined_unc" / rgb_unc_ply_file.name
            combined_unc_file.parent.mkdir(exist_ok=True, parents=True)
            write_unc_ply(
                combined_unc_file,
                rgb_recon_np,
                uncertainties=combined_unc,
            )
            combined_ply_list.append(combined_unc_file)
    run_silvr_filtering = True
    # gt_cloud_file = "/home/yifu/workspace/T-RO_2025/carla_lidar_sky_normal_new_eval/carla_map.ply"
    gt_cloud_file = "/home/yifu/workspace/T-RO_2025/roq_1/gt_5cm_lidar_frame.pcd"

    # recon_cloud_ply_file_list = combined_ply_list
    recon_cloud_ply_file_list = [
        # "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal_new_eval/silvr/combined_plot/2025-01-01_132924_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal_new_eval/silvr/combined_plot/2025-01-01_133414_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal_new_eval/silvr/combined_plot/2025-01-01_133918_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/carla_lidar/carla_lidar_sky_normal_new_eval/silvr/combined_plot/2025-01-01_134441_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-normal-sky/combined_unc/2025-01-03_174712_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-normal-sky/combined_unc/2025-01-03_180413_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-normal-sky/combined_unc/2025-01-03_182322_rgb.ply",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-normal-sky/combined_unc/2025-01-03_184318_rgb.ply",
    ]
    recon_folder_list = [
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-sky/bayes-lidar-normal-nerfacto/depth_unc",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-lidar-sky/bayes-lidar-normal-nerfacto/rgb_unc",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-rgb/bayes-lidar-normal-nerfacto/depth_unc",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-rgb/bayes-lidar-normal-nerfacto/rgb_unc",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-rgb-sky/depth_unc",
        # "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-rgb-sky/rgb_unc",
        # "/home/yifu/workspace/T-RO_2025/carla_lidar_sky_normal_new_eval/new/rgb_unc",
        "/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation_new/rgb_unc"
    ]
    for recon_folder in recon_folder_list:
        recon_cloud_ply_file_list += list(Path(recon_folder).glob("*.ply"))

    for recon_cloud_ply_file in recon_cloud_ply_file_list:
        if not run_silvr_filtering:
            continue
        depth_unc_ply_dir = recon_cloud_ply_file.parent.parent / "depth_unc"
        depth_unc_ply_file = depth_unc_ply_dir / (recon_cloud_ply_file.name[:-8] + "_depth.ply")
        depth_unc_ply_file = str(depth_unc_ply_file)
        loss_unc_ply_dir = recon_cloud_ply_file.parent.parent / "loss_unc"
        loss_unc_ply_file = loss_unc_ply_dir / (recon_cloud_ply_file.name[:-8] + "_loss.ply")
        recon_cloud_ply_file = str(recon_cloud_ply_file)
        convert_ply_to_pcd(str(recon_cloud_ply_file))
        # unc_ply_name = Path(recon_cloud_pcd_file).parent.parent.stem + "_depth.ply"
        # recon_cloud_unc_ply_file = str(Path(recon_cloud_pcd_file).parent / unc_ply_name)
        recon_pcd_fname = Path(recon_cloud_ply_file).stem + ".pcd"
        recon_pcd_file = str(Path(recon_cloud_ply_file).parent / recon_pcd_fname)
        # silvr_filtering(
        #     recon_pcd_file,
        #     gt_cloud_file,
        #     # gt_octomap_file,
        #     # lidar_cloud_file=lidar_cloud_file,
        #     recon_unc_ply_file=recon_cloud_ply_file,
        # )
        silvr_filtering_v2(
            recon_pcd_file,
            gt_cloud_file,
            rgb_unc_ply_file=recon_cloud_ply_file,
            depth_unc_ply_file=depth_unc_ply_file,
            loss_unc_ply_file=loss_unc_ply_file,
        )

    # merge_csv("/home/yifu/workspace/T-RO_2025/roq_1/unc_ablation/silvr-rgb-sky/bayes-lidar-normal-nerfacto",False)
