import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# from spires_cpp import convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints
from silvr.utils.eval import save_error_cloud
from silvr.utils.eval_unc import eval_ause, get_recon_unc_metrics
from silvr.utils.io import read_unc_ply

logger = logging.getLogger(__name__)


def convert_ply_to_pcd(unc_ply_file):
    original_recon_cloud_np_, unc = read_unc_ply(unc_ply_file, return_uncertainty=True)
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(original_recon_cloud_np_)
    o3d.io.write_point_cloud(unc_ply_file.replace(".ply", ".pcd"), o3d_cloud)


def plot_sparsification(
    ratio_removed_rgb, ause_err_rgb, ause_err_by_var_rgb, ause_err_by_var_depth, save_path="sparsification.png"
):
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
    plt.savefig(save_path)


def plot_error_by_variance(
    ratio_removed_rgb, ause_err_slice_interval_list_rgb, ause_err_slice_interval_list_depth, save_path
):
    plot_line_width = 3
    plt.figure()
    # plt.title("Error by variance")
    plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_rgb[::-1], "-r", linewidth=plot_line_width)
    plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_depth[::-1], "-b", linewidth=plot_line_width)
    # plt.plot(ratio_removed_rgb, ause_err_slice_interval_list_loss[::-1], "-g", linewidth=plot_line_width)
    plt.legend(["Error by RGB uncertainty", "Error by depth uncertainty"])
    plt.xlabel("relative uncertainty")
    # plt.ylabel("Point-to-point error")
    plt.savefig(save_path)


def silvr_filtering_v2(
    recon_cloud_file, gt_cloud_file, rgb_unc_ply_file, depth_unc_ply_file, T_gt_nerf=None, output_dir=None
):
    output_dir = Path(output_dir) if output_dir is not None else Path(recon_cloud_file).parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_cloud = o3d.io.read_point_cloud(gt_cloud_file)
    gt_cloud_np = np.array(gt_cloud.points)

    recon_cloud = o3d.io.read_point_cloud(recon_cloud_file)
    recon_cloud_np = np.array(recon_cloud.points)

    rgb_cloud_np, rgb_unc_np = read_unc_ply(rgb_unc_ply_file, return_uncertainty=True)
    assert np.allclose(rgb_cloud_np, recon_cloud_np), "recon cloud mismatch"

    depth_cloud_np, depth_unc_np = read_unc_ply(depth_unc_ply_file, return_uncertainty=True)
    assert np.allclose(rgb_cloud_np, depth_cloud_np), "rgb depth unc ply mismatch"

    if T_gt_nerf is not None:
        assert T_gt_nerf.shape == (4, 4), "T_gt_nerf should be a 4x4 matrix"
        rgb_cloud_np = np.dot(rgb_cloud_np, T_gt_nerf[:3, :3].T) + T_gt_nerf[:3, 3]
        depth_cloud_np = np.dot(depth_cloud_np, T_gt_nerf[:3, :3].T) + T_gt_nerf[:3, 3]
    # loss_cloud_np, loss_unc_np = read_unc_ply(loss_unc_ply_file, return_uncertainty=True)

    unc_range_list = [[0.0, 1.0], [0.0, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    rgb_error_cloud_file = output_dir / f"{Path(rgb_unc_ply_file).stem}_unc_remaining_error.pcd"
    depth_error_cloud_file = output_dir / f"{Path(depth_unc_ply_file).stem}_unc_remaining_error.pcd"
    # loss_error_cloud_file = str(output_dir / f"{Path(loss_unc_ply_file).stem}_unc_remaining_error.pcd")

    logger.info(f"evaluating {Path(rgb_unc_ply_file).stem} against {Path(gt_cloud_file).stem}")
    rgb_error_cloud_threshold_folder = rgb_error_cloud_file.parent / "rgb_error_cloud"
    distances_acc, _ = get_recon_unc_metrics(
        rgb_cloud_np,
        gt_cloud_np,
        rgb_unc_np,
        compute_recall=False,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=rgb_error_cloud_threshold_folder / "error.pcd",
        csv_path=rgb_error_cloud_file.with_suffix(".csv"),
    )
    save_top_bottom_unc(rgb_cloud_np, rgb_unc_np, distances_acc, rgb_error_cloud_threshold_folder, 0.998, 0.3)

    rgb_json_save_path = str(rgb_error_cloud_file).replace(".pcd", "_metrics.json")
    spar_save_path = str(rgb_error_cloud_file).replace(".pcd", "_sparsification.png")
    err_plot_save_path = str(rgb_error_cloud_file).replace(".pcd", "_error.png")
    eval_ause(rgb_unc_np, distances_acc, "rmse", rgb_json_save_path, spar_save_path, err_plot_save_path)

    logger.info(f"evaluating {Path(depth_unc_ply_file).stem} against {Path(gt_cloud_file).stem}")
    depth_error_cloud_threshold_folder = depth_error_cloud_file.parent / "depth_error_cloud"
    distances_acc, _ = get_recon_unc_metrics(
        depth_cloud_np,
        gt_cloud_np,
        depth_unc_np,
        compute_recall=False,
        unc_min_max_list=unc_range_list,
        save_error_cloud_path=depth_error_cloud_threshold_folder / "error.pcd",
        csv_path=depth_error_cloud_file.with_suffix(".csv"),
    )
    save_top_bottom_unc(depth_cloud_np, depth_unc_np, distances_acc, depth_error_cloud_threshold_folder, 0.998, 0.3)

    depth_json_save_path = str(depth_error_cloud_file).replace(".pcd", "_metrics.json")
    spar_save_path = str(depth_error_cloud_file).replace(".pcd", "_sparsification.png")
    err_plot_save_path = str(depth_error_cloud_file).replace(".pcd", "_error.png")
    eval_ause(depth_unc_np, distances_acc, "rmse", depth_json_save_path, spar_save_path, err_plot_save_path)

    combine_rgb_depth_plots(rgb_json_save_path, depth_json_save_path)


def save_top_bottom_unc(cloud_np, unc_np, distances_acc, save_folder, top_percentage=0.998, bottom_percentage=0.3):
    # save the lowest x% of points based on rgb uncertainty
    bottom_percentage_idx = int(unc_np.shape[0] * bottom_percentage)
    rgb_unc_sorted_idxs = np.argsort(unc_np)
    rgb_cloud_np_sorted = cloud_np[rgb_unc_sorted_idxs][:bottom_percentage_idx]
    dist_rgb_sorted = distances_acc[rgb_unc_sorted_idxs][:bottom_percentage_idx]
    save_error_cloud(
        rgb_cloud_np_sorted,
        str(save_folder / f"bottom_{bottom_percentage}.pcd"),
        distances=dist_rgb_sorted,
    )

    top_percentage_idx = int(unc_np.shape[0] * top_percentage)
    rgb_cloud_np_sorted = cloud_np[rgb_unc_sorted_idxs][top_percentage_idx:]
    dist_rgb_sorted = distances_acc[rgb_unc_sorted_idxs][top_percentage_idx:]
    save_error_cloud(
        rgb_cloud_np_sorted,
        str(save_folder / f"top_{top_percentage}.pcd"),
        distances=dist_rgb_sorted,
    )


def combine_rgb_depth_plots(rgb_metrics_file, depth_metrics_file):
    with open(rgb_metrics_file, "r") as f:
        data = json.load(f)
    ratio_removed_rgb = np.array(data["ratio_removed"])
    ause_err_rgb = np.array(data["ause_err"])
    ause_err_by_var_rgb = np.array(data["ause_err_by_var"])
    ause_err_slice_interval_list_rgb = np.array(data["ause_err_slice_interval_list"])

    with open(depth_metrics_file, "r") as f:
        data = json.load(f)
    ratio_removed_depth = np.array(data["ratio_removed"])
    assert np.allclose(ratio_removed_rgb, ratio_removed_depth), "ratio removed mismatch"
    ause_err_depth = np.array(data["ause_err"])
    if not np.allclose(ause_err_rgb, ause_err_depth):
        logger.error("ause err mismatch")  # TODO: different ause err normalisation. Might use old_impl to solve this
    ause_err_by_var_depth = np.array(data["ause_err_by_var"])
    ause_err_slice_interval_list_depth = np.array(data["ause_err_slice_interval_list"])

    plot_sparsification(
        ratio_removed_rgb,
        ause_err_rgb,
        ause_err_by_var_rgb,
        ause_err_by_var_depth,
        save_path=rgb_metrics_file.replace(".json", "_combined_sparsification.png"),
    )
    plot_error_by_variance(
        ratio_removed_rgb,
        ause_err_slice_interval_list_rgb,
        ause_err_slice_interval_list_depth,
        save_path=rgb_metrics_file.replace(".json", "_error_by_variance.png"),
    )


def evaluate_unc_ply(ns_output_cloud_folder, gt_cloud_file, T_gt_nerf_file=None):
    T_gt_nerf = np.loadtxt(T_gt_nerf_file) if (T_gt_nerf_file is not None and T_gt_nerf_file != "") else None
    recon_cloud_ply_file_list = list(Path(ns_output_cloud_folder).glob("*.ply"))

    for recon_cloud_ply_file in recon_cloud_ply_file_list:
        if not recon_cloud_ply_file.name.endswith("_rgb.ply"):
            continue
        depth_unc_ply_file = recon_cloud_ply_file.parent / (recon_cloud_ply_file.name[:-8] + "_depth.ply")
        depth_unc_ply_file = str(depth_unc_ply_file)
        recon_cloud_ply_file = str(recon_cloud_ply_file)
        convert_ply_to_pcd(str(recon_cloud_ply_file))

        recon_pcd_fname = Path(recon_cloud_ply_file).stem + ".pcd"
        recon_pcd_file = str(Path(recon_cloud_ply_file).parent / recon_pcd_fname)
        silvr_filtering_v2(
            recon_pcd_file,
            gt_cloud_file,
            rgb_unc_ply_file=recon_cloud_ply_file,
            depth_unc_ply_file=depth_unc_ply_file,
            T_gt_nerf=T_gt_nerf,
        )
