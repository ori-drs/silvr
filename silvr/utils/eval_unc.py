import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from silvr.utils.eval import get_distances, get_recon_metrics_from_dist, save_error_cloud
from silvr.utils.io import dict_to_csv, read_unc_ply

logger = logging.getLogger(__name__)


def get_recon_unc_metrics(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    input_unc: np.ndarray,
    precision_threshold=0.05,
    recall_threshold=0.05,
    compute_precision=True,
    compute_recall=True,
    save_error_cloud_path=None,
    csv_path=None,
    max_distance=np.inf,
    unc_min_max_list=[(0.0, 1.0)],
):
    assert isinstance(input_cloud, np.ndarray) and isinstance(gt_cloud, np.ndarray)
    assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3
    distance_precision, distance_recall = get_distances(
        input_cloud,
        gt_cloud,
        compute_precision=compute_precision,
        compute_recall=compute_recall,
        max_distance=max_distance,
    )
    for min_unc, max_unc in unc_min_max_list:
        mask = (input_unc > min_unc) & (input_unc < max_unc)
        distances_acc_ = distance_precision[mask]
        if distances_acc_.shape[0] == 0:
            logger.warning(f"Input cloud is empty for min_unc: {min_unc}, max_unc: {max_unc}")
            continue
        # distances_cmpl_ = distance_recall[mask] # TODO: do we need to compute this?
        results = get_recon_metrics_from_dist(
            distances_acc_,
            distance_recall,  # TODO: is this correct?
            precision_threshold=precision_threshold,
            recall_threshold=recall_threshold,
        )
        logger.debug(f"unc: {min_unc} - {max_unc}, {distances_acc_.shape},  {results}")
        if save_error_cloud_path:
            Path(save_error_cloud_path).parent.mkdir(parents=True, exist_ok=True)
            new_save_error_cloud_path = str(save_error_cloud_path)[:-4] + f"_{min_unc}_{max_unc}.ply"
            save_error_cloud(input_cloud[mask], new_save_error_cloud_path, distances=distances_acc_)
        if csv_path:
            dict_to_csv(
                {"max_unc": max_unc, "min_unc": min_unc, **results},
                csv_path,
            )
    return distance_precision, distance_recall


def evaluate_cloud_unc(
    input_cloud_path,
    gt_cloud_path,
    T_gt_nerf_path="",
    save_error_cloud_path=None,
    unc_min_max_list=None,
    save_csv_path=None,
):
    input_cloud_np, unc = read_unc_ply(input_cloud_path, return_uncertainty=True)
    gt_cloud = o3d.io.read_point_cloud(str(gt_cloud_path))
    assert gt_cloud.has_points(), "GT cloud is empty"
    if T_gt_nerf_path != "":
        T_gt_nerf = np.loadtxt(T_gt_nerf_path)
        input_cloud_np = np.dot(input_cloud_np, T_gt_nerf[:3, :3].T) + T_gt_nerf[:3, 3]
        o3d_save = o3d.geometry.PointCloud()
        o3d_save.points = o3d.utility.Vector3dVector(input_cloud_np)
        o3d.io.write_point_cloud(str(input_cloud_path)[:-4] + "_gt.ply", o3d_save)
    gt_cloud_np = np.array(gt_cloud.points)
    distances_acc, distances_cmpl = get_recon_unc_metrics(
        input_cloud_np, gt_cloud_np, unc, unc_min_max_list=unc_min_max_list, csv_path=save_csv_path
    )
    # compute sparsefication plot
    eval_ause(
        unc,
        distances_acc,
        "rmse",
        json_save_path=str(input_cloud_path)[:-4] + "_metrics.json",
        spar_save_path=str(input_cloud_path)[:-4] + "_sparsification.png",
        err_plot_save_path=str(input_cloud_path)[:-4] + "_error.png",
    )

    if save_error_cloud_path:
        for min_unc, max_unc in unc_min_max_list:
            mask = (unc > min_unc) & (unc < max_unc)
            save_error_cloud_path_ = str(save_error_cloud_path)[:-4] + f"_{min_unc}_{max_unc}.ply"
            save_error_cloud(input_cloud_np[mask], gt_cloud_np, save_error_cloud_path_, distances_acc[mask])


def save_dict_to_json(dict, save_path):
    with open(save_path, "w") as f:
        json.dump(dict, f, indent=4)


def eval_ause(var_vec, err_vec, err_type="rmse", json_save_path=None, spar_save_path=None, err_plot_save_path=None):
    ause_err, ause_err_by_var, ause, ause_err_slice_interval_list, ratio_removed = compute_ause(
        var_vec,
        err_vec,
        "rmse",
    )
    results_dict = {
        "ause": ause,
        "ause_err": ause_err.tolist(),
        "ause_err_by_var": ause_err_by_var.tolist(),
        "ratio_removed": ratio_removed.tolist(),
        "ause_err_slice_interval_list": ause_err_slice_interval_list,
    }
    save_dict_to_json(results_dict, json_save_path)

    plot_sparsification(ause, ratio_removed, ause_err, ause_err_by_var, spar_save_path)

    plot_rmse_by_unc(ratio_removed, ause_err_slice_interval_list, err_plot_save_path)


# https://github.com/abdo-eldesokey/pncnn/blob/c6122e9c442eabeb0145b241121aeba0039eb5e7/utils/sparsification_plot.py#L10
# also see https://github.com/BayesRays/BayesRays/blob/edd549e323654c26d52797e43ef17de842befeef/bayesrays/metrics/ause.py#L6
def compute_ause(var_vec, err_vec, err_type="rmse", save_rmse_by_var=True, old_impl=False):
    # Sort the error
    err_vec_sorted = np.sort(err_vec)
    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    ratio_removal_steps = 100
    ratio_removed = np.linspace(0, 1, ratio_removal_steps, endpoint=False)
    for i, r in enumerate(ratio_removed):
        err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "mse":
            ause_err.append(np.sqrt(err_slice.mean()))
        elif err_type in ["rmse", "mae"]:
            ause_err.append(err_slice.mean())
        else:
            raise ValueError(f"Unknown error mode {err_type}")

    # Normalize RMSE. This is not used. See normalisation below
    # ause_err = ause_err / ause_err[0]  # first ause_err is largest

    # Sort by variance
    var_vec = np.sqrt(var_vec)
    var_vec_sorted_idxs = np.argsort(var_vec)

    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    ause_err_by_var = []
    ause_err_slice_interval_list = []
    for i, r in enumerate(ratio_removed):
        err_slice = err_vec_sorted_by_var[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "mse":
            ause_err_by_var.append(np.sqrt(err_slice.mean()))
        elif err_type in ["rmse", "mae"]:
            ause_err_by_var.append(err_slice.mean())

        if save_rmse_by_var:
            ause_err_slice_interval = err_vec_sorted_by_var[
                int((1 - r - 1 / ratio_removal_steps) * n_valid_pixels) : int((1 - r) * n_valid_pixels)
            ]
            if err_type == "mse":
                ause_err_slice_interval_list.append(np.sqrt(ause_err_slice_interval.mean()))
            elif err_type in ["rmse", "mae"]:
                ause_err_slice_interval_list.append(ause_err_slice_interval.mean())

    # Normalize RMSE, this ensures that normalisation is fair, better than original version. Same as BayesRays
    if old_impl:
        ause_err = ause_err / max(ause_err)
        ause_err_by_var = ause_err_by_var / max(ause_err_by_var)

    else:
        max_err = max(*ause_err, *ause_err_by_var)
        ause_err = ause_err / max_err
        ause_err_by_var = ause_err_by_var / max_err

    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)

    return ause_err, ause_err_by_var, ause, ause_err_slice_interval_list, ratio_removed


def plot_sparsification(ause, ratio_removed, ause_err, ause_err_by_var, save_path):
    plt.clf()
    plt.figure()
    plt.title(f"Sparsification plot with AUSE={ause:.4f}")
    plt.plot(ratio_removed, ause_err, "--")
    plt.plot(ratio_removed, ause_err_by_var, "-r")
    plt.legend(["RMSE", "Normalised RMSE by variance"])
    plt.xlabel("Fraction of pixels removed")
    plt.ylabel("RMSE")
    plt.savefig(save_path)


def plot_rmse_by_unc(ratio_removed, ause_err_slice_interval_list, save_path):
    plt.clf()
    plt.figure()
    plt.title("RMSE by variance")
    plt.plot(ratio_removed, ause_err_slice_interval_list[::-1], "-r")
    plt.xlabel("relative uncertainty")
    plt.ylabel("MSE")
    plt.savefig(save_path[:-4] + "_interval.png")
