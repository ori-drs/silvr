import csv
import json
from pathlib import Path

import numpy as np


def load_dataparser_transform(path):
    """Load the transform and scale from the dataparser_transforms.json file.
    p_nerf = T_nerf_metric @ p_metric = scale @ transform @ p_metric
    """
    with open(path, "r") as f:
        data_transforms = json.load(f)
        transform = data_transforms["transform"]
        transform = np.array(transform)
        transform = np.vstack([transform, np.array([0, 0, 0, 1])])
        scale = data_transforms["scale"]
    return transform, scale


def load_transformation_matrix(path):
    transform, scale = load_dataparser_transform(path)
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, :3] *= scale
    T_nerf_metric = scaling_matrix @ transform
    return T_nerf_metric


def dict_to_csv(dict_data, filename):
    with open(filename, "a", newline="") as output_file:
        dict_writer = csv.writer(output_file)
        dict_writer.writerow(dict_data.keys())
        dict_writer.writerow(dict_data.values())


def filter_cloud_by_uncertainty(input_cloud: np.ndarray, uncertainty: np.ndarray, min_unc=0.0, max_unc=0.5):
    assert input_cloud.shape[0] == uncertainty.shape[0]
    mask = (uncertainty > min_unc) & (uncertainty < max_unc)
    filtered_cloud = input_cloud[mask]
    return filtered_cloud


def read_unc_ply(filename, min_unc=0.0, max_unc=0.5, return_uncertainty=False):
    assert Path(filename).suffix == ".ply", "File must be a .ply file"
    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip header
    property_list = []
    for i, line in enumerate(lines):
        if line.startswith("property"):
            # get the word after second space and before \n
            field = line.split()[2]
            assert field in ["x", "y", "z", "red", "green", "blue", "uncertainty", "scalar_uncertainty"]
            property_list.append(field)
        if line == "end_header\n":
            break
    data = np.loadtxt(lines[i + 1 :])

    xyz_indices = [i for i, field in enumerate(property_list) if field in ["x", "y", "z"]]
    assert len(xyz_indices) == 3, "must have x, y, z fields"
    # rgb_indices = [i for i, field in enumerate(property_list) if field in ["red", "green", "blue"]]
    points = data[:, xyz_indices]
    if "uncertainty" in property_list:
        uncertainty_index = property_list.index("uncertainty")
        uncertainties = data[:, uncertainty_index]
    elif "scalar_uncertainty" in property_list:
        uncertainty_index = property_list.index("scalar_uncertainty")
        uncertainties = data[:, uncertainty_index]
    else:
        raise ValueError("No uncertainty field found in the .ply file")
    if return_uncertainty:
        return points, uncertainties

    points = filter_cloud_by_uncertainty(points, uncertainties, min_unc, max_unc)

    return points


def read_unc_ply_multi(filenames, min_uncs, max_uncs):
    # check files
    assert len(filenames) == len(min_uncs) == len(max_uncs)
    clouds = []
    for filename, min_unc, max_unc in zip(filenames, min_uncs, max_uncs):
        assert Path(filename).suffix == ".ply", "File must be a .ply file"
        assert 0 <= min_unc <= 1
        assert 0 <= max_unc <= 1
        assert min_unc <= max_unc
        # make sure all clouds have the same xyz, just different uncertainties
        cloud = read_unc_ply(filename, min_unc=0.0, max_unc=1)
        clouds.append(cloud)
        assert np.allclose(cloud, clouds[0])

    clouds = []
    for filename, min_unc, max_unc in zip(filenames, min_uncs, max_uncs):
        cloud = read_unc_ply(filename, min_unc, max_unc)
        clouds.append(cloud)
    # TODO! merge, but can also get intersection, or compute the total uncertainty
    clouds_cat = np.concatenate(clouds)
    _, indices = np.unique(clouds_cat, axis=0, return_index=True)
    clouds = clouds_cat[indices]

    return clouds


def write_unc_ply(filename, points, colors=None, uncertainties=None):
    num_points = len(points)
    if colors is not None:
        assert colors.shape[0] == num_points
    if uncertainties is not None:
        assert uncertainties.shape[0] == num_points

    with open(filename, "w") as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if uncertainties is not None:
            f.write("property float uncertainty\n")
        f.write("end_header\n")

        # Write data
        data = [points]
        fmt = ["%.6f", "%.6f", "%.6f"]

        if colors is not None:
            colors_int = (colors * 255).astype(int)
            data.append(colors_int)
            fmt.extend(["%d", "%d", "%d"])

        if uncertainties is not None:
            data.append(uncertainties.reshape(-1, 1))
            fmt.append("%.6f")

        combined_data = np.hstack(data)
        np.savetxt(f, combined_data, fmt=" ".join(fmt))
