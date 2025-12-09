import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from utils import setup_logging

from nerfstudio.data.utils.colmap_parsing_utils import read_next_bytes, read_points3D_binary

logger = logging.getLogger(__name__)


def split_submap_dist(input_json_path, submap_save_dir, submap_num=8, overlap_dist=5):
    with open(input_json_path, "r") as f:
        traj = json.load(f)
    xyz_list = np.array([get_xyz(frame) for frame in traj["frames"]])

    spectral = SpectralClustering(n_clusters=submap_num, affinity="nearest_neighbors", random_state=42)
    clusters = spectral.fit_predict(xyz_list)

    submap_save_dir = Path(submap_save_dir)
    submap_save_dir.mkdir(parents=True, exist_ok=True)
    viz_submap_cluster(clusters, xyz_list, save_path=submap_save_dir / "submap_plt_spectral.png")
    save_submap_cluster(traj, clusters, submap_save_dir, xyz_list, overlap_dist=overlap_dist)


def get_xyz(frame):
    return np.array(frame["transform_matrix"])[:3, 3]


def submap_partitioning_visibility(
    input_json_file, colmap_sparse_folder, output_dir, num_clusters_list=[5], overlap_dist=5
):
    colmap_points_bin_path = Path(colmap_sparse_folder) / "points3D.bin"
    colmap_image_bin_path = Path(colmap_sparse_folder) / "images.bin"

    with open(colmap_image_bin_path, "rb") as fid:
        # read number of images
        num_images = read_next_bytes(fid, 8, "Q")[0]
    with open(input_json_file, "r") as f:
        traj = json.load(f)
    colmap_img_id_to_file_path = {frame["colmap_img_id"]: frame["file_path"] for frame in traj["frames"]}
    logger.info(f"num_images: {num_images}, num_frames: {len(colmap_img_id_to_file_path)}")

    points3D = read_points3D_binary(colmap_points_bin_path)
    max_colmap_img_id = max(colmap_img_id_to_file_path.keys())
    if max_colmap_img_id > num_images:
        logger.warning(
            f"max_colmap_img_id {max_colmap_img_id} > num_images {num_images}. Some images are not registered."
        )

    adjacency_matrix = np.zeros((max_colmap_img_id, max_colmap_img_id), dtype=int)
    xyz_list = np.zeros((max_colmap_img_id, 3), dtype=float)
    traj_colmap_img_idx = []
    for frame in traj["frames"]:
        colmap_img_id = frame["colmap_img_id"]
        xyz_list[colmap_img_id - 1] = np.array(frame["transform_matrix"])[:3, 3]
        traj_colmap_img_idx.append(colmap_img_id - 1)

    for point3D in tqdm.tqdm(points3D.values()):
        for i, image_id in enumerate(point3D.image_ids):
            for j, other_image_id in enumerate(point3D.image_ids):
                if i != j:
                    adjacency_matrix[image_id - 1, other_image_id - 1] += 1  # colmap id starts from 1
    assert is_symmetric_matrix(adjacency_matrix)

    subgraph, mask = get_largest_subgraph(adjacency_matrix)
    logger.info(f"largest subgraph has {subgraph.shape[0]} images")
    logger.info(f"xyz_list before filtering: {xyz_list.shape[0]} points")
    xyz_list = xyz_list[mask]
    logger.info(f"xyz_list after filtering: {xyz_list.shape[0]} points")

    traj_filtered = traj.copy()
    traj_filtered["frames"] = [frame for frame, m in zip(traj["frames"], mask) if m]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_matrix(adjacency_matrix, max_value=500, save_path=(output_dir / "adjacency_matrix.png"))

    for num_clusters in num_clusters_list:
        output_dir_current = output_dir / f"clusters_{num_clusters}"
        output_dir_current.mkdir(parents=True, exist_ok=True)
        clustering = SpectralClustering(n_clusters=num_clusters, affinity="precomputed", random_state=0, n_jobs=-1)
        clustering.fit(subgraph)
        labels = clustering.labels_
        viz_submap_cluster(labels, xyz_list, save_path=(output_dir_current / f"submap_cluster_{num_clusters}.png"))
        # use traj_colmap_imd_idx to get labels for original traj
        labels_full = np.full(max_colmap_img_id, -1, dtype=int)
        labels_full[mask] = labels
        labels_in_traj = np.zeros(len(traj["frames"]), dtype=int) - 1

        for i, colmap_img_idx in enumerate(traj_colmap_img_idx):
            if mask[colmap_img_idx]:
                labels_in_traj[i] = labels_full[colmap_img_idx]
            # else:
            #     logger.warning(f"colmap_img_idx {colmap_img_idx} not in largest subgraph.")
        save_submap_cluster(traj, labels_in_traj, output_dir_current, overlap_dist=overlap_dist)


def is_symmetric_matrix(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)


def get_largest_subgraph(adjacency_matrix):
    sparse_matrix = csr_matrix(adjacency_matrix)
    _, labels = connected_components(sparse_matrix, directed=False)

    component_sizes = np.bincount(labels)
    largest_component_id = np.argmax(component_sizes)
    mask = labels == largest_component_id
    return adjacency_matrix[mask][:, mask], mask


def viz_matrix(matrix, max_value=None, save_path=None):
    colourmap = plt.cm.get_cmap("viridis")
    if max_value is not None:
        matrix = np.clip(matrix, 0, max_value)

    plt.imshow(matrix, cmap=colourmap)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(str(save_path))
    else:
        plt.show()
    plt.close()


def save_submap_cluster(traj, clusters, save_dir, xyz_list=None, overlap_dist=0, viz=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = np.unique(clusters)

    for label in unique_labels:
        output_traj = traj.copy()
        submap_poses = [frame for frame, m in zip(traj["frames"], clusters == label) if m]

        output_traj["frames"] = submap_poses
        save_path = save_dir / f"submap_{label}.json"
        with open(save_path, "w") as f:
            json.dump(output_traj, f, indent=4)


def viz_submap_cluster(clusters, points, save_path=None):
    import matplotlib.pyplot as plt

    unique_labels = np.unique(clusters)
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for label in unique_labels:
        color = colors[label % len(colors)]
        cluster_points = points[clusters == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f"submap_{label}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Submap Clustering")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    # plt.show()


if __name__ == "__main__":
    setup_logging()
    input_json_path = "/home/yifu/projects/silvr_release/data/bod_1_2_merged_colmap/transforms_colmap_scaled_lidar.json"
    save_dir = "/home/yifu/projects/silvr_release/data/bod_1_2_merged_colmap/submaps_vis_new/"
    # split_submap_dist(input_json_path, save_dir, submap_num=6, overlap_dist=10)
    colmap_sparse_folder = "/home/yifu/projects/silvr_release/data/bod_1_2_merged_colmap/sparse/0/"
    submap_partitioning_visibility(
        input_json_path,
        colmap_sparse_folder=colmap_sparse_folder,
        output_dir=save_dir,
        num_clusters_list=[4, 6, 8, 10],
        overlap_dist=5,
    )
