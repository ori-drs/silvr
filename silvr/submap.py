import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class TrainedNeRF:
    _trained_model_folder_path: Path
    input_trajectory_path: Path
    render_folder_path: Path = None
    training_config_path: Path = None
    training_dataparser_transforms_path: Path = None

    @property
    def trained_model_folder_path(self):
        return self._trained_model_folder_path

    @trained_model_folder_path.setter
    def trained_model_folder_path(self, trained_model_folder_path):
        """Set the config and dataparser_transforms path based on the output_folder_path"""
        trained_model_folder_path = Path(trained_model_folder_path)
        self._trained_model_folder_path = trained_model_folder_path
        self.training_config_path = trained_model_folder_path / "config.yml"
        self.training_dataparser_transforms_path = trained_model_folder_path / "dataparser_transforms.json"

    def __init__(self, trained_model_folder_path, input_trajectory_path, render_folder_path=None):
        self.trained_model_folder_path = Path(trained_model_folder_path)
        self.input_trajectory_path = Path(input_trajectory_path)
        if render_folder_path is not None:
            self.render_folder_path = Path(render_folder_path)
        self.all_poses = {}
        with open(self.input_trajectory_path, "r", encoding="utf-8") as f:
            submap_input_traj = json.load(f)
            for frame in submap_input_traj["frames"]:
                xy = np.array(frame["transform_matrix"])[:2, 3]
                self.all_poses[frame["file_path"]] = xy

    def compute_min_dist(self, query_pose_xy):
        dist = np.linalg.norm(np.array(list(self.all_poses.values())) - query_pose_xy, axis=1)
        min_dist_idx = np.argsort(dist)[0]
        min_dist = dist[min_dist_idx]
        return min_dist


@dataclass
class CameraTrajNeRF(TrainedNeRF):
    camera_path: Path = None  # trajectory generated from ns viewer

    def __init__(self, trained_model_folder_path, input_trajectory_path, camera_path):
        super().__init__(trained_model_folder_path, input_trajectory_path)
        self.camera_path = Path(camera_path)


@dataclass
class SubmapManager:
    submaps: List[TrainedNeRF]

    def __init__(self, input_traj_list, trained_model_list, render_folder_list=None):
        self.submaps = []
        if render_folder_list is None:
            for input_traj, trained_model in zip(input_traj_list, trained_model_list):
                self.submaps.append(TrainedNeRF(trained_model, input_traj))
        else:
            for input_traj, trained_model, render_folder in zip(
                input_traj_list, trained_model_list, render_folder_list
            ):
                self.submaps.append(TrainedNeRF(trained_model, input_traj, render_folder))

    def get_submap_dist(self, query_pose_xy):
        dists = {}
        for i, submap in enumerate(self.submaps):
            dists[i] = submap.compute_min_dist(query_pose_xy)
        return dists

    def get_i_nearest_submap(self, query_pose_xy, i=0, return_dist=False):
        # i=0 nearest; i=1 second nearest etc
        dists = self.get_submap_dist(query_pose_xy)
        sorted_dists = sorted(dists.items(), key=lambda x: x[1])
        i_nearest_submap_idx = sorted_dists[i][0]
        return i_nearest_submap_idx if not return_dist else (i_nearest_submap_idx, sorted_dists[i][1])
