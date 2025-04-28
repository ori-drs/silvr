from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


def get_normal_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    cam_to_world: torch.Tensor,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes normal images.
    Filepath points to a 16-bit or 32-bit normal image, or a numpy array `*.npy`.

    Args:
        filepath: Path to normal image.
        height: Target normal image height.
        width: Target normal image width.
        scale_factor: Factor by which to scale normal image.
        interpolation: Normal value interpolation for resizing.

    Returns:
        Normal image torch tensor with shape [width, height, 3].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath)
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_UNCHANGED)
        # convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## mask for pixel of value 128 128 128
        mask = np.all(image == 128, axis=-1)
        img_encoding = 255.0
        image = image.astype(np.float32) / img_encoding
        assert image.shape[2] == 3
        assert image.shape[0] == height and image.shape[1] == width

        image = cv2.resize(image, (width, height), interpolation=interpolation)

        assert np.all(image >= 0.0) and np.all(image <= 1.0)
        image = image * 2.0 - 1.0
        image = torch.from_numpy(image)
        normal = image.reshape(-1, 3)
        normal = normal.permute(1, 0)  # [3, H*W]
        normal = torch.nn.functional.normalize(normal, p=2, dim=0)

        rot = cam_to_world[:3, :3]

        # normal image in computer vision convention; pose in graphics convention
        graphics_to_colmap = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        graphics_to_colmap = torch.from_numpy(graphics_to_colmap).float()
        normal_world = rot @ graphics_to_colmap @ normal
        image = normal_world.permute(1, 0).reshape(image.shape)
        assert torch.all(image <= 1.0 + 1e-5) and torch.all(
            image >= -1.0 - 1e-5
        ), f"{filepath}: {image.min()}-{image.max()}"
        masked_image = image * ~mask[..., None]
    return masked_image


class LidarDepthDataset(InputDataset):
    """Dataset that returns images and depths.
    Author: Michal Pandy

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        )
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        self.exclude_batch_keys_from_device += ["depth_image"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}


class LidarDepthNormalDataset(LidarDepthDataset):
    """Dataset that returns images, depths, and normals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "normal_filenames" in self.metadata.keys() and self.metadata["normal_filenames"] is not None
        self.normal_filenames = self.metadata["normal_filenames"]
        self.exclude_batch_keys_from_device += ["normal_image"]

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)
        filepath = self.normal_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        camera_to_world = self._dataparser_outputs.cameras.camera_to_worlds[data["image_idx"]]
        normal_image = get_normal_image_from_path(
            filepath=filepath, height=height, width=width, cam_to_world=camera_to_world
        )
        metadata["normal_image"] = normal_image
        return metadata
