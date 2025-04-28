from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class SiLVRPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SiLVRPipeline)
    """target class to instantiate"""
    # datamanager: DataManagerConfig = DataManagerConfig()
    # """specifies the datamanager config"""
    # model: ModelConfig = ModelConfig()
    # """specifies the model config"""


class SiLVRPipeline(VanillaPipeline):
    """SiLVR pipeline"""

    def __init__(
        self,
        config: SiLVRPipelineConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        # self._model.config.dataparser_outputs = self.datamanager.train_dataparser_outputs
        self._model.config.dataparser_scale = self.datamanager.train_dataparser_outputs.dataparser_scale

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        with torch.autocast(self.device.type, enabled=True):
            return super().get_eval_loss_dict(step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        with torch.autocast(self.device.type, enabled=True):
            return super().get_eval_image_metrics_and_images(step)

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        with torch.autocast(self.device.type, enabled=True):
            return super().get_average_eval_image_metrics(step, output_path, get_std)
