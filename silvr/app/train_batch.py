import logging
from copy import deepcopy
from pathlib import Path

import yaml

from silvr.app.main import silvr_main
from silvr.app.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def update_yaml_fileds(base_yaml, new_yaml):
    white_list_keys = ["label", "skip"]

    def update_recursive(base_dict, new_dict):
        for key in new_dict:
            if isinstance(new_dict[key], dict):
                if key in base_dict:
                    update_recursive(base_dict[key], new_dict[key])
            else:
                if key in base_dict or key in white_list_keys:
                    base_dict[key] = new_dict[key]
                    logger.info(f"⚙️  Updated {key} to {new_dict[key]}")
                else:
                    logger.warning(f"⚠️  Key {key} not found in base config. Skipping update.")

    yaml_data = deepcopy(base_yaml)
    update_recursive(yaml_data, new_yaml)

    return yaml_data


def run_batch(yaml_batch_path, base_yaml_path):
    with open(Path(yaml_batch_path), "r") as f:
        yaml_batch = yaml.safe_load(f)
    with open(Path(base_yaml_path), "r") as f:
        yaml_base = yaml.safe_load(f)
    for individual_yaml in yaml_batch:
        if individual_yaml["skip"]:
            logger.info(f"ℹ️  Skipping batch: {individual_yaml['label']}")
            continue
        yaml_data = update_yaml_fileds(yaml_base, individual_yaml)
        config = TrainingConfig(yaml_data)
        silvr_main(config)
