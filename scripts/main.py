import argparse
import logging

import yaml
from utils import get_commit_hash, setup_logging

from silvr.app.main import silvr_main
from silvr.app.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SiLVR")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/docker_dev/silvr/config/2024-03-13-roq-01.yaml",
        help="Path to the config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    with open(args.config, "r") as f:
        yaml_data = yaml.safe_load(f)
    config = TrainingConfig(yaml_data)
    logger.info(f"Commit hash: {get_commit_hash()}")
    silvr_main(config)
