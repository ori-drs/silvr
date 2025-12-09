import logging
import subprocess
from datetime import datetime
from pathlib import Path


def setup_logging(level=logging.INFO, save_folder="logs", log_file_prefix="silvr_"):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(save_folder).mkdir(exist_ok=True)
    logging.basicConfig(
        filename=f"{save_folder}/{log_file_prefix}{time}.log",  # Log file
        level=level,  # Set the logging level
        format="%(asctime)s %(levelname)s %(name)s %(lineno)s: %(message)s",  # Log format
    )
    console_handler = logging.StreamHandler()  # Create a console handler
    console_handler.setLevel(logging.INFO)  # Set the logging level
    root_logger = logging.getLogger()  # Get the root logger
    root_logger.addHandler(console_handler)  # Add the console handler to the logger


def get_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        commit_hash = "N/A"
    return commit_hash
