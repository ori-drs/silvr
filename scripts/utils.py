import logging
from datetime import datetime
from pathlib import Path


def setup_logging():
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename=f"logs/recon_benchmark_{time}.log",  # Log file
        level=logging.DEBUG,  # Set the logging level
        format="%(asctime)s %(levelname)s %(name)s %(lineno)s: %(message)s",  # Log format
    )
    console_handler = logging.StreamHandler()  # Create a console handler
    console_handler.setLevel(logging.INFO)  # Set the logging level
    root_logger = logging.getLogger()  # Get the root logger
    root_logger.addHandler(console_handler)  # Add the console handler to the logger