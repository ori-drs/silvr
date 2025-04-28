import logging
import shutil
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download
from utils import setup_logging

logger = logging.getLogger(__name__)


class HFDatasetDownloader:
    def __init__(self, hf_repo_id: str, local_base_dir: str = "data"):
        self.repo_id = hf_repo_id
        self.local_base_dir = Path(local_base_dir)
        self.local_base_dir.mkdir(parents=True, exist_ok=True)
        self.remote_files = self.load_all_remote_files()

    def load_all_remote_files(self):
        try:
            remote_files = sorted(list_repo_files(self.repo_id, repo_type="dataset"))
            logger.debug(f"Loaded {len(remote_files)} files")
            return remote_files
        except Exception as e:
            logger.error(f"Failed to load files: {str(e)}")

    def download_from_remote_subfolder(self, subfolder: str):
        # Download the files from the subfolder using pattern "{subfolder}/*"
        logger.info(f"Downloading from subfolder: {subfolder}")
        snapshot_download(
            repo_id=self.repo_id,
            allow_patterns=f"{subfolder}/*",
            local_dir=str(self.local_base_dir),
            repo_type="dataset",
        )
        logger.info(f"Downloaded files to {self.local_base_dir}")

        # unzip the downloaded files
        zip_files = list((self.local_base_dir / subfolder).glob("*.zip"))
        for zip_file in zip_files:
            logger.info(f"Unzipping {zip_file}")
            shutil.unpack_archive(zip_file, extract_dir=zip_file.parent)
            zip_file.unlink()
        logger.info(f'Subfolder "{subfolder}" saved to {self.local_base_dir}')


if __name__ == "__main__":
    setup_logging()
    logging.basicConfig(level=logging.INFO)

    local_dir = "/home/docker_dev/data"
    silvr_hf_repo_id = "yifutao/silvr_data"
    downloader = HFDatasetDownloader(silvr_hf_repo_id, local_dir)
    downloader.download_from_remote_subfolder("2023-02-24-bodleian")
