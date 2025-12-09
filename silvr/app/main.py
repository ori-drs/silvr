import logging

from nerfstudio.scripts.train import entrypoint
from silvr.app.post_process import run_post_processing, run_post_processing_only
from silvr.submap.utils import create_submap_symlink, remove_submap_symlink

logger = logging.getLogger(__name__)


def run_silvr(config):
    config.set_args()
    logger.info(f"Start training SILVR on {config.base.data}")
    entrypoint()
    logger.info("Finish training SILVR")
    config.clean_argv()


def run_silvr_submap(config):
    """create symlinks from the submap_folder to data_main_folder.
    This is because nerfstudio expects the transforms.json to be in the same folder as the images.
    In our case, the submap jsons are not necessarily in the same folder as the images.
    Therefore, we create symlinks to the submap jsons in the data_main_folder, and remove them after training.
    Note that in rendering time, we will need to recreate the symlinks.

    Args:
    data_main_folder: the folder containing images, transforms.json
    submap_folder: the folder containing submaps pose as json files
    """
    remove_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)
    trajs_full_path = create_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)

    for i, traj in enumerate(trajs_full_path):
        logger.info(f"🚀 {i + 1}/{len(trajs_full_path)}: {traj} Starting batch")
        config.base.data = str(traj)
        run_silvr(config)
        run_post_processing(config)

    remove_submap_symlink(config.submap.submap_folder, config.submap.data_main_folder)


def silvr_main(config):
    if config.only_post_process.turn_on:
        run_post_processing_only(config)
    else:
        if not config.submap.run_submap:
            run_silvr(config)
            run_post_processing(config)
        else:
            run_silvr_submap(config)
