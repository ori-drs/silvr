# SiLVR: Scalable Lidar-Visual NeRF Reconstruction
<div align="center">
  <img src="./docs/rendering_roq.gif" alt="rendering gif">
</div>

- [Project Page](https://ori-drs.github.io/projects/silvr/)
- [Paper](https://arxiv.org/abs/2403.06877)
- [Video](https://www.youtube.com/watch?v=kA11bdMbhMo)
- [Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)

## News
(Jan 2025) An updated version with uncertainty estimation will be released. Meanwhile, you can access the new [paper](https://arxiv.org/abs/2502.02657) and the [video](https://www.youtube.com/watch?si=yREPN_90_kxuvwxn&v=AU6gvRFfCMQ&feature=youtu.be).
## Setup
### Docker
Build a docker image to install dependencies and run SiLVR. 
```
docker compose -f .docker/docker_compose.yaml run --build silvr
```
### Manual Installation
You can also install SiLVR to your system manually.
```
# install Pytorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install Tiny-CUDA-NN
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

# clone the repo
git clone https://github.com/ori-drs/silvr.git
cd silvr
pip install -e .
```


## Running
Download sample data from [Hugging face](https://huggingface.co/datasets/ori-drs/silvr_data/tree/main), setup the [config file](./scripts/config_train.yaml), and then run the training script.
```bash
python scripts/data_downloader.py
python scripts/train.py --config config/2024-03-13-roq-01.yaml
```

To reproduce the [demo video](https://www.youtube.com/watch?v=9lYauDc5Df4&list=TLGGIUf8rfY28WUwNDA1MjAyNQ) which involves longer training on a bigger dataset, run the following training config.
```bash
python scripts/train.py --config config/2023-09-02-roq-hbac.yaml
```


## Contributing
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter. Install `pre-commit` will fix your code when you commit.
```
pre-commit install
```

## Citation
If you found this software package useful, please consider citing our paper as 
```
@inproceedings{tao2024silvr,
  title = {SiLVR: Scalable Lidar-Visual Reconstruction with Neural Radiance Fields for Robotic Inspection},
  author = {Tao, Yifu and Bhalgat, Yash and Fu, Lanke Frank Tarimo and Mattamala, Matias and Chebrolu, Nived and Fallon, Maurice},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2024},
}
```
