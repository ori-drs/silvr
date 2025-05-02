# SiLVR: Scalable Lidar-Visual Reconstruction with Neural Radiance Fields
SiLVR is a large-scale reconstruction system fusing both lidar and vision based on Neural Radiance Field.
- [Project Page](https://ori-drs.github.io/projects/silvr/)
- [ICRA Paper ](https://arxiv.org/abs/2403.06877)
- [Video](https://www.youtube.com/watch?v=kA11bdMbhMo)

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
```
python scripts/data_downloader.py
python scripts/train.py
```



## Contributing
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter. Install `pre-commit` will fix your code when you commit.
```
pre-commit install
```