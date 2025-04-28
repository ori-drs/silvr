## Introduction
SiLVR (Scalable Lidar-Visual Reconstruction) is a large-scale reconstruction system fusing both lidar and vision based on Neural Radiance Field.

## Installation
### PyTorch
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
### tiny-cuda-nn
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Be careful if you have more than one GPUs. You might have to use `CUDA_VISIBLE_DEVICES=0`
### SiLVR
```
make
```

## Quick Start
```
python train.py
```

## Docker
```
make docker
```

## Contributing
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter. Install `pre-commit` will fix your code when you commit.
```
pre-commit install
```