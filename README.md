# ⚪SiLVR: Scalable Lidar-Visual NeRF Reconstruction
<div align="center">
  <img src="./docs/rendering_roq.gif" alt="rendering gif">
</div>

- [Project Page](https://dynamic.robots.ox.ac.uk/projects/silvr/)
- [T-RO 25 Paper](https://arxiv.org/abs/2502.02657), [ICRA 24 Paper](https://arxiv.org/abs/2403.06877)
- [T-RO 25 Video](https://www.youtube.com/watch?si=yREPN_90_kxuvwxn&v=AU6gvRFfCMQ&feature=youtu.be), [ICRA 24 Video](https://www.youtube.com/watch?v=kA11bdMbhMo)
- [Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)

## News
(Dec 2025) SiLVR 2.0 Code release for the T-RO version with epistemic uncertainty estimation.

(Sep 2025) SiLVR with uncertainty estimation has been accepted by the [IEEE Transactions on Robotics (T-RO)](https://www.ieee-ras.org/publications/t-ro) and it will appear in the [Special Section on Visual SLAM](https://www.ieee-ras.org/publications/t-ro/special-issues/visual-slam). New code will be released also here.
## Setup
### Docker
Build a docker image to install dependencies and run SiLVR. 
```
docker compose -f .docker/docker_compose.yaml run --build silvr
```
Note: You can use your own `CUDA_ARCHITECTURES` in the Dockerfile to make the `tinycudann` build quicker.
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
### T-RO 25 Results
```
python scripts/data_downloader.py

python scripts/main.py --config config/2024-03-13-roq-01-unc.yaml

python scripts/main.py --config config/2024-bodleian-01+02-unc.yaml
```
### ICRA 24 Results
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
```bibtex
@article{tao2025silvr,
  title={SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification},
  author={Tao, Yifu and Fallon, Maurice},
  journal={IEEE Transactions on Robotics}, 
  year={2025},
}
```
