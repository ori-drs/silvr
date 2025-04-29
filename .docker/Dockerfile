ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11

RUN python -m pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
# ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
ARG CUDA_ARCHITECTURES=90;89;86;80;75
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

ARG GID
ARG UID
ENV UNAME=docker_dev
RUN addgroup --gid $GID $UNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME

ARG SILVR_DIR=/home/docker_dev/silvr
WORKDIR ${SILVR_DIR}

COPY ./requirements.txt ${SILVR_DIR}/requirements.txt
RUN pip install -r requirements.txt
COPY ./silvr/ ${SILVR_DIR}/silvr
COPY ./pyproject.toml ${SILVR_DIR}/pyproject.toml
COPY ./scripts/ ${SILVR_DIR}/scripts
RUN pip install -e .
# Make the outputs of the container match the host

RUN chown -R ${UID}:${GID} ${SILVR_DIR}/*
USER ${UNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc

CMD ["/bin/bash"]
