ARG PLATFORM=nvidia

# ----------------------------------------------------------------------------
# Choose base image based on the ${PLATFORM} variable
# ----------------------------------------------------------------------------

FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 as base-nvidia
FROM ubuntu:20.04 as base-cpu
FROM base-${PLATFORM} as base

# ----------------------------------------------------------------------------
# Common dependencies
# ----------------------------------------------------------------------------

ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    g++ \
    gcc \
    git \
    libboost-python-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    python3-pip \
    python3-venv \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=carla-pedestrians-client
ENV HOME /home/${USERNAME}

RUN groupadd -g ${GROUP_ID} ${USERNAME} \
    && useradd -ms /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && mkdir ${HOME}/.vscode-server ${HOME}/.vscode-server-insiders /outputs /venv /app \
    && chown ${USERNAME}:${USERNAME} ${HOME}/.vscode-server ${HOME}/.vscode-server-insiders /outputs /venv /app

# Everything else can be run as user since we need venv anyway
USER carla-pedestrians-client

# Create venv to allow editable installation of python packages
RUN python3 -m venv /venv

# Update basic python packages
RUN /venv/bin/python -m pip install --no-cache-dir -U \
    pip==21.3.1 \
    setuptools==60.1.0 \
    wheel==0.37.1

# Automatically activate virtualenv for user
RUN echo 'source /venv/bin/activate' >> ${HOME}/.bashrc

ENV torch_version=1.9.1
ENV torchvision_version=0.10.1
ENV pytorch3d_version=0.6.0

# ----------------------------------------------------------------------------
# NVIDIA-specific dependencies
# ----------------------------------------------------------------------------
FROM base as torch-nvidia

ENV PYOPENGL_PLATFORM=egl

RUN /venv/bin/python -m pip install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html \
    torch==${torch_version}+cu111 \
    torchvision==${torchvision_version}+cu111

# ----------------------------------------------------------------------------
# CPU-specific dependencies
# ----------------------------------------------------------------------------
FROM base as torch-cpu

ENV PYOPENGL_PLATFORM=osmesa

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    freeglut3 \
    freeglut3-dev \
    libgle3 \
    libgle3-dev \
    libosmesa6 \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*
USER carla-pedestrians-client

RUN /venv/bin/python -m pip install --no-cache-dir -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torch==${torch_version}+cpu \
    torchvision==${torchvision_version}+cpu

# -------------------------------------------------------------------------------------------------
# Common II
# -------------------------------------------------------------------------------------------------
FROM torch-${PLATFORM} as torch

# PyTorch3D (need to compile to get newer version, wheels are up to 0.3.0)
RUN /venv/bin/python -m pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@v${pytorch3d_version}"

# TODO: get rid of apt packages that are only needed to compile PyTorch3D someday

# separate some utility/development requirements, since they will change much slower than project ones
RUN /venv/bin/python -m pip install --no-cache-dir \
    autopep8 \
    pylint \
    pytest \
    pytest-cov \
    torch-tb-profiler

# Let's pretend we've installed CARLA via easy_install
# It's client for Python 3.7 and in Ubuntu 20.04 there's Python 3.8 but hopefully this will work
# TODO: update it to installable, official CARLA package once we make a switch to 0.9.13
COPY --from=carlasim/carla:0.9.11 --chown=${USERNAME}:${USERNAME} /home/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg /venv/lib/python3.8/site-packages/carla-0.9.11-py3.7-linux-x86_64.egg
RUN echo "import sys; sys.__plen = len(sys.path)\n./carla-0.9.11-py3.7-linux-x86_64.egg\nimport sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)" > /venv/lib/python3.8/site-packages/easy_install.pth

# Direct project dependencies are defined in pedestrians-video-2-carla/setup.cfg
# However, we want to leverage the cache, so we're going to specify at least basic ones with versions here
RUN /venv/bin/python -m pip install --no-cache-dir \
    av==8.0.3 \
    cameratransform==1.2 \
    dotmap==1.3.26 \
    einops==0.3.2 \
    gym==0.21.0 \
    h5pickle==0.4.2 \
    h5py==3.6.0 \
    matplotlib==3.5.0 \
    moviepy==1.0.3 \
    numpy==1.21.5 \
    opencv-python-headless==4.5.4.58 \
    pandas==1.3.5 \
    Pillow==8.4.0 \
    pims==0.5 \
    pyrender==0.1.45 \
    pytorch-lightning==1.5.2 \
    pyyaml==6.0 \
    randomname==0.1.5 \
    scikit-image==0.18.3 \
    scipy==1.7.2 \
    timm==0.4.12 \
    torchmetrics==0.6.0 \
    tqdm==4.62.3 \
    transforms3d==0.3.1 \
    trimesh==3.9.36 \
    wandb==0.12.9 \
    xmltodict==0.12.0 \
    git+https://github.com/nghorbani/human_body_prior.git@0278cb45180992e4d39ba1a11601f5ecc53ee148#egg=human-body-prior \
    git+https://github.com/nghorbani/body_visualizer@be9cf756f8d1daed870d4c7ad1aa5cc3478a546c#egg=body-visualizer \
    git+https://github.com/MPI-IS/configer.git@8cd1e3e556d9697298907800a743e120be57ac36#egg=configer \
    git+https://github.com/MPI-IS/mesh.git@49e70425cf373ec5269917012bda2944215c5ccd#egg=psbody-mesh

# install newer version of pyopengl, since pyrender has obsolete dependency
RUN /venv/bin/python -m pip install --no-cache-dir \
    PyOpenGL==3.1.5

# Copy client files so that we can do editable pip install
COPY --chown=${USERNAME}:${USERNAME} . /app

ARG COMMIT="0000000"
ENV COMMIT=${COMMIT}

ENTRYPOINT [ "/app/entrypoint.sh" ]

# Run infinite loop to allow easily attach to container
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]