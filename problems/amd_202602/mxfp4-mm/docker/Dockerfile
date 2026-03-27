FROM ghcr.io/actions/actions-runner:latest

ENV CXX=clang++
ARG GPU_ARCH="gfx950"
ENV GPU_ARCH_LIST=$GPU_ARCH

RUN sudo apt-get update -y \
&& sudo apt-get install -y --no-install-recommends \
software-properties-common curl ca-certificates git jq sudo unzip zip \
cmake ninja-build clang lld wget psmisc \
python3-venv python3-pip python3-setuptools python3-wheel python3-dev \
&& sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt update -y \
&& sudo usermod -a -G render,video runner \
&& wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb \
&& sudo apt install -y ./amdgpu-install_7.1.70100-1_all.deb \
&& sudo apt update -y \
&& sudo apt install -y rocm

ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH="/opt/rocm/lib"

RUN sudo pip install --break-system-packages --no-cache-dir \
torch==2.10.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1

RUN sudo pip install --break-system-packages ninja numpy packaging wheel
