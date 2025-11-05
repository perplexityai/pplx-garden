FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 as gdrcopy-builder

RUN apt-get update && apt-get install -y build-essential devscripts debhelper fakeroot pkg-config wget
RUN cd /tmp && \
    wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.1.tar.gz && \
    tar -xf v2.5.1.tar.gz && \
    cd gdrcopy-2.5.1/packages/ && \
    CUDA=/usr/local/cuda ./build-deb-packages.sh -t -k





FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 as final

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    patchelf \
    libclang-dev \
    clang-18 \
    clang-format-18 \
    git \
    build-essential \
    cmake \
    libssl-dev \
    wget \
    curl \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3-build \
    python3-venv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# PyTorch
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV TORCH_CUDA_ARCH_LIST="9.0a;10.0a+PTX"
RUN python3 -m pip install torch==2.9.0+cu129 --index-url https://download.pytorch.org/whl/cu129


# EFA (including libfabric and libibverbs)
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.44.0.tar.gz && \
    tar -xf aws-efa-installer-1.44.0.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer*
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent


# GDRCopy
COPY --from=gdrcopy-builder /tmp/gdrcopy-2.5.1/packages/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb /tmp/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb
RUN dpkg -i /tmp/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb && \
    rm -rf /tmp/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb


# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0 --component llvm-tools-preview
ENV PATH="/root/.cargo/bin:$PATH" \
    CARGO_HOME="/root/.cargo" \
    RUSTUP_HOME="/root/.rustup"


# Python dependencies
RUN python3 -m pip install numpy ninja maturin \
    pytest coverage mypy pylint ruff

