pplx-garden
===========

Perplexity AI open source garden for inference technology

## Research Paper

[RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656)

## P2P MoE dispatch/combine kernel

* Support both NVIDIA ConnectX-7 and AWS EFA (potentially other RDMA NICs as wel)
* Use NVLink for intra-node data transfer and RDMA for inter-node
* Optimize for decode, while also support prefill
* Split send and recv stages for both dispatch and combine, allow micro-batching
* SM-free while RDMA transfer
* Support CUDA Graph

## RDMA TransferEngine library

* Support both NVIDIA ConnectX-7 and AWS EFA (potentially other RDMA NICs as well)
* Support aggregation of multiple NICs per GPU
* Support reliable unordered transport protocol

# System requirements

* Linux Kernel 5.15 or higher
* (Recommended) Linux Kernel 6.1 or higher (for DMA-BUF support)
* CUDA 12.8 or higher
* libfabric
* libibverbs
* GDRCopy
* `SYS_PTRACE` and `SYS_ADMIN` Linux capabilities for `pidfd_getfd`. You can obtain these by running as root, with sudo, or inside docker with `--cap-add=SYS_PTRACE --cap-add=SYS_ADMIN`.
* RDMA network with GPUDirect RDMA support

# Docker dev image

We provide a docker image for the convenience of development. You can build it with the following command:

```bash
docker build -t pplx-garden-dev - < docker/dev.Dockerfile
```

Run the container with the following command:

```bash
./scripts/run-docker.sh
```

# Run fabric-debug

This is the benchmark for our network library.

Build the benchmark binary:

```bash
cd /app
cargo build --release --bin fabric-debug
```

Usage:

* Server: `fabric-debug [GPUs separated by comma] [NICs per GPU]`
* Client: `fabric-debug [GPUs separated by comma] [NICs per GPU] [server address]` where the server address is the one printed by the server.


Example:

```
server$ /app/target/release/fabric-debug 0,1,2,3,4,5,6,7 2
client$ /app/target/release/fabric-debug 0,1,2,3,4,5,6,7 2 fe80xxxx
```

# Build and Install Python Wheel

```bash
cd /app
export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
python3 -m build --wheel
python3 -m pip install /app/dist/*.whl
```

# Run All-to-All Benchmark

```bash
# Environment variables
NUM_NODES=...
NODE_RANK=...  # [0, NUM_NODES)
MASTER_IP=...

# Run on all nodes
cd /app
python3 -m benchmarks.bench_all_to_all \
    --world-size $((NUM_NODES * 8)) --nets-per-gpu 2 --init-method=tcp://$MASTER_IP:29500 \
    --node-rank=$NODE_RANK --nvlink=8
```

Note:

* Remove `--nvlink` flag if you want to use RDMA only.
* Set `--nets-per-gpu` accordingly based on the VM instance type.

# All-to-All Performance Results

Decode (128 tokens) Dispatch and Combine:

|      | pplx-EFA | pplx-CX7 | DeepEP-CX7 | x | pplx-EFA | pplx-CX7 | DeepEP-CX7 |
|------|---------:|---------:|-----------:|---|---------:|---------:|-----------:|
| EP64 | 266.7 μs | 187.5 μs |   177.9 μs | x | 391.2 μs | 309.1 μs |   325.0 μs |
| EP32 | 229.1 μs | 153.9 μs |   159.1 μs | x | 335.0 μs | 266.3 μs |   285.0 μs |
| EP16 | 214.8 μs | 110.2 μs |   123.9 μs | x | 241.5 μs | 185.5 μs |   203.0 μs |
| EP8  |  49.7 μs |  50.5 μs |    42.6 μs | x |  64.2 μs |  65.3 μs |    72.0 μs |


Prefill (4096 tokens) Dispatch and Combine:

| x    |  pplx-EFA |  pplx-CX7 | DeepEP-CX7 | x |  pplx-EFA |  pplx-CX7 | DeepEP-CX7 |
|------|----------:|----------:|-----------:|---|----------:|----------:|-----------:|
| EP64 | 5334.3 μs | 4665.2 μs |  5071.6 μs | x | 9779.3 μs | 8771.1 μs |  5922.7 μs |
| EP32 | 4619.0 μs | 4011.8 μs |  3680.2 μs | x | 8271.5 μs | 7526.8 μs |  3565.4 μs |
| EP16 | 3196.7 μs | 2734.8 μs |  2481.9 μs | x | 5379.1 μs | 1062.2 μs |  1863.9 μs |
| EP8  | 1052.4 μs | 5071.1 μs |  1810.3 μs | x | 1396.7 μs | 1405.1 μs |   962.9 μs |


# Directory Structure

* `fabric-lib/`: RDMA TransferEngine library
* `p2p-all-to-all/`: P2P MoE All-to-All implementation
* `python-ext/`: Python extension module from Rust code
* `python/pplx_garden/`: Python code for the `pplx_garden` package
* `rust/`: Rust utility libraries

# Acknowledgments

Our RDMA library is inspired by [MoonCake](https://www.usenix.org/conference/fast25/presentation/qin).
Our MoE kernel is inspired by [DeepEP](https://github.com/deepseek-ai/DeepEP).

# Citation

If you find this work useful, please cite:

```
@misc{pplx-rdma-p2p,
      title={RDMA Point-to-Point Communication for LLM Systems}, 
      author={Nandor Licker and Kevin Hu and Vladimir Zaytsev and Lequn Chen},
      year={2025},
      eprint={2510.27656},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2510.27656}, 
}
```
