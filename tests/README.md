# Test tiers

PPLX Garden tests are split by runtime requirements so default development does
not require CXI, RDMA, or multi-node hardware.

## Default tests

Run with:

```bash
pytest -m "not (cuda or distributed or fabric or kernel or perf)"
```

These tests should be CPU-only and cheap enough for ordinary CI.

## CUDA and fabric tests

Run CUDA/fabric functionality explicitly:

```bash
pytest -m "cuda and fabric and kernel"
```

These tests require a built `pplx_garden` extension, CUDA PyTorch, visible GPUs,
and a working libfabric provider.

## Distributed tests

Tests marked `distributed` launch multiple ranks or processes. Combine this
marker with the hardware markers needed by the specific test, for example:

```bash
pytest tests/p2p_all_to_all -m "distributed and cuda and fabric"
```

## Performance tests

Tests or scripts marked `perf` are benchmark/regression tools, not default
correctness tests. They should emit shape and timing metadata and should be run
only when explicitly selected.
