# ruff: noqa: T201

import pickle

import torch

from pplx_garden.fabric_lib import TransferEngine
from tests.markers import gpu_only, mark_fabric


@mark_fabric
@gpu_only
def test_pickle_unpickle_descriptor() -> None:
    # Build a transfer engine.
    group = TransferEngine.detect_topology()[0]
    builder = TransferEngine.builder()
    builder.add_gpu_domains(
        group.cuda_device,
        group.domains,
        group.cpus[0],
        group.cpus[1],
    )
    engine = builder.build()

    # Allocate and register a CPU buffer.
    src_buf = torch.ones(
        (4096,),
        dtype=torch.uint8,
        device="cpu",
    )

    # Check pickle round-trip.
    _, descriptor = engine.register_tensor(src_buf)
    pickled = pickle.dumps(descriptor)
    unpickled = pickle.loads(pickled)
    assert descriptor.as_bytes() == unpickled.as_bytes()
