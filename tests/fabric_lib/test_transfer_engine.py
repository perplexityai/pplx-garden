# ruff: noqa: T201

import dataclasses
import pickle
import queue
import signal
import threading
from typing import Any, assert_never

import pytest
import torch
import torch.multiprocessing as mp
import triton  # type: ignore
import triton.language as tl  # type: ignore

from pplx_garden.fabric_lib import (
    DomainAddress,
    MemoryRegionDescriptor,
    MemoryRegionHandle,
    PageIndices,
    TransferEngine,
)
from pplx_garden.utils import logging_utils
from tests.fabric import get_nets_per_gpu
from tests.markers import gpu_only, mark_ci_4gpu, mark_fabric

logger = logging_utils.get_logger(__name__)

MESSAGE_BUF_SIZE = 64 << 20
CUDA_BUF_SIZE = 256 << 20


@dataclasses.dataclass(slots=True)
class Paged:
    seed: int
    mr_desc: list[MemoryRegionDescriptor]
    offset: int
    indices: list[int]
    len: int
    stride: int
    imm: int | None = None


@dataclasses.dataclass(slots=True)
class Single:
    seed: int
    mr_desc: MemoryRegionDescriptor
    offset: int
    len: int


@dataclasses.dataclass(slots=True)
class Imm:
    addr: DomainAddress
    mr_desc: MemoryRegionDescriptor
    imm: int


@dataclasses.dataclass(slots=True)
class Request:
    addr: DomainAddress
    content: Paged | Single | Imm


def build_engine(selected_gpus: list[int], nets_per_gpu: int) -> TransferEngine:
    system_topo = TransferEngine.detect_topology()
    builder = TransferEngine.builder()

    for group in system_topo:
        if group.cuda_device not in selected_gpus:
            continue
        worker_cpu = group.cpus[0]
        uvm_cpu = group.cpus[1]
        builder.add_gpu_domains(
            group.cuda_device,
            group.domains,
            worker_cpu,
            uvm_cpu,
        )
        logger.info(
            "Registered CUDA device %d, CPU #%d, UVM CPU #%d",
            group.cuda_device,
            worker_cpu,
            uvm_cpu,
        )

    return builder.build()


@dataclasses.dataclass(slots=True)
class CudaResource:
    cuda_device: int
    cuda_buf: torch.Tensor
    cuda_mr_handle: MemoryRegionHandle
    cuda_mr_desc: MemoryRegionDescriptor


def alloc_and_register_memory(
    selected_gpus: list[int],
    engine: TransferEngine,
) -> list[CudaResource]:
    cuda_res = []
    for cuda_device in selected_gpus:
        cuda_buf = torch.empty(
            CUDA_BUF_SIZE,
            dtype=torch.uint8,
            device=f"cuda:{cuda_device}",
        )

        cuda_mr_handle, cuda_mr_desc = engine.register_tensor(cuda_buf)
        cuda_res.append(
            CudaResource(cuda_device, cuda_buf, cuda_mr_handle, cuda_mr_desc)
        )
    return cuda_res


def fill_paged(
    data: torch.Tensor,
    seed: int,
    offset: int,
    indices: list[int],
    page_len: int,
    stride: int,
) -> None:
    rng = torch.Generator("cpu")
    rng.manual_seed(seed)
    for page in indices:
        off = offset + page * stride
        page_slice = data[off : off + page_len]
        torch.randint(
            0,
            256,
            (page_len,),
            dtype=torch.uint8,
            generator=rng,
            out=page_slice,
        )


def generate_random_paged_data(request: Paged, seed: int, rank: int) -> torch.Tensor:
    data = torch.zeros(CUDA_BUF_SIZE, dtype=torch.uint8, device="cpu")
    fill_paged(
        data,
        seed << 4 | rank,
        request.offset,
        request.indices,
        request.len,
        request.stride,
    )
    return data


def generate_random_bytes(seed: int, size: int) -> torch.Tensor:
    rng = torch.Generator("cpu")
    rng.manual_seed(seed)
    return torch.randint(0, 256, (size,), dtype=torch.uint8, generator=rng)


def on_error_panic(error: str) -> None:
    raise RuntimeError("fabric-lib error:" + error)


class Counter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.count = 0

    def inc(self) -> None:
        with self.lock:
            self.count += 1


class CounterDict:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: dict[int, int] = {}

    def inc(self, imm: int) -> None:
        with self.lock:
            self.counts[imm] = self.counts.get(imm, 0) + 1


def _test_paged_write_server(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    try:
        # Build the transfer engine.
        engine = build_engine(selected_gpus, nets_per_gpu)

        # Send over the server address.
        logger.info("Server address: %s", engine.main_address)
        conn.put(engine.main_address)
    finally:
        conn.close()

    # Register memory
    cuda_res = alloc_and_register_memory(selected_gpus, engine)

    # Submit bouncing recvs
    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    # Setup signal handler
    stop_server = False

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop_server
        stop_server = True

    signal.signal(signal.SIGHUP, handle_signal)

    # Main loop
    logger.info("Waiting for one client request")
    msg = recv_queue.get()

    request = pickle.loads(msg)  # pyright: ignore[reportPossiblyUnboundVariable]
    assert isinstance(request, Request)

    cond = threading.Condition()
    cnt_completions = 0

    def transfer_callback() -> None:
        nonlocal cnt_completions
        with cond:
            cnt_completions += 1
            cond.notify_all()

    match request.content:
        case Paged() as paged:
            logger.info(
                "Paged: page_bytes=%d, num_pages=%d",
                paged.len,
                len(paged.indices),
            )

            for i, res in enumerate(cuda_res):
                tmp = generate_random_paged_data(paged, paged.seed, i)
                res.cuda_buf.copy_(tmp)

            # Generate transfer requests
            page_indices = PageIndices(paged.indices)

            total_completions = len(cuda_res)
            for i, res in enumerate(cuda_res):
                engine.submit_paged_writes(
                    paged.len,
                    res.cuda_mr_handle,
                    page_indices,
                    paged.stride,
                    paged.offset,
                    paged.mr_desc[i],
                    page_indices,
                    paged.stride,
                    paged.offset,
                    paged.imm,
                    transfer_callback,
                    on_error_panic,
                )

        case Single() as single:
            logger.info("Single: bytes=%d", single.len)
            res = cuda_res[0]

            total_completions = 1

            # Generate random data if client wants to verify
            res.cuda_buf[single.offset : single.offset + single.len].copy_(
                generate_random_bytes(single.seed - 1, single.len)
            )

            # Generate transfer requests
            engine.submit_write(
                res.cuda_mr_handle,
                single.offset,
                single.len,
                None,
                single.mr_desc,
                single.offset,
                transfer_callback,
                on_error_panic,
            )

        case Imm() as imm:
            logger.info("Imm: imm=%d", imm.imm)

            total_completions = 1

            engine.submit_imm(
                imm.imm,
                imm.mr_desc,
                transfer_callback,
                on_error_panic,
            )

        case _:
            assert_never(request.content)

    with cond:
        while cnt_completions < total_completions:
            cond.wait()

    send_done = threading.Event()
    engine.submit_send(
        request.addr,
        pickle.dumps(None),
        send_done.set,
        on_error_panic,
    )
    send_done.wait()


def generate_paged_write_request(
    cuda_res: list[CudaResource],
    page_bytes: int,
    num_pages: int,
) -> Paged:
    stride = page_bytes + 128
    offset = 1024
    max_pages = (CUDA_BUF_SIZE - offset) // stride
    seed = 0xABCDABCD987

    # Generate page_indices
    rng = torch.Generator("cpu")
    rng.manual_seed(0x123)
    page_indices = torch.randint(
        0, max_pages, (num_pages,), generator=rng, dtype=torch.int32
    ).tolist()

    return Paged(
        seed=seed,
        mr_desc=[res.cuda_mr_desc for res in cuda_res],
        offset=offset,
        indices=page_indices,
        len=page_bytes,
        stride=stride,
    )


def _test_paged_write_client(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    server_address = conn.get()
    logger.info("Received server address %s", server_address)

    engine = build_engine(selected_gpus, nets_per_gpu)

    cuda_res = alloc_and_register_memory(selected_gpus, engine)

    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    imm_queue: queue.Queue[int] = queue.Queue()
    engine.set_imm_callback(imm_queue.put)

    page_bytes = 128
    num_pages = 2

    # Send request
    content = generate_paged_write_request(cuda_res, page_bytes, num_pages)
    request = Request(
        addr=engine.main_address,
        content=content,
    )
    data = pickle.dumps(request)
    send_done = threading.Event()
    engine.submit_send(server_address, data, send_done.set, on_error_panic)

    # Wait for SEND and RECVcompletion
    send_done.wait()
    recv_queue.get()

    # Verify data
    for i, res in enumerate(cuda_res):
        gold = generate_random_paged_data(content, content.seed, i)
        buf = res.cuda_buf.to("cpu")
        for page_idx in content.indices:
            offset = content.offset + page_idx * content.stride
            page_gold = gold[offset : offset + content.len]
            page_buf = buf[offset : offset + content.len]
            assert torch.equal(page_gold, page_buf)


@pytest.fixture
def nets_per_gpu() -> int:
    return get_nets_per_gpu()


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_paged_write(nets_per_gpu: int) -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_paged_write_server,
        args=(queue, [2, 3], nets_per_gpu),
    )
    server.start()

    client = ctx.Process(
        target=_test_paged_write_client,
        args=(queue, [0, 1], nets_per_gpu),
    )
    client.start()
    client.join()
    assert client.exitcode == 0

    server.join()
    assert server.exitcode == 0


def _test_single_write_client(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    server_address = conn.get()
    logger.info("Received server address %s", server_address)

    engine = build_engine(selected_gpus, nets_per_gpu)

    cuda_res = alloc_and_register_memory(selected_gpus, engine)
    res = cuda_res[0]

    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    # Send request
    content = Single(
        seed=0xABCDABCD987,
        mr_desc=res.cuda_mr_desc,
        offset=1024,
        len=1000,
    )
    request = Request(
        addr=engine.main_address,
        content=content,
    )
    data = pickle.dumps(request)
    send_done = threading.Event()
    engine.submit_send(server_address, data, send_done.set, on_error_panic)

    # Wait for SEND and RECVcompletion
    send_done.wait()
    recv_queue.get()

    # Verify data
    gold = generate_random_bytes(content.seed - 1, content.len)
    buf = res.cuda_buf[content.offset : content.offset + content.len].to("cpu")

    assert torch.equal(gold, buf)


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_single_write(nets_per_gpu: int) -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_paged_write_server,
        args=(queue, [2, 3], nets_per_gpu),
    )
    server.start()

    client = ctx.Process(
        target=_test_single_write_client,
        args=(queue, [0, 1], nets_per_gpu),
    )
    client.start()
    client.join()
    assert client.exitcode == 0

    server.join()
    assert server.exitcode == 0


def _test_imm_client(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    server_address = conn.get()
    logger.info("Received server address %s", server_address)

    engine = build_engine(selected_gpus, nets_per_gpu)

    cuda_res = alloc_and_register_memory(selected_gpus, engine)

    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    imm_queue: queue.Queue[int] = queue.Queue()
    engine.set_imm_callback(imm_queue.put)

    # Prepare IMM request.
    content = Imm(
        addr=engine.main_address,
        mr_desc=cuda_res[0].cuda_mr_desc,
        imm=0xDEADBEEF,
    )
    request = Request(addr=engine.main_address, content=content)

    # Send request and wait for completion.
    data = pickle.dumps(request)
    send_done = threading.Event()
    engine.submit_send(server_address, data, send_done.set, on_error_panic)
    send_done.wait()

    # Wait for IMM
    imm_data = imm_queue.get()
    assert imm_data == content.imm

    # Wait for RECV completion
    recv_queue.get()


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_imm(nets_per_gpu: int) -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_paged_write_server,
        args=(queue, [2, 3], nets_per_gpu),
    )
    server.start()

    client = ctx.Process(
        target=_test_imm_client,
        args=(queue, [0, 1], nets_per_gpu),
    )
    client.start()
    client.join()
    assert client.exitcode == 0

    server.join()
    assert server.exitcode == 0


# ruff: noqa: ANN001
@triton.jit
def _inc_u64_kernel(ptr) -> None:
    ptr = tl.load(ptr).to(tl.pointer_type(tl.uint64))
    tl.store(ptr, tl.load(ptr) + 1)


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_uvm_watcher(nets_per_gpu: int) -> None:
    """Tests that the UVM watcher notices counter changes."""

    engine = build_engine([0, 1, 2, 3], nets_per_gpu)

    watcher_ptr = torch.empty(1, dtype=torch.uint64, device="cuda")
    watcher_done = threading.Event()

    # Allocate a watcher.
    def callback(old_value: int, new_value: int) -> bool:
        watcher_done.set()
        return True

    watcher_ptr[0] = engine.alloc_scalar_watcher(callback).ptr

    # Increment the counter.
    _inc_u64_kernel[(1,)](watcher_ptr)

    # Blocks until engine detects the counter has been incremented.
    watcher_done.wait()


def _test_single_write_cpu_send(queue: mp.Queue) -> None:
    try:
        dst_mr_desc = queue.get()

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

        # Allocate and register two CPU buffers.
        src_buf = torch.ones(
            (CUDA_BUF_SIZE,),
            dtype=torch.uint8,
            device="cpu",
        )
        src_mr_handle, _ = engine.register_tensor(src_buf)

        send_cond = threading.Condition()

        def on_write_complete() -> None:
            with send_cond:
                send_cond.notify_all()

        engine.submit_write(
            src_mr=src_mr_handle,
            offset=0,
            length=1024,
            imm_data=555,
            dst_mr=dst_mr_desc,
            dst_offset=0,
            on_done=on_write_complete,
            on_error=on_error_panic,
            num_shards=None,
        )

        # Wait for the send to complete.
        with send_cond:
            send_cond.wait()
    except:
        logger.exception("Failed to send CPU write")
        raise


def _test_single_write_cpu_recv(queue: mp.Queue) -> None:
    # Build a transfer engine.
    try:
        group = TransferEngine.detect_topology()[0]
        builder = TransferEngine.builder()
        builder.add_gpu_domains(
            group.cuda_device,
            group.domains,
            group.cpus[0],
            group.cpus[1],
        )
        engine = builder.build()

        def on_imm(imm: int) -> None:
            assert imm == 555
            with recv_cond:
                recv_cond.notify_all()

        engine.set_imm_callback(on_imm)

        dst_buf = torch.zeros(
            (CUDA_BUF_SIZE,),
            dtype=torch.uint8,
            device="cpu",
        )
        _, dst_mr_desc = engine.register_tensor(dst_buf)

        queue.put(dst_mr_desc)

        recv_cond = threading.Condition()

        # Wait for the packet to be received.
        with recv_cond:
            recv_cond.wait()

        assert torch.all(dst_buf[:1024] == 1)
    except:
        logger.exception("Failed to receive CPU write")
        raise


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_single_write_cpu_tensor() -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_single_write_cpu_send,
        args=(queue,),
    )
    server.start()

    client = ctx.Process(
        target=_test_single_write_cpu_recv,
        args=(queue,),
    )
    client.start()

    client.join()
    assert client.exitcode == 0
    server.join()
    assert server.exitcode == 0


def _test_shard_single_write_at_the_end_of_mr_client(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    server_address = conn.get()

    engine = build_engine(selected_gpus, nets_per_gpu)

    cuda_res = alloc_and_register_memory(selected_gpus, engine)
    res = cuda_res[0]

    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    imm_queue: queue.Queue[int] = queue.Queue()
    engine.set_imm_callback(imm_queue.put)

    # Small single WRITE at the end of the MR
    size = 128
    content = Single(
        seed=0xABCDABCD987,
        mr_desc=res.cuda_mr_desc,
        offset=res.cuda_buf.nelement() - size,
        len=size,
    )
    request = Request(addr=engine.main_address, content=content)

    # Send request and wait for completion.
    data = pickle.dumps(request)
    send_done = threading.Event()
    engine.submit_send(server_address, data, send_done.set, on_error_panic)
    send_done.wait()

    # Wait for RECV completion
    recv_queue.get()


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_shard_single_write_at_the_end_of_mr(nets_per_gpu: int) -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_paged_write_server,
        args=(queue, [2], nets_per_gpu),
    )
    server.start()

    client = ctx.Process(
        target=_test_shard_single_write_at_the_end_of_mr_client,
        args=(queue, [0], nets_per_gpu),
    )
    client.start()
    client.join()
    assert client.exitcode == 0

    server.join()
    assert server.exitcode == 0


def _test_imm_count_client(
    conn: mp.Queue,
    selected_gpus: list[int],
    nets_per_gpu: int,
) -> None:
    server_address = conn.get()
    logger.info("Received server address %s", server_address)

    engine = build_engine(selected_gpus, nets_per_gpu)

    cuda_res = alloc_and_register_memory(selected_gpus, engine)

    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        1,
        MESSAGE_BUF_SIZE,
        recv_queue.put,
        on_error_panic,
    )

    num_expected = 10
    num_extra = 7
    imm = 0xDEADBEEF
    imm_data = CounterDict()
    engine.set_imm_callback(imm_data.inc)
    imm_count_reached = Counter()
    engine.set_imm_count_expected(
        imm, num_expected * len(cuda_res), imm_count_reached.inc
    )

    # Prepare Paged request.
    content = Paged(
        seed=0xABCDABCD987,
        mr_desc=[res.cuda_mr_desc for res in cuda_res],
        offset=0,
        indices=list(range(num_expected + num_extra)),
        len=0,
        stride=0,
        imm=imm,
    )
    request = Request(addr=engine.main_address, content=content)

    # Send request and wait for completion.
    data = pickle.dumps(request)
    send_done = threading.Event()
    engine.submit_send(server_address, data, send_done.set, on_error_panic)
    send_done.wait()

    # Wait for RECV completion
    recv_queue.get()

    # Check ImmCount
    assert imm_count_reached.count == 1

    # Check ImmData
    assert imm_data.counts == {imm: num_extra * len(cuda_res)}


@mark_fabric
@gpu_only
@mark_ci_4gpu
def test_imm_count(nets_per_gpu: int) -> None:
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()
    server = ctx.Process(
        target=_test_paged_write_server,
        args=(queue, [2, 3], nets_per_gpu),
    )
    server.start()

    client = ctx.Process(
        target=_test_imm_count_client,
        args=(queue, [0, 1], nets_per_gpu),
    )
    client.start()
    client.join()
    assert client.exitcode == 0

    server.join()
    assert server.exitcode == 0
