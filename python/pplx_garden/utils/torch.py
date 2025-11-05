import dataclasses
import gc
import os
from collections.abc import Callable, Iterator
from contextlib import ContextDecorator, contextmanager
from functools import wraps
from logging import Logger
from pathlib import Path
from typing import Any, Generic, Literal, Optional, ParamSpec, TypeVar

import torch
from torch.profiler import ProfilerActivity
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

T = TypeVar("T")
P = ParamSpec("P")
TModule = TypeVar("TModule", bound=torch.nn.Module)


def cleanup(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator which explicitly cleans up device memory after completion.
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        ret = fn(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        return ret

    return wrapper


@contextmanager
def cleanup_context() -> Iterator[None]:
    """Context manager to cleanup device memory before andafter completion."""
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


def has_cuda() -> bool:
    """Checks whether a CUDA device is available."""
    return torch.cuda.is_available()


def has_fp8() -> bool:
    """Checks whether the device has fp8 support."""

    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability(device=None)
    return major >= 9 or (major == 8 and minor >= 9)


def has_tp(tp: int) -> bool:
    """Checks whether the host has sufficient GPUs."""
    return has_cuda() and tp <= torch.cuda.device_count()


def has_triton_interpreter() -> bool:
    """Checks wheter triton is running in interpreter mode."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def add_nvtx(module: torch.nn.Module, name: str | None = None) -> None:
    """Add NVTX ranges around forward pass of a module and its children."""

    nvtx_name = (f"{name}:" if name else "") + module.__class__.__name__

    def push(module: torch.nn.Module, args: Any) -> None:
        torch.cuda.nvtx.range_push(nvtx_name)

    def pop(module: torch.nn.Module, args: Any, output: Any) -> None:
        torch.cuda.nvtx.range_pop()

    module.register_forward_pre_hook(push)
    module.register_forward_hook(pop)

    for name, child in module.named_children():
        add_nvtx(child, name)


def add_logger(
    module: torch.nn.Module,
    logger: Logger,
    negative_filters: Optional[list[str]] = None,
) -> None:
    """Log all arguments/results using prints through a model."""
    negative_filters = negative_filters or []

    def add_logger_rec(module: torch.nn.Module, names: list[str]) -> None:
        indent = " " * len(names) * 2
        path = ".".join(names)
        model_name = f"{module.__class__.__name__}@{path}"

        def push(module: torch.nn.Module, args: Any) -> None:
            logger.info("%s%s forward: %s", indent, model_name, args)

        def pop(module: torch.nn.Module, args: Any, output: Any) -> None:
            logger.info("%s%s result: %s", indent, model_name, output)

        matches_negative_filter = any(f in model_name for f in negative_filters)
        if not matches_negative_filter:
            module.register_forward_pre_hook(push)
            module.register_forward_hook(pop)

        module.register_forward_pre_hook(push)
        module.register_forward_hook(pop)

        for child_name, child in module.named_children():
            add_logger_rec(child, names + [child_name])

    add_logger_rec(module, [])


def capture_layer_io(
    module: torch.nn.Module, target_layers: list[str]
) -> tuple[dict[str, dict[str, Any]], list[RemovableHandle]]:
    """
    Capture inputs and outputs from the specified layers in a PyTorch model.
    Example:
        model = MyModel()
        target_layers = ["encoder.layer1", "encoder.layer2"]
        captured_io, hooks = capture_layer_io(model, target_layers)
        _ = model(torch.randn( ... ))
        for layer_name, io_dict in captured_io.items():
            print(layer_name, io_dict["input"][0].shape, io_dict["output"].shape)
    """
    captured_io: dict[str, dict[str, Any]] = {}
    hooks: list[RemovableHandle] = []

    def add_hook_rec(submodule: torch.nn.Module, names: list[str]) -> None:
        path = ".".join(names)
        model_name = f"{module.__class__.__name__}@{path}"

        def hook_fn(
            layer: torch.nn.Module, inputs: tuple[Any, ...], outputs: Any
        ) -> None:
            captured_io[model_name] = {"input": inputs, "output": outputs}

        if model_name in target_layers:
            handle = submodule.register_forward_hook(hook_fn)
            hooks.append(handle)

        for child_name, child_module in submodule.named_children():
            add_hook_rec(child_module, names + [child_name])

    add_hook_rec(module, [])
    return captured_io, hooks


@dataclasses.dataclass(frozen=True)
class CapturedLayer:
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]


@dataclasses.dataclass(frozen=True)
class CapturedActivations:
    captured_layers: dict[str, CapturedLayer]
    all_activated_layers: list[str]
    hooks: list[RemovableHandle]

    def cleanup(self) -> None:
        for hook in self.hooks:
            hook.remove()


def copy_layer_io(io: Any) -> list[torch.Tensor]:
    """Copy tensor data from layer inputs/outputs."""
    if isinstance(io, torch.Tensor):
        return [io.clone()]

    if isinstance(io, (tuple, list)):
        tensors = []
        for item in io:
            if isinstance(item, torch.Tensor):
                tensors.append(item.clone())
        return tensors

    return []


def capture_model_activations(
    module: torch.nn.Module,
    target_layers: list[str],
) -> CapturedActivations:
    """
    Capture inputs and outputs from the specified layers in a PyTorch model.
    Example:
        model = MyModel()
        target_layers = ["encoder.layer1", "encoder.layer2"]
        captured_io, hooks = capture_layer_io(model, target_layers)
        _ = model(torch.randn( ... ))
        for layer_name, io_dict in captured_io.items():
            print(layer_name, io_dict["input"][0].shape, io_dict["output"].shape)
    """
    ret = CapturedActivations(captured_layers={}, all_activated_layers=[], hooks=[])

    def add_hook_rec(submodule: torch.nn.Module, names: list[str]) -> None:
        path = ".".join(names)
        model_name = f"{module.__class__.__name__}@{path}"

        def hook_fn(
            layer: torch.nn.Module, inputs: tuple[Any, ...], outputs: Any
        ) -> None:
            ret.captured_layers[model_name] = CapturedLayer(
                inputs=copy_layer_io(inputs),
                outputs=copy_layer_io(outputs),
            )

        ret.all_activated_layers.append(model_name)
        if model_name in target_layers:
            handle = submodule.register_forward_hook(hook_fn)
            ret.hooks.append(handle)

        for child_name, child_module in submodule.named_children():
            add_hook_rec(child_module, names + [child_name])

    add_hook_rec(module, [])
    return ret


def format_activations_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
    layer_name: str,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> str:
    RED = "\033[91m"
    RESET = "\033[0m"

    actual = actual.squeeze()
    expected = expected.squeeze()
    eps = 1e-12

    if actual.shape != expected.shape:
        return f"{RED}{layer_name} SHAPE MISMATCH{RESET} - Actual: {actual.shape}, Expected: {expected.shape}"
    if actual.dtype != expected.dtype:
        return f"{RED}{layer_name} DTYPE MISMATCH{RESET} - Actual: {actual.dtype}, Expected: {expected.dtype}"
    if actual.device != expected.device:
        expected = expected.to(actual.device)
    actual = actual.float()
    expected = expected.float()

    diff = actual - expected
    abs_diff = torch.abs(diff)
    l2_norm = torch.norm(diff).item()
    ref_l2_norm = torch.norm(expected).item()
    relative_l2 = l2_norm / (ref_l2_norm + eps)
    cosine_sim = torch.nn.functional.cosine_similarity(
        actual.flatten().unsqueeze(0), expected.flatten().unsqueeze(0)
    ).item()

    total_elements = actual.numel()
    atol_mismatches = (abs_diff > atol).sum().item()
    rtol_mismatches = ((abs_diff / (torch.abs(expected) + eps)) > rtol).sum().item()
    avg_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    std_diff = abs_diff.std().item()
    snr = ref_l2_norm / (l2_norm + eps)

    atol_pct = atol_mismatches * 100 / total_elements
    rtol_pct = rtol_mismatches * 100 / total_elements
    atol_color = RED if atol_pct > 0 else ""
    rtol_color = RED if rtol_pct > 0 else ""
    atol_reset = RESET if atol_pct > 0 else ""
    rtol_reset = RESET if rtol_pct > 0 else ""

    return (
        f"{layer_name} "
        f"L2 Abs.Diff: {l2_norm:.6f} L2 Rel.Diff: {relative_l2:.6f} "
        f"COS: {cosine_sim:.4f} SNR: {snr:.2f} "
        f"ATOL: {atol_color}{atol_pct:.1f}%{atol_reset} RTOL: {rtol_color}{rtol_pct:.1f}%{rtol_reset} "
        f"Avg: {avg_diff:.6f} Max: {max_diff:.6f} Std: {std_diff:.6f}"
    )


@contextmanager
def profile_range(name: str) -> Iterator[None]:
    """Helper to mark functions with both NVTX and torch ranges."""

    with torch.cuda.nvtx.range(name):
        with torch.profiler.record_function(name):
            yield


def str_to_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)


def type_to_str(ty: torch.dtype) -> str:
    match ty:
        case torch.int8:
            return "INT8"
        case torch.float8_e4m3fn:
            return "FP8"
        case torch.bfloat16:
            return "BFP16"
        case torch.float16:
            return "FP16"
        case torch.float32:
            return "FP32"
        case _:
            raise ValueError("Unsupported dtype")


class stream_context(ContextDecorator):
    """A wrapper around streams with support for both CPUs and GPUS.

    In CPU-only tests, no streams are needed and the context becomes a no-op.
    Otherwise, the context wraps around a stream and waits for it to finish.
    """

    def __init__(self, device: Optional[torch.device]) -> None:
        self._stream: Optional[torch.cuda.Stream]
        if device is not None and device.type != "cpu":
            self._stream = torch.cuda.Stream()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self._stream = None

        self._wrapper: Optional[Any] = None

    def __enter__(self) -> Self:
        if self._stream is not None:
            self._wrapper = torch.cuda.stream(self._stream)
            self._wrapper.__enter__()  # type: ignore
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        if self._stream is not None:
            assert self._wrapper is not None
            self._wrapper.__exit__(*exc)
            torch.cuda.current_stream().wait_stream(self._stream)
        return False


@contextmanager
def capture_profile(trace_file: Optional[Path]) -> Iterator[None]:
    if trace_file is None:
        yield
        return

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        yield

    prof.export_chrome_trace(str(trace_file))


def get_mem_info(device: torch.device) -> tuple[int, int]:
    """Returns the available and total memory on the device."""

    gc.collect()

    if device.type == "cpu":
        return 0, 0

    torch.cuda.empty_cache()
    mem_avail, mem_total = torch.cuda.mem_get_info(device)
    return mem_avail, mem_total


def get_available_mem(device: torch.device) -> int:
    """Returns the total memory available on the device."""
    gc.collect()
    torch.cuda.empty_cache()
    mem_avail, _ = get_mem_info(device)
    return mem_avail


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


@contextmanager
def set_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class BorrowedModule(Generic[TModule]):
    """
    A simple wrapper around a nn.Module.
    1. Make it clear that the module is non-owning.
    2. Avoids the module being registered as a submodule (not in named_children()).
    """

    def __init__(self, module: TModule) -> None:
        self.module = module
