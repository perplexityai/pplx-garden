from functools import cache
from pathlib import Path


@cache
def count_sys_nvidia() -> int:
    return len(list(Path("/sys/bus/pci/drivers/nvidia/").glob("0000:*")))


@cache
def count_sys_infiniband_verbs() -> int:
    return len(list(Path("/sys/class/infiniband_verbs/").glob("uverbs*")))


def get_nets_per_gpu() -> int:
    return count_sys_infiniband_verbs() // count_sys_nvidia()
