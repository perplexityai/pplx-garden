def _compact_offsets(tokens_per_expert: list[int], expert_padding: int) -> list[int]:
    offsets: list[int] = []
    base = 0
    for count in tokens_per_expert:
        offsets.append(base)
        base += ((count + expert_padding - 1) // expert_padding) * expert_padding
    return offsets


def _batched_offsets(num_local_experts: int, max_tokens_per_expert: int) -> list[int]:
    return [
        local_expert * max_tokens_per_expert
        for local_expert in range(num_local_experts)
    ]


def test_compact_offsets_respect_padding() -> None:
    assert _compact_offsets([3, 0, 5], 4) == [0, 4, 4]
    assert _compact_offsets([1, 4, 5], 4) == [0, 4, 8]


def test_batched_offsets_are_fixed_stride() -> None:
    assert _batched_offsets(3, 8) == [0, 8, 16]
    assert _batched_offsets(3, 128) == [0, 128, 256]
