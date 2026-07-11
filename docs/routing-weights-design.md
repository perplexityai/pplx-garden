# Optional routing weights in MoE dispatch (issue #7)

## Problem

`a2a_dispatch_send` extracts per-token routing weights but does not transmit them in the
send buffer. Downstream expert MLPs cannot apply `swiglu(y) * routing_weights` inside the
expert; weights are only applied at combine time.

## Proposed API

```python
def dispatch(
    self,
    out_expert_num_tokens: torch.Tensor,
    out_expert_x: torch.Tensor,
    out_expert_weights: Optional[torch.Tensor] = None,  # [num_recv_tokens, top-k]
    ...
) -> None:
```

Default `out_expert_weights=None` preserves current bandwidth behavior.

## Implementation sketch

1. **CUDA send path** — append `sizeof(float) * top_k` per routed token in the send buffer
   when `return_weights` flag is set on the kernel launch.
2. **CUDA recv path** — unpack weights into `out_expert_weights` in permuted recv order.
3. **Python bindings** — thread optional `out_expert_weights_ptr` through
   `dispatch_send` / `dispatch_recv` in `python-ext/src/py_p2p_all_to_all.rs`.
4. **Tests** — extend `tests/p2p_all_to_all/test_p2p_all_to_all.py` to assert weight parity
   against host reference for 2–4 GPU ranks (or CPU mock when CUDA unavailable).

## Bandwidth tradeoff

`+ top_k * 4 bytes` per dispatched token when enabled. For `top_k=8`, `hidden=4096`, this is
negligible relative to activation traffic but should remain opt-in.

## References

- GitHub issue: https://github.com/perplexityai/pplx-garden/issues/7
- Send kernel: `p2p-all-to-all/a2a-kernels/src/a2a/a2a_dispatch_send.cu`
