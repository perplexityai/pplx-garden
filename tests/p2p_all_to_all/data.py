from dataclasses import dataclass
from typing import Optional

import torch


def rand_topk_idx(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    scores = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scores = scores.abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    return topk_idx.to(torch.uint32)


@dataclass
class RankTestData:
    indices: torch.Tensor
    weights: torch.Tensor
    dp_x: torch.Tensor
    dp_x_scale: Optional[torch.Tensor]
    bound_m: Optional[torch.Tensor]
    expected_num_tokens: torch.Tensor

    @classmethod
    def rand_indices_and_count(
        cls,
        num_experts: int,
        num_experts_per_token: int,
        max_num_tokens: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = rand_topk_idx(
            max_num_tokens,
            num_experts,
            num_experts_per_token,
            generator,
            device,
        )
        expected_num_tokens = torch.bincount(
            indices.flatten().long(),
            minlength=num_experts,
        ).to(torch.int32)
        return indices, expected_num_tokens

    @classmethod
    def create(
        cls,
        *,
        num_experts: int,
        num_experts_per_token: int,
        max_num_tokens: int,
        hidden_dim: int,
        hidden_dim_scale: Optional[int],
        in_dtype: torch.dtype,
        scale_dtype: Optional[torch.dtype],
        generator: torch.Generator,
        device: torch.device,
    ) -> "RankTestData":
        assert num_experts_per_token <= num_experts

        indices, expected_num_tokens = cls.rand_indices_and_count(
            num_experts, num_experts_per_token, max_num_tokens, generator, device
        )
        dp_x = torch.randn(
            (max_num_tokens, hidden_dim),
            device=device,
            generator=generator,
        ).to(in_dtype)

        dp_x_scale: Optional[torch.Tensor]
        if hidden_dim_scale is not None or scale_dtype is not None:
            assert hidden_dim_scale is not None
            assert scale_dtype is not None
            dp_x_scale = torch.randn(
                (max_num_tokens, hidden_dim_scale),
                device=device,
                generator=generator,
            ).to(scale_dtype)
        else:
            dp_x_scale = None

        weights = torch.rand(
            (max_num_tokens, num_experts_per_token),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        return cls(
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            expected_num_tokens=expected_num_tokens,
            bound_m=None,
        )
