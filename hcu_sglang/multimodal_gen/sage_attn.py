# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return False

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if extra_impl_args.get("dropout_p", 0.0) != 0.0:
            raise ValueError("HCU SageAttention does not support dropout.")
        sage = import_module("sageattention")
        self._sageattn = sage.sageattn
        self._sageattn_varlen = sage.sageattn_varlen
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        return self._sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            return_lse=return_softmax_lse,
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = cu_seqlens_host
        if bounds is None:
            bounds = tuple(int(x) for x in cu_seqlens.tolist())
        if any(right < left for left, right in zip(bounds[:-1], bounds[1:])):
            raise ValueError("packed sequence offsets must be nondecreasing")
        if query.shape[0] == 0:
            return torch.empty_like(query)
        if any(right == left for left, right in zip(bounds[:-1], bounds[1:])):
            bounds = tuple(dict.fromkeys(bounds))
            cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device=query.device)
        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
        output = self._sageattn_varlen(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output[0] if isinstance(output, (tuple, list)) else output
