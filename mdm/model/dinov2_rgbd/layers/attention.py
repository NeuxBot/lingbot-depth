# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


def use_xformers(x: Tensor) -> bool:
    """Whether to use xFormers' memory-efficient attention for tensor ``x``.

    xFormers' ``memory_efficient_attention`` is a CUDA kernel, so it is only
    selected when xFormers is installed *and* the data lives on a CUDA device.
    On CPU (or CUDA without xFormers) we fall back to PyTorch's native scaled
    dot-product attention, which is the fastest portable option there.
    """
    return XFORMERS_AVAILABLE and x.is_cuda


class BlockDiagonalAttnMask:
    """Pure-PyTorch stand-in for xFormers' ``BlockDiagonalMask``.

    Used on the SDPA path (CPU, or CUDA without xFormers) to pack several
    variable-length sequences into a single batch element while restricting
    attention to within each sequence. A single sequence needs no mask at all
    (full attention), which keeps the common batch-size-1 inference path on the
    fast, unmasked SDPA kernel.
    """

    def __init__(self, seqlens) -> None:
        self.seqlens = [int(s) for s in seqlens]
        self._mask_cache = {}

    @classmethod
    def from_seqlens(cls, seqlens) -> "BlockDiagonalAttnMask":
        return cls(seqlens)

    def split(self, x: Tensor):
        """Split a packed ``(1, sum(seqlens), ...)`` tensor back into a list."""
        return list(x.split(self.seqlens, dim=1))

    def materialize(self, device: torch.device):
        """Boolean attention mask (``True`` keeps) of shape ``(N, N)``.

        Returns ``None`` for a single sequence so SDPA can use its fastest
        unmasked kernel. Masks are cached per-device since the same mask object
        is reused across every transformer block in a forward pass.
        """
        if len(self.seqlens) <= 1:
            return None
        key = (device.type, device.index)
        mask = self._mask_cache.get(key)
        if mask is None:
            total = sum(self.seqlens)
            mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            start = 0
            for s in self.seqlens:
                end = start + s
                mask[start:end, start:end] = True
                start = end
            self._mask_cache[key] = mask
        return mask


def _to_sdpa_mask(attn_bias, device: torch.device):
    """Convert ``attn_bias`` into a mask accepted by ``F.scaled_dot_product_attention``."""
    if attn_bias is None:
        return None
    if isinstance(attn_bias, BlockDiagonalAttnMask):
        return attn_bias.materialize(device)
    return attn_bias  # already a tensor mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # # Deprecated implementation, extremely slow
    # def forward(self, x: Tensor, attn_bias=None) -> Tensor:
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    #     attn = q @ k.transpose(-2, -1)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

        q, k, v = qkv.unbind(0)      # (B, H, N, C // H)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=_to_sdpa_mask(attn_bias, x.device))
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # On CUDA with xFormers, use its memory-efficient attention kernel.
        # Otherwise (CPU, or CUDA without xFormers) fall back to PyTorch's
        # native SDPA, handled by the base class.
        if not use_xformers(x):
            return super().forward(x, attn_bias)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
