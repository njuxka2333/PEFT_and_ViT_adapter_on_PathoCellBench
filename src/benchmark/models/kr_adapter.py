"""KRLinear adapter for UNI2 ViT layers (pathocellbench; not in Pathology-Foundation-Model-Benchmark).

Used by ``benchmark.models.simple_segmentation_model.load_model_and_transform`` when ``kradapter_enable`` is true.
Trainable factors are ``kr_a`` / ``kr_b``; frozen pretrained weights stay in the wrapped ``nn.Linear``.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KRLinear(nn.Linear):
    """KRAdapter-style linear wrapper.

    Delta weight is generated from two Kronecker-factorized tensors following:
    update = flatten((A * B), start_dim=1), with orientation/cropping to match base weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank_order: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.rank_order = int(rank_order)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        max_dim = max(in_features, out_features)
        min_dim = min(in_features, out_features)
        # If bottleneck_dim is set, use it as the Kronecker "rank" budget (similar intent to LoRA rank).
        if bottleneck_dim is not None:
            bd = max(1, int(bottleneck_dim))
            effective_rank = min(bd, max_dim)
            rank2 = max(1, int(math.ceil(max_dim / float(effective_rank))))
        else:
            effective_rank = max(1, int(max_dim ** (1.0 / max(self.rank_order, 1))))
            rank2 = int(math.ceil(max_dim / float(effective_rank)))

        self.kr_a = nn.Parameter(torch.zeros(min_dim, 1, rank2))
        self.kr_b = nn.Parameter(torch.zeros(min_dim, effective_rank, 1))
        nn.init.kaiming_uniform_(self.kr_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.kr_b, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def _kr_update(self):
        w = self.weight
        update = (self.kr_a * self.kr_b).flatten(start_dim=1)
        if update.shape[0] != w.shape[0]:
            update = update.T
        update = update[:, : w.shape[1]]
        return update

    def forward(self, x: torch.Tensor):
        update = self._kr_update().to(dtype=self.weight.dtype, device=self.weight.device)
        adapted_weight = self.weight + self.alpha * update
        x = self.dropout(x)
        return F.linear(x, adapted_weight, self.bias)
