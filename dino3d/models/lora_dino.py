import math
import types

import einops as E
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.modules.utils as nn_utils

from dino3d.models.utils.spatial_adapter import DinoSpatialAdapter
from dino3d.models.base_dino import DINO, tokens_to_output
from kornia.feature.dedode.transformer.layers.patch_embed import PatchEmbed

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv

class LORA_DINO(DINO):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vits14",
        output="dense",
        layer=-1,
        return_multilayer=True,
        stride=7,
        disable_dino_pe=False,
        lora_rank=4,
    ):
        super().__init__(dino_name=dino_name,
                         model_name=model_name,
                         output=output,
                         layer=layer,
                         return_multilayer=return_multilayer,
                         stride=stride,
                         disable_dino_pe=disable_dino_pe)

        self.lora_layers = list(range(len(self.vit.blocks)))
        self.w_a = []
        self.w_b = []

        for i, block in enumerate(self.vit.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features

            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, lora_rank)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, lora_rank)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

