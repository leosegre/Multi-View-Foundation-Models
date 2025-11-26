from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import Tensor

from FiT3D.model_utils import build_2d_model

finetuned_checkpoints = {
    "dinov2_small": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_small_finetuned.pth",
    "dinov2_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_base_finetuned.pth",
    "dinov2_reg_small": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_reg_small_finetuned.pth",
    "clip_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/clip_base_finetuned.pth",
    "mae_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/mae_base_finetuned.pth",
    "deit3_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/deit3_base_finetuned.pth"
}


class FiT3D(nn.Module):
    def __init__(
            self,
            backbone_type,
            with_cls_token=False
    ):
        super().__init__()

        self.vit = build_2d_model(backbone_type)
        self.finetuned_model = build_2d_model(backbone_type)
        fine_ckpt = torch.hub.load_state_dict_from_url(finetuned_checkpoints[backbone_type], map_location='cpu')
        msg = self.finetuned_model.load_state_dict(fine_ckpt, strict=False)
        self.with_cls_token = with_cls_token

        print(msg)

        # self.fuse_layer = nn.Linear(self.vit.embed_dim*2, self.vit.embed_dim)

    def forward(
            self,
            x: Tensor,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
            return_dict=False,
            return_channel_first=False,
    ) -> Tensor:
        # run backbone if backbone is there
        prefix_tokens, raw_vit_feats = None, None
        if self.vit is not None:
            if len(x.shape) == 4:
                x = x.unsqueeze(0)  # add sequence dimension
            B, S, C, H, W = x.shape
            x = x.reshape(B * S, C, H, W)
            with torch.no_grad():
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[len(self.vit.blocks) - 1],
                    reshape=True,
                    return_prefix_tokens=return_prefix_tokens,
                    return_class_token=return_class_token,
                    norm=norm,
                )  # [2, 384, 37, 37]
                vit_outputs = (
                    vit_outputs[-1]
                    if return_prefix_tokens or return_class_token
                    else vit_outputs
                )
                raw_vit_feats = vit_outputs[0].permute(0, 2, 3, 1).detach()

                vit_outputs_fine = self.finetuned_model.get_intermediate_layers(
                    x,
                    n=[len(self.finetuned_model.blocks) - 1],
                    reshape=True,
                    return_prefix_tokens=return_prefix_tokens,
                    return_class_token=self.with_cls_token,
                    norm=norm,
                )
                if self.with_cls_token:
                    raw_vit_feats_fine, cls_token = vit_outputs_fine[0]
                    h, w = raw_vit_feats_fine.shape[2:4]
                    cls_token = cls_token.detach()[:, None, None].expand(B * S, h, w, -1)
                else:
                    raw_vit_feats_fine = vit_outputs_fine[0]

                raw_vit_feats_fine = raw_vit_feats_fine.permute(0, 2, 3, 1).detach()

                ## strategy 1: concatenate
                x = torch.cat([raw_vit_feats, raw_vit_feats_fine], -1)
                if self.with_cls_token:
                    x = torch.cat([x, cls_token], -1)
                ## strategy 2: adding
                # x = raw_vit_feats + raw_vit_feats_fine
                ## strategy 3: linear fusion
                # x = self.fuse_layer(x)

                if return_prefix_tokens or return_class_token:
                    prefix_tokens = vit_outputs[1]
        BS, H, W, C = x.shape
        x = x.reshape(B, S, H, W, C)
        if raw_vit_feats is not None:
            _, H, W, C = raw_vit_feats.shape
            raw_vit_feats = raw_vit_feats.reshape(B, S, H, W, C)
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.reshape(B, S, -1)
        if raw_vit_feats_fine is not None:
            _, H, W, C = raw_vit_feats_fine.shape
            raw_vit_feats_fine = raw_vit_feats_fine.reshape(B, S, H, W, C)
        out_feat = x

        if return_channel_first:
            out_feat = out_feat.permute(0, 1, 4, 2, 3)
            raw_vit_feats = (
                raw_vit_feats.permute(0, 1, 4, 2, 3) if raw_vit_feats is not None else None
            )
            raw_vit_feats_fine = (
                raw_vit_feats_fine.permute(0, 1, 4, 2, 3) if raw_vit_feats_fine is not None else None
            )
        if return_dict:
            return {
                "raw_vit_feats": raw_vit_feats,
                "prefix_tokens": prefix_tokens,
                "raw_vit_feats_fine": raw_vit_feats_fine,
                "out_feat": out_feat,
            }
        if prefix_tokens is not None:
            return out_feat, prefix_tokens
        return out_feat