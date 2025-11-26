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


class DINO3D_V2(DINO):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vits14",
        output="dense",
        layer=-1,
        return_multilayer=True,
        stride=7,
        pe_embedding_strategy="concat",
        plucker_emb_dim=128,
        disable_plucker=False,
        use_sa_cls_token=False,
        disable_dino_pe=False,
        fix_spatial_attn=False,
        sa_parametric_spatial_conv=False,
        spatial_reduction=True,
        sa_before_spatial=True,
        lora_finetune=False,
        plucker_mlp=False,
        sa_layers=(),
    ):
        super().__init__(dino_name=dino_name,
                         model_name=model_name,
                         output=output,
                         layer=layer,
                         return_multilayer=return_multilayer,
                         stride=stride,
                         disable_dino_pe=disable_dino_pe,
                         lora_finetune=lora_finetune)

        dino_hidden_dim = self.feat_dims[model_name]
        self.sa_before_spatial = sa_before_spatial
        if not sa_before_spatial and pe_embedding_strategy == "concat":
            raise ValueError("If sa_before_spatial is False, then pe_embedding_strategy must be 'addition'.")
        if pe_embedding_strategy not in ["concat", "addition"]:
            raise ValueError(f"PE embedding strategy {pe_embedding_strategy} not supported.")
        self.pe_embedding_strategy = pe_embedding_strategy
        if pe_embedding_strategy == "concat" and not disable_plucker:
            input_dim = dino_hidden_dim + plucker_emb_dim
        else:
            input_dim = dino_hidden_dim
            plucker_emb_dim = dino_hidden_dim
        self.sa_layers = sa_layers
        if len(sa_layers) == 0:
            self.sa_layers = tuple(range(self.num_layers))
        first_spatial_adapter_layer = DinoSpatialAdapter(input_dim,
                                                         dino_hidden_dim,
                                                         parametric_spatial_conv=sa_parametric_spatial_conv,
                                                         spatial_reduction=spatial_reduction,
                                                         use_ffn=False)
        spatial_adapters_layer = DinoSpatialAdapter(dino_hidden_dim,
                                                    dino_hidden_dim,
                                                    parametric_spatial_conv=sa_parametric_spatial_conv,
                                                    spatial_reduction=spatial_reduction,
                                                    use_ffn=False)
        self.spatial_adapters = nn.ModuleList([first_spatial_adapter_layer] +
                                              [spatial_adapters_layer for _ in range(len(self.sa_layers)-1)])

        self.plucker_mlp = plucker_mlp
        if plucker_mlp:
            self.plucker_embed = nn.Sequential(
                nn.Conv2d(6, plucker_emb_dim, kernel_size=self.patch_size, stride=self.stride),
                nn.SiLU(),
                nn.Conv2d(plucker_emb_dim, plucker_emb_dim, kernel_size=1, stride=1),
            )
        else:
            self.plucker_embed = PatchEmbed(
                patch_size=self.patch_size,
                in_chans=6,
                embed_dim=plucker_emb_dim,
            )
            self.plucker_embed.proj.stride = nn_utils._pair(self.stride)

        self.disable_plucker = disable_plucker

        self.use_sa_cls_token = use_sa_cls_token
        # Freeze Dino original parameters
        if not lora_finetune:
            self.vit.requires_grad_(False)


    def forward(self, images, camera_pe):
        # pad images (if needed) to ensure it matches patch_size
        # images = center_padding(images, self.patch_size)
        # camera_pe = center_padding(camera_pe, self.patch_size)

        b, s, _, h, w = images.shape
        h, w = 1 + (h - self.patch_size) // self.stride, 1 + (w - self.patch_size) // self.stride

        images = rearrange(images, 'b s c h w -> (b s) c h w')
        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        # Add plucker embedding
        pe = self.prepare_plucker_embedding(camera_pe, (b, s, h, w))

        embeds = []
        sa_layers = list(self.sa_layers)
        j = 0
        for i, dino_blk in enumerate(self.vit.blocks):
            # Base Dino Forward Pass
            if self.sa_before_spatial:
                # 3D Attention
                if len(sa_layers) > 0 and i == sa_layers[0]:
                    spatial_tokens, cls_token = x[:, 1:], x[:, 0].unsqueeze(1)
                    if len(self.spatial_adapters) == len(sa_layers):
                        spatial_tokens_in = self.add_plucker_embedding(x, pe, (b, s, h, w))
                    else:
                        spatial_tokens_in = spatial_tokens
                    input_sa_tokens = self.input_tokens_to_sa(spatial_tokens_in, cls_token)
                    consistency_tokens = self.spatial_adapters[j](input_sa_tokens, (b, s, h, w))
                    x = self.sa_to_output_tokens(spatial_tokens, cls_token, consistency_tokens)
                    sa_layers = sa_layers[1:]  # drop the first entry
                    j += 1
                # Spatial Attention
                x = x + dino_blk.ls1(dino_blk.attn(dino_blk.norm1(x)))
                # FFN
                x = x + dino_blk.ls2(dino_blk.mlp(dino_blk.norm2(x)))
            else:
                # Spatial Attention
                x = x + dino_blk.ls1(dino_blk.attn(dino_blk.norm1(x)))
                # 3D Attention
                if len(sa_layers) > 0 and i == sa_layers[0]:
                    spatial_tokens, cls_token = x[:, 1:], x[:, 0].unsqueeze(1)
                    if len(self.spatial_adapters) == len(sa_layers):
                        spatial_tokens_in = self.add_plucker_embedding(x, pe, (b, s, h, w))
                    else:
                        spatial_tokens_in = spatial_tokens
                    input_sa_tokens = self.input_tokens_to_sa(spatial_tokens_in, cls_token)
                    consistency_tokens = self.spatial_adapters[j](input_sa_tokens, (b, s, h, w))
                    x = self.sa_to_output_tokens(spatial_tokens, cls_token, consistency_tokens)
                    sa_layers = sa_layers[1:]  # drop the first entry
                    j += 1
                # FFN
                x = x + dino_blk.ls2(dino_blk.mlp(dino_blk.norm2(x)))

            # Save intermediate outputs
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            x_i = rearrange(x_i, '(b s) c h w -> b s c h w', b=b, s=s)
            outputs.append(x_i)
        return outputs[0] if len(outputs) == 1 else outputs

    def input_tokens_to_sa(self, spatial_tokens, cls_token):
        if self.use_sa_cls_token:
            return torch.cat([cls_token, spatial_tokens], dim=1)
        return spatial_tokens

    def sa_to_output_tokens(self, spatial_tokens, cls_token, consistency_tokens):
        if self.use_sa_cls_token:
            x = torch.cat([cls_token, spatial_tokens], dim=1)
            return consistency_tokens + x
        # Concat cls token
        x = torch.cat([cls_token, spatial_tokens + consistency_tokens], dim=1)
        return x

    def add_plucker_embedding(self, x, pe, feat_shape):
        if self.disable_plucker:
            return x[:, 1:]
        b, s, h, w = feat_shape
        spatial_tokens = rearrange(x[:, 1:], '(b s) (h w) c -> b s c h w', b=b, s=s, h=h, w=w)
        if self.pe_embedding_strategy == "addition":
            spatial_tokens = spatial_tokens + pe
        elif self.pe_embedding_strategy == "concat":
            spatial_tokens = torch.cat([spatial_tokens, pe], dim=2)
        spatial_tokens = rearrange(spatial_tokens, 'b s c h w -> (b s) (h w) c', b=b, s=s, h=h, w=w)
        return spatial_tokens

    def prepare_plucker_embedding(self, plucker, feat_shape):
        if self.disable_plucker:
            return None
        b, s, h, w = feat_shape
        pe = rearrange(plucker, 'b s c h w -> (b s) c h w')
        pe = self.plucker_embed(pe)
        if self.plucker_mlp:
            pe = rearrange(pe, '(b s) c h w -> b s c h w', b=b, s=s, h=h, w=w)
        else:
            pe = rearrange(pe, '(b s) (h w) c -> b s c h w', b=b, s=s, h=h, w=w)
        return pe