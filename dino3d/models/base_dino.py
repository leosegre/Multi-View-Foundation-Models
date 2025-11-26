import math
import os
import types
import einops as E
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
from peft import LoraConfig, get_peft_model


def resize_pos_embed(
    pos_embed: torch.Tensor, hw: tuple[int, int], has_cls_token: bool = True
):
    """
    Resize positional embedding for arbitrary image resolution. Resizing is done
    via bicubic interpolation.

    Args:
        pos_embed: Positional embedding tensor of shape ``(n_patches, embed_dim)``.
        hw: Target height and width of the tensor after interpolation.
        has_cls_token: Whether ``pos_embed[0]`` is for the ``[cls]`` token.

    Returns:
        Tensor of shape ``(new_n_patches, embed_dim)`` of resized embedding.
        ``new_n_patches`` is ``new_height * new_width`` if ``has_cls`` is False,
        else ``1 + new_height * new_width``.
    """

    n_grid = pos_embed.shape[0] - 1 if has_cls_token else pos_embed.shape[0]

    # Do not resize if already in same shape.
    if n_grid == hw[0] * hw[1]:
        return pos_embed

    # Get original position embedding and extract ``[cls]`` token.
    if has_cls_token:
        cls_embed, pos_embed = pos_embed[[0]], pos_embed[1:]

    orig_dim = int(pos_embed.shape[0] ** 0.5)

    pos_embed = E.rearrange(pos_embed, "(h w) c -> 1 c h w", h=orig_dim)
    pos_embed = F.interpolate(
        pos_embed, hw, mode="bicubic", align_corners=False, antialias=True
    )
    pos_embed = E.rearrange(pos_embed, "1 c h w -> (h w) c")

    # Add embedding of ``[cls]`` token back after resizing.
    if has_cls_token:
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)

    return pos_embed


def center_padding(images, patch_size):
    print(images.shape)
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images


def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output


class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vits14",
        output="dense",
        layer=-1,
        return_multilayer=True,
        stride=7,
        disable_dino_pe=False,
        lora_finetune=False,
        interp_pe=True,
    ):
        super().__init__()
        self.feat_dims = {
            "vits14": 384,
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        if 'dinov3' in dino_name:
            repo_dir = "/home/adminubuntu/research/Dino3D/third_party/dinov3"
            checkpoint_path = os.path.join(repo_dir, 'dinov3_weights', self.checkpoint_name)
            dino_vit = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=checkpoint_path)
        else:
            dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        self.lora_finetune = lora_finetune
        if lora_finetune:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv", "ffn"],
                lora_dropout=0.1,
                bias="none",
            )
            self.vit = get_peft_model(self.vit, config)

        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = self.feat_dims[model_name.replace('_reg', '')]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        self.feat_dim = feat_dim
        num_layers = len(self.vit.blocks)
        self.num_layers = num_layers
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.stride = stride
        self.disable_dino_pe = disable_dino_pe
        
        if 'dinov2' in dino_name and interp_pe:
            self.set_overlapping_patches()

    def forward(self, images):

        b, _, h, w = images.shape
        h, w = 1 + (h - self.patch_size) // self.stride, 1 + (w - self.patch_size) // self.stride

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, dino_blk in enumerate(self.vit.blocks):
            x = dino_blk(x)
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
            outputs.append(x_i)
        return outputs[0] if len(outputs) == 1 else outputs

    def set_overlapping_patches(self):
        # if self.patch_size == self.stride:
        #     return
        stride = nn_utils._pair(self.stride)
        # fix the stride
        self.vit.patch_embed.proj.stride = stride
        # fix the positional encoding code
        self.vit.patch_embed.forward = types.MethodType(DINO.custom_patch_embed, self.vit.patch_embed)

        if self.lora_finetune:
            self.vit.base_model.model.interpolate_pos_encoding = types.MethodType(DINO._fix_pos_enc(self.patch_size, stride, self.disable_dino_pe), self.vit)
        else:
            self.vit.interpolate_pos_encoding = types.MethodType(DINO._fix_pos_enc(self.patch_size, stride, self.disable_dino_pe), self.vit)

    @staticmethod
    def _fix_pos_enc(patch_size, stride_hw, disable_dino_pe=False):
        def interpolate_pos_encoding(self, x, w, h):
            if disable_dino_pe:
                return 0.
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def custom_patch_embed(self, x):
        _, _, H, W = x.shape

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x  # or your own implementation