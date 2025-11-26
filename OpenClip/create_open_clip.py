
import timm
import torch
from einops import rearrange
from kornia.feature.dedode.transformer.layers.patch_embed import PatchEmbed as KorniaPatchEmbed
from dino3d.models.utils.spatial_adapter import DinoSpatialAdapter
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from timm.layers.pos_embed import resample_abs_pos_embed
import torch.nn.functional as F
import math
from timm.layers.pos_embed import resample_abs_pos_embed


def sa_to_output_tokens(spatial_tokens, cls_token, consistency_tokens):
    # Concat cls token
    x = torch.cat([cls_token, spatial_tokens + consistency_tokens], dim=0)
    return x

def add_plucker_embedding(self, x, pe, feat_shape):
    if self.disable_plucker:
        return x
    b, s, h, w = feat_shape
    pe = rearrange(pe, 'b s c h w -> (b s) h w c')
    x = torch.cat([x, pe], dim=3)
    return x

def custom_pos_embed(self, x, hw: tuple[int, int] | None = None):
    """Add class token and absolute positional embedding.

    If hw (H, W) is provided, resample the pos_embed to that exact grid.
    Otherwise, falls back to a square-root heuristic (not recommended).
    """
    if self.cls_token is not None:
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        if hw is not None:
            H, W = hw
        else:
            # Fallback: infer from token count (may be wrong if non-square)
            num_patches = x.shape[1] - self.num_prefix_tokens
            H = W = int(math.sqrt(num_patches))
        pos_embed = resample_abs_pos_embed(self.pos_embed, [H, W], num_prefix_tokens=self.num_prefix_tokens)
        x = x + pos_embed.to(x.device, dtype=x.dtype)
    return x

def prepare_plucker_embedding(self, plucker, feat_shape):
    if self.disable_plucker:
        return None
    b, s, h, w = feat_shape
    pe = rearrange(plucker, 'b s c h w -> (b s) c h w')
    pe = self.plucker_embed(pe)
    pe = rearrange(pe, '(b s) (h w) c -> b s c h w', b=b, s=s, h=h, w=w)
    return pe


def custom_forward_features(self, 
                            x: torch.Tensor, 
                            camera_pe: torch.Tensor = None,
                            return_cls_token=False) -> torch.Tensor:
    """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
    b, s, _, _, _ = x.shape
    h = x.shape[3] // 16
    w = x.shape[4] // 16
    x = rearrange(x, 'b s c h w -> (b s) c h w')
    x = self.patch_embed(x)
    # Use exact (h, w) derived from input resolution to avoid token mismatch
    x = custom_pos_embed(self, x, hw=(h, w))
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    if camera_pe is None:
        for blk in self.blocks:
            x = blk(x)
    else:
        pe = prepare_plucker_embedding(self, camera_pe, (b, s, h, w))
        for i, block in enumerate(self.blocks):
            x = block(x)
            patches = x[:, 1:, :]
            spatial_tokens = rearrange(patches, '(b s) (h w) c -> (b s) h w c', b=b, s=s, h=h, w=w)
            if i == 0:
                spatial_tokens = add_plucker_embedding(self, spatial_tokens, pe, (b, s, h, w))
            spatial_tokens = rearrange(spatial_tokens, '(b s) h w c -> (b s) (h w) c', b=b, s=s, h=h, w=w)
            consistency_tokens = self.spatial_adapters[i](spatial_tokens, (b, s, h, w))
            patches = patches + consistency_tokens
            x = torch.cat([x[:, 0:1, :], patches], dim=1)
    x = self.norm(x)
    spatial_tokens = x[:, 1:, :]
    spatial_tokens = rearrange(spatial_tokens, '(b s) (h w) c -> b s c h w', b=b, s=s, h=h, w=w)
    if return_cls_token:
        cls_token = x[:, 0, :]
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        cls_token = rearrange(cls_token, '(b s) c h w -> b s c h w', b=b, s=s, h=h, w=w)
        return torch.cat([cls_token, spatial_tokens], dim=2)
    return spatial_tokens

def open_clip_load(args,
                   spatial_reduction=False,
                   use_sa_ffn=True,
                   plucker_emb_dim=128,
                   disable_plucker=False):
    """Create a CLIP model with custom forward_features method and optional
    spatial adapters and plucker embedding for SAM3D/CLIP3D functionality.
    """
    # Load a CLIP model from timm
    # Note: you can change the model name to load different CLIP variants
    model = timm.create_model('vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', pretrained=True)
    
    # Allow variable image sizes
    model.patch_embed.img_size = None
    model._pos_embed = custom_pos_embed.__get__(model, type(model))
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)

    # Freeze all parameters first, then unfreeze the newly added layers so we train only them.
    for p in model.parameters():
        p.requires_grad = False

    
    if getattr(args, "use_lora", False):
        # Auto-detect actual submodule attribute names that are supported by PEFT
        target_modules = set()
        for name, module in model.named_modules():
            # Only target leaf modules of supported types so we don't pass composite modules like Mlp
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.MultiheadAttention)):
                target_modules.add(name.split('.')[-1])

        # Add some common fallback names used in transformer/ViT implementations
        fallback_names = ["qkv", "q_proj", "k_proj", "v_proj", "q", "k", "v", "fc1", "fc2", "proj", "out_proj"]
        if not target_modules:
            target_modules = set(fallback_names)

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)

    # Monkey patch pretrained model
    model.vision_patch_size = model_args['patch_size']
    model.extended = (plucker_emb_dim is not None and plucker_emb_dim != 0) or spatial_reduction or use_sa_ffn
    if model.extended:
        print("Creating Transformer in extended 3D mode")
        # spatial adapters and plucker embedding used by CLIP3D
        if disable_plucker:
            plucker_emb_dim = 0
        else:
            model.plucker_embed = KorniaPatchEmbed(
            patch_size=model_args['patch_size'],
            in_chans=6,
            embed_dim=plucker_emb_dim,
            )
        first_spatial_adapter_layer = DinoSpatialAdapter(model_args['embed_dim'] + plucker_emb_dim,
                                                        model_args['embed_dim'],
                                                        spatial_reduction=spatial_reduction,
                                                        use_ffn=use_sa_ffn)
        spatial_adapters_layer = DinoSpatialAdapter(model_args['embed_dim'],
                                                    model_args['embed_dim'],
                                                    spatial_reduction=spatial_reduction,
                                                    use_ffn=use_sa_ffn)
        model.spatial_adapters = nn.ModuleList([first_spatial_adapter_layer] + [spatial_adapters_layer for _ in range(model_args['depth']-1)])

    
    model.image_features = custom_forward_features.__get__(model, type(model))
    model.disable_plucker = disable_plucker
    return model.float()


if __name__ == "__main__":
    # Example usage
    model = open_clip_load({}, plucker_emb_dim=128)
    x = torch.randn(1, 4, 3, 512, 512)
    pe = torch.randn(1, 4, 6, 512, 512)  # Example plucker embedding
    out = model.image_features(x, pe, return_cls_token=True)
    print(out.shape)  # Should print the shape of the output features