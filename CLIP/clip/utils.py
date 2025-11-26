import torch
from peft import LoraConfig, get_peft_model

# Use the local altered CLIP implementation
from .clip import load as clip_load


def get_clip_model(args, 
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   spatial_reduction=None,
                   use_sa_ffn=None,
                   plucker_emb_dim=None) -> torch.nn.Module:
    """Load the local altered CLIP model. This tries to use
    `args.clip_model_name` (if present) or falls back to `args.model_name`.

    If `args.use_lora` is true the returned model will be wrapped with PEFT/LoRA.
    """
    # gather new-layer flags from args
    spatial_reduction = getattr(args, "spatial_reduction", False) if spatial_reduction is None else spatial_reduction
    use_sa_ffn = getattr(args, "use_sa_ffn", False) if use_sa_ffn is None else use_sa_ffn
    plucker_emb_dim = getattr(args, "plucker_emb_dim", 0) if plucker_emb_dim is None else plucker_emb_dim

    # clip_load returns (model, preprocess). We only need the model here.
    model, _ = clip_load(args.model_name, 
                         device=device,
                         image_size=args.image_size,
                         spatial_reduction=spatial_reduction,
                         use_sa_ffn=use_sa_ffn,
                         plucker_emb_dim=plucker_emb_dim,
                         )


    # Freeze all parameters first, then unfreeze the newly added layers so we train only them.
    for p in model.parameters():
        p.requires_grad = False

    # New layers to unfreeze: spatial adapters and plucker patch embed inside the vision transformer
    def _unfreeze_new_layers(m: torch.nn.Module):
        names = [
            "transformer.spatial_adapters",
            "transformer.plucker_embed",
            # also allow older code that might nest under visual.transformer
            "visual.transformer.spatial_adapters",
            "visual.transformer.plucker_embed",
        ]
        for n in names:
            sub = m
            parts = n.split('.')
            try:
                for p in parts:
                    sub = getattr(sub, p)
            except Exception:
                sub = None
            if sub is not None:
                for param in sub.parameters():
                    param.requires_grad = True

    # Try unfreezing both top-level transformer and visual.transformer paths
    _unfreeze_new_layers(model)

    if getattr(args, "use_lora", False):
        # Auto-detect actual submodule attribute names that are supported by PEFT
        target_modules = set()
        for name, module in model.visual.named_modules():
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
        model.visual = get_peft_model(model.visual, config)

    return model