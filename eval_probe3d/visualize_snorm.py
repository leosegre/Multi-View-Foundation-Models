"""
Visualize surface normals on the validation (test) split.

- Loads model + probe from a checkpoint saved by eval_probe3d/train_snorm.py
- Supports eval_base and eval_fit3d modes (skip camera PE and use multilayers accordingly)
- Saves RGB visualizations for input image, predicted normals, and GT normals

Usage (Hydra):
    # Note: keys not in the base config must be added with a leading '+'
    python eval_probe3d/visualize_snorm.py exp_directory=work_dirs \
        +config_file=snorm_training.yaml \
        +ckpt_path=outputs/your_exp/ckpt.pth \
        +vis_max_samples=50 \
        eval_base=true  # or eval_fit3d=true

Notes:
- We treat the repo's "test" split as the validation set (the training script logs call it validation).
- For multi-view batches, we save only the first view per sample to avoid duplicates.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from evals.datasets.builder import build_loader  # type: ignore

# Reuse helpers from training
from eval_probe3d.train_snorm import (
    load_dino3d_model,
    load_base_dino_model,
    load_fit3d_model,
    get_multilayers_from_model,
)


def get_mean_std(image_mean: str) -> Tuple[List[float], List[float]]:
    if image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif image_mean == "None":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        # default to imagenet if unknown
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def denorm_image(img: torch.Tensor, image_mean: str) -> torch.Tensor:
    """img: tensor [3,H,W] normalized by dataset mean/std. Returns [3,H,W] in [0,1]."""
    mean, std = get_mean_std(image_mean)
    device = img.device
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    img = img * std_t + mean_t
    return img.clamp(0, 1)


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """Convert surface normals in [-1,1] to RGB in [0,1].
    normals: [3,H,W] or [B,3,H,W]
    """
    if normals.dim() == 3:
        n = normals
    else:
        assert normals.dim() == 4 and normals.size(1) >= 3
        n = normals[:, :3]
    # normalize to unit vectors just in case
    if n.dim() == 3:
        n = F.normalize(n, dim=0)
    else:
        n = F.normalize(n, dim=1)
    rgb = (n + 1.0) * 0.5
    return rgb.clamp(0, 1)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """img in [0,1], shape [3,H,W]."""
    img = (img.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def create_2x2_grid(images: List[Image.Image], padding: int = 0) -> Image.Image:
    """Create a 2x2 grid from 4 images.
    
    Args:
        images: List of 4 PIL Images
        padding: Padding between images in pixels (default: 0 for no borders)
    
    Returns:
        PIL Image containing the 2x2 grid
    """
    assert len(images) == 4, "Need exactly 4 images for 2x2 grid"
    
    # Get dimensions (assume all images are same size)
    w, h = images[0].size
    
    # Create output image with padding
    grid_w = 2 * w + padding  # 2 images + 1 padding space in middle (if any)
    grid_h = 2 * h + padding  # 2 images + 1 padding space in middle (if any)
    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))  # White background
    
    # Paste images in 2x2 layout
    positions = [
        (0, 0),                      # Top-left
        (w + padding, 0),            # Top-right
        (0, h + padding),            # Bottom-left
        (w + padding, h + padding)   # Bottom-right
    ]
    
    for img, pos in zip(images, positions):
        grid.paste(img, pos)
    
    return grid


def apply_mask_rgb(rgb: torch.Tensor, mask: torch.Tensor, bg: float = 0.0) -> torch.Tensor:
    """Apply binary mask to RGB in [0,1].
    rgb: [B,3,H,W] or [3,H,W]; mask: [B,1,H,W] or [1,H,W] with 1=valid, 0=invalid.
    bg: background fill value in [0,1].
    """
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Ensure mask has exactly 1 channel (take first channel if multiple)
    if mask.shape[1] > 1:
        mask = mask[:, :1]
    
    mask = mask.to(rgb.dtype)
    mask3 = mask.expand(-1, 3, -1, -1)
    bg_tensor = torch.full_like(rgb, fill_value=bg)
    out = rgb * mask3 + bg_tensor * (1.0 - mask3)
    return out


def forward_model(model, images, plucker=None, skip_camera_pe=False, use_fit3d=False):
    """Forward pass producing 5D feature list: [B,V,C,H,W] per layer (or a single tensor).
    Handles 5D -> flatten for 2D-only models (base/FiT3D) and reshapes back.
    """
    is_multiview = images.ndim == 5
    if skip_camera_pe:
        # flatten to 2D batch
        if is_multiview:
            B, V = images.shape[:2]
            images_2d = rearrange(images, 'b s c h w -> (b s) c h w')
        else:
            B, V = images.shape[0], 1
            images_2d = images
        if use_fit3d:
            multilayers = get_multilayers_from_model(model)
            feats = model.get_intermediate_layers(images_2d, n=multilayers, reshape=True)
            if isinstance(feats, tuple):
                feats = list(feats)
            elif not isinstance(feats, list):
                feats = [feats]
        else:
            feats = model(images_2d)
        # reshape back to 5D list
        if is_multiview:
            feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B, s=V) for f in (feats if isinstance(feats, list) else [feats])]
        else:
            feats = [f.unsqueeze(1) for f in (feats if isinstance(feats, list) else [feats])]  # add V=1 dim
    else:
        feats = model(images, camera_pe=plucker)
        feats = feats if isinstance(feats, list) or isinstance(feats, tuple) else [feats]
    return feats


@hydra.main(config_name="snorm_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    # Set random seed for reproducibility across different model runs
    # This ensures the same samples are selected for visualization regardless of model type
    vis_seed = int(getattr(cfg, 'vis_seed', 42))
    torch.manual_seed(vis_seed)
    np.random.seed(vis_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(vis_seed)
    logger.info(f"Set visualization seed to {vis_seed} for reproducible sampling")
    
    # Merge external config file if provided
    if 'config_file' in cfg and cfg.config_file:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / cfg.config_file
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            external_cfg = yaml.safe_load(f)
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(external_cfg))
        logger.info(f"Merged config from {cfg.config_file}")

    # Required: checkpoint path
    ckpt_path = Path(getattr(cfg, 'ckpt_path', ''))
    assert ckpt_path and ckpt_path.exists(), "Please provide a valid ckpt_path to a train_snorm checkpoint (ckpt.pth)."

    # Build validation loader (repo uses 'test' split as validation)
    # Prefer a smaller default batch size for visualization to avoid OOM
    batch_size = int(getattr(cfg, 'vis_batch_size', 0) or getattr(cfg, 'batch_size', 1) or 1)
    test_loader = build_loader(cfg.dataset, "test", batch_size, 1)
    logger.info(f"Validation loader built: {len(test_loader.dataset)} samples | batch_size={batch_size}")

    # Load checkpoint (also brings stored training cfg for robust model reconstruction)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_cfg = OmegaConf.create(ckpt.get('cfg', {})) if isinstance(ckpt.get('cfg', {}), (dict, list)) else ckpt.get('cfg', {})

    # Load model type flags (prefer CLI cfg, then fall back to checkpoint cfg)
    use_base = cfg.get('eval_base', ckpt_cfg.get('eval_base', False) if isinstance(ckpt_cfg, DictConfig) or isinstance(ckpt_cfg, dict) else False)
    use_fit3d = cfg.get('eval_fit3d', ckpt_cfg.get('eval_fit3d', False) if isinstance(ckpt_cfg, DictConfig) or isinstance(ckpt_cfg, dict) else False)

    # Determine model type for output naming
    if use_base:
        model_type = "base_dino"
    elif use_fit3d:
        model_type = "fit3d"
    else:
        model_type = "dino3d"
    
    # Output dir with meaningful name based on model type
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    out_root = Path(getattr(cfg, 'vis_out_dir', 'visualizations')) / f"snorm_{model_type}_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.add(out_root / "visualize.log")

    # Build model
    if use_base:
        # Prefer current cfg, else use checkpoint cfg
        ref_cfg = cfg if (hasattr(cfg, 'model_name') or hasattr(cfg, 'stride')) else ckpt_cfg
        model = load_base_dino_model(ref_cfg)
        logger.info("Loaded base DINO (eval_base=true)")
    elif use_fit3d:
        model = load_fit3d_model(cfg)
        logger.info("Loaded FiT3D (eval_fit3d=true)")
    else:
        if ('model_name' in cfg) or hasattr(cfg, 'model_name'):
            model = load_dino3d_model(cfg)
            logger.info("Loaded DINO3D from config")
        elif (isinstance(ckpt_cfg, DictConfig) or isinstance(ckpt_cfg, dict)) and (('model_name' in ckpt_cfg) or hasattr(ckpt_cfg, 'model_name')):
            model = load_dino3d_model(ckpt_cfg)
            logger.info("Loaded DINO3D from checkpoint cfg")
        elif (isinstance(ckpt_cfg, DictConfig) or isinstance(ckpt_cfg, dict)) and ('backbone' in ckpt_cfg):
            model = instantiate(ckpt_cfg['backbone'])
            logger.info("Loaded model via Hydra instantiate from checkpoint cfg")
        else:
            # Sensible fallback: base DINO small
            model = load_base_dino_model(cfg)
            logger.warning("Falling back to base DINO (no model config found)")

    # Build probe with correct feat_dims
    if hasattr(model, 'feat_dim'):
        feat_dims = model.feat_dim if isinstance(model.feat_dim, list) else [model.feat_dim] * 4
    elif use_fit3d or use_base:
        feat_dims = [384, 384, 384, 384]
    else:
        feat_dims = [384, 384, 384, 384]
    probe = instantiate(cfg.probe, feat_dim=feat_dims)

    # Load checkpoint weights
    print("Loading checkpoint from", ckpt_path)
    model.load_state_dict(ckpt['model'], strict=False)
    probe.load_state_dict(ckpt['probe'], strict=False)
    print("Checkpoint loaded.")
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    probe = probe.to(device).eval()

    # Image un-normalization
    image_mean = cfg.dataset.get('image_mean', 'imagenet') if hasattr(cfg, 'dataset') else 'imagenet'

    # Viz limits and sparse sampling
    max_samples = int(getattr(cfg, 'vis_max_samples', 50))
    saved = 0
    
    # Calculate stride to sample sparsely across the dataset
    # This ensures we get diverse samples instead of consecutive ones
    total_batches = len(test_loader)
    sample_stride = max(1, total_batches // max_samples)  # Skip batches to spread samples across dataset
    logger.info(f"Sampling every {sample_stride} batch(es) from {total_batches} total batches for diversity")

    use_amp = torch.cuda.is_available()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            # Skip batches to sample sparsely across the dataset
            if batch_idx % sample_stride != 0:
                continue
            images = batch.get("images")  # multiview datasets
            if images is None:
                # single image datasets (NYU)
                # create shape [B,V,C,H,W] with V=1
                img = batch["image"].unsqueeze(1)
                depths = batch["depth"].unsqueeze(1)
                snorms = batch["snorm"].unsqueeze(1)
                plucker = None
                images = img
            else:
                depths = batch["depths"]
                snorms = batch["snorms"]
                plucker = batch.get("plucker")

            images = images.to(device)
            depths = depths.to(device)
            snorms = snorms.to(device)
            if plucker is not None:
                plucker = plucker.to(device)

            # Forward backbone -> features (5D list), then probe
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = forward_model(
                    model,
                    images,
                    plucker=plucker,
                    skip_camera_pe=(use_base or use_fit3d),
                    use_fit3d=use_fit3d,
                )

            # Keep features as 5D [B, V, C, H, W] for probe (it expects this shape)
            B, V = images.shape[:2]
            
            # Probe expects 5D features [B, V, C, H, W] (as list or single tensor)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = probe(feats)
            
            # Reshape pred to [B*V, C, H, W] after probe
            pred = pred.reshape(B * V, *pred.shape[2:])
            target = snorms.reshape(B * V, *snorms.shape[2:])

            # Resize to GT spatial size
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            # Select only first view for visualization
            first_view_idx = torch.arange(0, B * V, V, device=device)
            pred_first = pred[first_view_idx]
            target_first = target[first_view_idx]
            # Build corresponding validity mask from depth>0
            depth_first = depths.reshape(B * V, *depths.shape[2:])[first_view_idx]
            # Ensure mask is [B, 1, H, W] by taking any channel and keeping only 1 channel
            if depth_first.dim() == 4 and depth_first.shape[1] > 1:
                depth_first = depth_first[:, :1]  # Take only first channel
            mask_first = (depth_first > 0).to(pred_first.dtype)
            if mask_first.dim() == 3:
                mask_first = mask_first.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            # Also get the corresponding input image (de-normalized)
            imgs_first = images[:, 0]  # [B,3,H,W]
            imgs_first = torch.stack([denorm_image(imgs_first[i], image_mean) for i in range(imgs_first.size(0))], dim=0)

            # Take first 3 channels and normalize
            pred_rgb = normals_to_rgb(pred_first[:, :3])
            gt_rgb = normals_to_rgb(target_first[:, :3])
            # Apply mask for consistency with evaluation (white background)
            pred_rgb = apply_mask_rgb(pred_rgb, mask_first, bg=1.0)
            gt_rgb = apply_mask_rgb(gt_rgb, mask_first, bg=1.0)

            # Process all views for grid visualization (if V=4)
            if V == 4:
                # Prepare all view data [B*V, ...]
                pred_all = pred
                target_all = target
                depths_all = depths.reshape(B * V, *depths.shape[2:])
                images_all = images.reshape(B * V, *images.shape[2:])
                
                # Create masks for all views
                if depths_all.dim() == 4 and depths_all.shape[1] > 1:
                    depths_all = depths_all[:, :1]
                mask_all = (depths_all > 0).to(pred_all.dtype)
                if mask_all.dim() == 3:
                    mask_all = mask_all.unsqueeze(1)
                
                # Denormalize all images
                images_all_denorm = torch.stack([denorm_image(images_all[j], image_mean) for j in range(images_all.size(0))], dim=0)
                
                # Convert normals to RGB for all views
                pred_rgb_all = normals_to_rgb(pred_all[:, :3])
                gt_rgb_all = normals_to_rgb(target_all[:, :3])
                pred_rgb_all = apply_mask_rgb(pred_rgb_all, mask_all, bg=1.0)
                gt_rgb_all = apply_mask_rgb(gt_rgb_all, mask_all, bg=1.0)
            
            # Save each item in the batch
            for i in range(pred_rgb.size(0)):
                out_dir = out_root / f"{batch_idx:05d}_{i:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Save first view (original behavior)
                tensor_to_pil(imgs_first[i]).save(out_dir / "input_view0.png")
                tensor_to_pil(pred_rgb[i]).save(out_dir / "pred_normals_view0.png")
                tensor_to_pil(gt_rgb[i]).save(out_dir / "gt_normals_view0.png")
                mask_vis = mask_first[i].repeat(3,1,1).clamp(0,1)
                tensor_to_pil(mask_vis).save(out_dir / "mask_view0.png")
                
                # Create and save 2x2 grids if we have 4 views
                if V == 4:
                    # Extract the 4 views for this batch item
                    view_indices = [i * V + v for v in range(4)]
                    
                    # Input images grid
                    input_views = [tensor_to_pil(images_all_denorm[idx]) for idx in view_indices]
                    input_grid = create_2x2_grid(input_views)
                    input_grid.save(out_dir / "input_grid_4views.png")
                    
                    # Predicted normals grid
                    pred_views = [tensor_to_pil(pred_rgb_all[idx]) for idx in view_indices]
                    pred_grid = create_2x2_grid(pred_views)
                    pred_grid.save(out_dir / "pred_normals_grid_4views.png")
                    
                    # GT normals grid
                    gt_views = [tensor_to_pil(gt_rgb_all[idx]) for idx in view_indices]
                    gt_grid = create_2x2_grid(gt_views)
                    gt_grid.save(out_dir / "gt_normals_grid_4views.png")
                    
                    # Optionally save mask grid
                    mask_views = [tensor_to_pil(mask_all[idx].repeat(3,1,1).clamp(0,1)) for idx in view_indices]
                    mask_grid = create_2x2_grid(mask_views)
                    mask_grid.save(out_dir / "mask_grid_4views.png")

                saved += 1
                if saved >= max_samples:
                    logger.info(f"Saved {saved} samples to {out_root}. Stopping early as requested.")
                    return

    logger.info(f"Done. Saved {saved} samples to {out_root}")


if __name__ == "__main__":
    main()
