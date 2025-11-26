"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
from einops import rearrange
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import angular_loss
from evals.utils.metrics import evaluate_surface_norm
from evals.utils.optim import cosine_decay_linear_warmup


def ddp_setup(rank: int, world_size: int, port: int = 12355):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    rank=0,
    world_size=1,
    valid_loader=None,
    skip_camera_pe: bool = False,
    use_fit3d: bool = False,
    base_model=None,
    train_first_view_only: bool = False,  # New parameter: train only on first view
    rotate_pred_to_canonical: bool = False,  # Rotate predicted camera-space normals to canonical (view-0) frame in loss
):
    for ep in range(n_epochs):

        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["images"].to(rank)
            plucker = batch["plucker"].to(rank)
            
            # Track if input is 5D for reshaping later
            is_multiview = images.ndim == 5
            if is_multiview:
                B, V = images.shape[:2]
                # Keep GT as 5D initially: [B, V, ...]
                depths_5d = batch["depths"].to(rank)
                snorms_5d = batch["snorms"].to(rank)
                # Flatten to [B*V, ...] for loss computation
                mask = (depths_5d.reshape(B * V, *depths_5d.shape[2:]) > 0)
                target = snorms_5d.reshape(B * V, *snorms_5d.shape[2:])
            else:
                B, V = images.shape[0], 1
                mask = batch["depths"].to(rank) > 0
                target = batch["snorms"].to(rank)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    # Model handles 5D input [B, V, C, H, W] and returns 5D features
                    if skip_camera_pe:
                        images_2d = rearrange(images, 'b s c h w -> (b s) c h w')
                        if use_fit3d:
                            # Extract intermediate layers from FiT3D
                            multilayers = get_multilayers_from_model(model)
                            feats = model.get_intermediate_layers(images_2d, n=multilayers, reshape=True)
                            if isinstance(feats, tuple):
                                feats = list(feats)
                            elif not isinstance(feats, list):
                                feats = [feats]
                        else:
                            feats = model(images_2d)
                        B_, S_ = images.shape[0], images.shape[1]
                        if base_model is not None:
                            base_feats = base_model(images_2d)
                            base_feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in
                                          (base_feats if isinstance(base_feats, list) else [base_feats])]
                        feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in (feats if isinstance(feats, list) else [feats])]
                    else:
                        # DEBUG FIXME PATCH
                        # plucker = torch.zeros_like(plucker)
                        feats = model(images, camera_pe=plucker)
                        if base_model is not None:
                            B_, S_ = images.shape[0], images.shape[1]
                            images_2d_for_base = rearrange(images, 'b s c h w -> (b s) c h w')
                            base_feats = base_model(images_2d_for_base)
                            base_feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in
                                          (base_feats if isinstance(base_feats, list) else [base_feats])]

                    if type(feats) is tuple or type(feats) is list:
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
                    if base_model is not None:
                        if type(base_feats) is tuple or type(base_feats) is list:
                            base_feats = [_f.detach() for _f in base_feats]
                        else:
                            base_feats = base_feats.detach()
            else:
                # Model handles 5D input [B, V, C, H, W] and returns 5D features
                if skip_camera_pe:
                    images_2d = rearrange(images, 'b s c h w -> (b s) c h w')
                    if use_fit3d:
                        multilayers = get_multilayers_from_model(model)
                        feats = model.get_intermediate_layers(images_2d, n=multilayers, reshape=True)
                        if isinstance(feats, tuple):
                            feats = list(feats)
                        elif not isinstance(feats, list):
                            feats = [feats]
                    else:
                        feats = model(images_2d)
                    B_, S_ = images.shape[0], images.shape[1]
                    feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in (feats if isinstance(feats, list) else [feats])]
                else:
                    feats = model(images, camera_pe=plucker)            
            # Reshape features from [B, V, C, H, W] to [B*V, C, H, W] for probe
            if is_multiview:
                if type(feats) is tuple or type(feats) is list:
                    # feats = torch.stack(feats, dim=1)
                    feats = [f for f in feats]
                    #  feats = [f.reshape(B * V, *f.shape[2:]) for f in feats]
                else:
                    feats = feats
                    # feats = feats.reshape(B * V, *feats.shape[2:])

            if base_model is not None:
                # print("feats.shape", feats[0].shape)
                # print("base_feats.shape", base_feats[0].shape)
                for i in range(len(feats)):
                    feats[i] = torch.cat((feats[i], base_feats[i]), dim=-3)


            # Probe expects [B*V, C, H, W]
            pred = probe(feats)

            # Reshape pred to [B*V, C, H, W]
            pred = pred.reshape(B*V, *pred.shape[2:])

            # Interpolate before reshaping back (operates on [B*V, C, H, W])
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")
            
            # Option to train only on first view (like validation)
            # This helps verify coordinate system alignment without multi-view complexity
            if train_first_view_only and is_multiview:
                # Extract first view from each batch item: indices 0, V, 2V, 3V, ... (B-1)*V
                first_view_indices = torch.arange(0, B * V, V, device=pred.device)
                pred = pred[first_view_indices]
                target = target[first_view_indices]
                mask = mask[first_view_indices]

            # If enabled, rotate predicted camera-space normals to the canonical (view-0) frame before loss
            # This makes the probe solve an easier local task (camera-space normals) while supervising in canonical frame.
            if rotate_pred_to_canonical:
                # We need extrinsics to compute R_rel = R_ref_w2c @ R_c2w
                # Expecting 'extrinsics' (w2c) in batch with shape [B, V, 4, 4] when multiview, else [B, 1, 4, 4]
                w2c = batch.get("extrinsics")
                if w2c is None:
                    raise RuntimeError("rotate_pred_to_canonical=True but 'extrinsics' not found in batch")
                w2c = w2c.to(rank)
                if not is_multiview:
                    # Expand to [B, 1, 4, 4] for unified handling
                    w2c = w2c.unsqueeze(1)
                # Compute rotations
                # R_ref_w2c: [B, 3, 3] from view 0
                R_ref_w2c = w2c[:, 0, :3, :3]
                # R_c2w for each view: transpose of w2c rotation
                R_c2w = w2c[:, :, :3, :3].transpose(-1, -2)  # [B, V, 3, 3]
                # R_rel = R_ref_w2c @ R_c2w
                R_rel = torch.einsum('bij,bvjk->bv ik', R_ref_w2c, R_c2w)
                # Flatten to [B*V, 3, 3] to match pred layout
                R_rel = R_rel.reshape(-1, 3, 3)
                # If we filtered to first view only, select corresponding rotations
                if train_first_view_only and is_multiview:
                    first_view_indices = torch.arange(0, B * V, V, device=pred.device)
                    R_rel = R_rel[first_view_indices]
                # Rotate only the first 3 channels (normal); keep extra channels (e.g., uncertainty) intact
                has_uncertainty = pred.shape[1] > 3
                n_pred = pred[:, :3, ...]  # [N, 3, H, W]
                N, _, Ht, Wt = n_pred.shape
                n_pred_flat = n_pred.reshape(N, 3, Ht * Wt)
                # Batched rotation
                n_pred_rot = torch.bmm(R_rel, n_pred_flat)
                n_pred_rot = F.normalize(n_pred_rot, dim=1, eps=1e-6)
                n_pred_rot = n_pred_rot.reshape(N, 3, Ht, Wt)
                if has_uncertainty:
                    pred = torch.cat([n_pred_rot, pred[:, 3:, ...]], dim=1)
                else:
                    pred = n_pred_rot
            
            # Compute loss on 4D tensors [B*V, ...] or [B, ...] if first_view_only
            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(f"{ep} | loss: {_loss:.4f} probe_lr: {pr_lr:.2e}")

        train_loss /= len(train_loader)

        if rank == 0 and valid_loader is not None:
            valid_loss, valid_metrics = validate(model, probe, valid_loader, skip_camera_pe=skip_camera_pe, base_model=None)
            logger.info(f"Final valid loss       | {valid_loss:.4f}")
            for metric in valid_metrics:
                logger.info(f"Final valid {metric:10s} | {valid_metrics[metric]:.4f}")


def validate(model, probe, loader, verbose=True, aggregate=True, skip_camera_pe: bool = False, use_fit3d: bool = False, base_model=None, rotate_pred_to_canonical: bool = False):
    total_loss = 0.0
    metrics = None
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["images"].cuda()
            plucker = batch["plucker"].cuda()
            
            # Track if input is 5D
            is_multiview = images.ndim == 5
            if is_multiview:
                B, V = images.shape[:2]
                # Keep GT as 5D initially: [B, V, ...]
                depths_5d = batch["depths"].cuda()
                snorms_5d = batch["snorms"].cuda()
                # Flatten to [B*V, ...] for loss computation
                mask = (depths_5d.reshape(B * V, *depths_5d.shape[2:]) > 0)
                target = snorms_5d.reshape(B * V, *snorms_5d.shape[2:])
            else:
                B, V = images.shape[0], 1
                mask = batch["depths"].cuda() > 0
                target = batch["snorms"].cuda()

            # Model handles 5D input and returns 5D features
            if skip_camera_pe:
                images_2d = rearrange(images, 'b s c h w -> (b s) c h w')
                if use_fit3d:
                    multilayers = get_multilayers_from_model(model)
                    feats = model.get_intermediate_layers(images_2d, n=multilayers, reshape=True)
                    if isinstance(feats, tuple):
                        feats = list(feats)
                    elif not isinstance(feats, list):
                        feats = [feats]
                else:
                    feats = model(images_2d)
                B_, S_ = images.shape[0], images.shape[1]
                feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in (feats if isinstance(feats, list) else [feats])]
            else:
                feats = model(images, camera_pe=plucker)

            if base_model is not None:
                images_2d = rearrange(images, 'b s c h w -> (b s) c h w')
                base_feats = base_model(images_2d)
                B_, S_ = images.shape[0], images.shape[1]
                base_feats = [rearrange(f, '(b s) c h w -> b s c h w', b=B_, s=S_) for f in (base_feats if isinstance(base_feats, list) else [base_feats])]
            
            # Reshape features from [B, V, C, H, W] to [B*V, C, H, W] for probe
            if is_multiview:
                if type(feats) is tuple or type(feats) is list:
                    # feats = torch.stack(feats, dim=1)
                    feats = [f for f in feats]
                    # feats = [f.reshape(B * V, *f.shape[2:]) for f in feats]
                else:
                    feats = feats
                    # feats = feats.reshape(B * V, *feats.shape[2:])
                if base_model is not None:
                    if type(base_feats) is tuple or type(base_feats) is list:
                        base_feats = [f for f in base_feats]
                    else:
                        base_feats = base_feats

            if base_model is not None:
                for i in range(len(feats)):
                    feats[i] = torch.cat((feats[i], base_feats[i]), dim=-3)


            # Probe expects [B*V, C, H, W]
            pred = probe(feats)

            # Reshape pred to [B*V, C, H, W]
            pred = pred.reshape(B * V, *pred.shape[2:])

            # Interpolate before reshaping back (operates on [B*V, C, H, W])
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")
            
            # Optionally rotate predictions to canonical (view-0) frame before computing loss/metrics
            if rotate_pred_to_canonical:
                w2c = batch.get("extrinsics")
                if w2c is None:
                    raise RuntimeError("rotate_pred_to_canonical=True but 'extrinsics' not found in batch (validate)")
                w2c = w2c.cuda()
                if not is_multiview:
                    w2c = w2c.unsqueeze(1)
                R_ref_w2c = w2c[:, 0, :3, :3]
                R_c2w = w2c[:, :, :3, :3].transpose(-1, -2)
                R_rel = torch.einsum('bij,bvjk->bv ik', R_ref_w2c, R_c2w).reshape(-1, 3, 3)
                has_uncertainty = pred.shape[1] > 3
                n_pred = pred[:, :3, ...]
                N, _, Ht, Wt = n_pred.shape
                n_pred_flat = n_pred.reshape(N, 3, Ht * Wt)
                n_pred_rot = torch.bmm(R_rel, n_pred_flat)
                n_pred_rot = F.normalize(n_pred_rot, dim=1, eps=1e-6)
                n_pred_rot = n_pred_rot.reshape(N, 3, Ht, Wt)
                if has_uncertainty:
                    pred = torch.cat([n_pred_rot, pred[:, 3:, ...]], dim=1)
                else:
                    pred = n_pred_rot
            
            # For validation: only evaluate the first view to avoid counting same image multiple times
            # Extract first view from each batch item: indices 0, V, 2V, 3V, ... (B-1)*V
            # if is_multiview:
            #     first_view_indices = torch.arange(0, B * V, V, device=pred.device)
            #     pred_first = pred[first_view_indices]
            #     target_first = target[first_view_indices]
            #     mask_first = mask[first_view_indices]
            # else:
            #     pred_first = pred
            #     target_first = target
            #     mask_first = mask
            pred_first = pred
            target_first = target
            mask_first = mask
            
            # Compute loss and metrics only on first view
            uncertainty = pred_first.shape[1] > 3
            loss = angular_loss(pred_first, target_first, mask_first, uncertainty_aware=uncertainty)

            total_loss += loss.item()
            batch_metrics = evaluate_surface_norm(pred_first.detach(), target_first, mask_first)
            if metrics is None:
                metrics = {key: [batch_metrics[key]] for key in batch_metrics}
            else:
                for key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

    # aggregate
    total_loss = total_loss / len(loader)
    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def load_dino3d_model(cfg):
    """Load DINO3D model from config, handling both Hydra backbone configs and 3D model configs."""
    from dino3d.models.dino3d import DINO3D
    from dino3d.models.infalted_dino import DINO3D_V2
    from dino3d.checkpointing import CheckPoint
    from peft import LoraConfig, get_peft_model

    # Get config values
    model_name = cfg.get('model_name', 'dinov2_vits14')
    inflated_attn = cfg.get('inflated_attn', False)

    # print(cfg)
    # print(cfg.get('disable_plucker', False))

    # Common parameters
    common_params = dict(
        model_name=model_name,
        output=cfg.get('model_output_type', 'dense'),
        return_multilayer=True,  # Enable multilayer for probe
        stride=cfg.get('stride', 14),
        pe_embedding_strategy=cfg.get('pe_embedding_strategy', 'concat'),
        plucker_emb_dim=cfg.get('plucker_emb_dim', 128),
        disable_plucker=cfg.get('disable_plucker', False),
        use_sa_cls_token=cfg.get('use_sa_cls_token', False),
        disable_dino_pe=cfg.get('disable_dino_pe', False),
        sa_parametric_spatial_conv=cfg.get('sa_parametric_spatial_conv', False),
        spatial_reduction=cfg.get('spatial_reduction', False),
        lora_finetune=cfg.get('lora_finetune', False),
        plucker_mlp=cfg.get('plucker_mlp', False),
        sa_layers=cfg.get('sa_layers', []),
    )
    
    # Create model based on architecture type
    if inflated_attn:
        model = DINO3D_V2(
            **common_params,
            sa_before_spatial=cfg.get('sa_before_spatial', False),
        )
    else:
        model = DINO3D(
            **common_params,
            use_sa_ffn=cfg.get('use_sa_ffn', False),
            epipolar_enabled=cfg.get('use_epipolar_attention', False),
            num_epipolar_samples=cfg.get('num_epipolar_samples', 32),
            disable_3d=cfg.get('disable_3d', False),
        )
    
    # Apply LoRA if needed
    if cfg.get('use_lora', False):
        config = LoraConfig(
            r=cfg.get('lora_r', 16),
            lora_alpha=cfg.get('lora_alpha', 16),
            target_modules=["qkv", "ffn"],
            lora_dropout=0.1,
            bias="none",
        )
        model.vit = get_peft_model(model.vit, config)
    
    # Load checkpoint if specified
    # Construct checkpoint path from exp_directory and exp_name
    exp_directory = cfg.get('exp_directory', None)
    exp_name = cfg.get('exp_name', None)
    checkpoint_name = cfg.get('checkpoint_name', 'best.pth')  # Default to best.pth
    
    if exp_directory and exp_name:
        checkpoint_dir = f"{exp_directory}/{exp_name}/checkpoints/"
        checkpointer = CheckPoint(checkpoint_dir)
        model = checkpointer.load_model(model, checkpoint_name=checkpoint_name)
        logger.info(f"Loaded DINO3D checkpoint from {checkpoint_dir}{checkpoint_name}")
    elif exp_directory or exp_name:
        logger.warning(f"Both exp_directory and exp_name must be specified to load checkpoint. Got exp_directory={exp_directory}, exp_name={exp_name}")
    
    return model


def load_base_dino_model(cfg):
    """Load base 2D DINO model that matches DINO3D data handling.

    Expects a model that can accept 5D inputs [B, V, C, H, W] and optional camera_pe,
    returning 5D features [B, V, C, H, W]. The variant provided in this codebase supports
    the same data structure as DINO3D.
    """
    from dino3d.models.base_dino import DINO

    model = DINO(
        model_name=cfg.get('model_name', 'dinov2_vits14'),
        output=cfg.get('model_output_type', 'dense'),
        return_multilayer=True,  # enable multilayer for probe compatibility
        stride=cfg.get('stride', 14),
    )

    return model


def load_fit3d_model(cfg):
    """Load FiT3D model for evaluation similar to eval_base.

    Uses build_2d_model("dinov2_reg_small") and extracts multilayer features via
    get_intermediate_layers with reshape=True.
    """
    from FiT3D.model_utils import build_2d_model
    
    model = build_2d_model("dinov2_reg_small")
    # attach helper attrs for logging/compat
    setattr(model, 'checkpoint_name', 'FiT3D_dinov2_reg_small')
    # best guess for patch size for dinov2 small
    setattr(model, 'patch_size', 14)
    # Set feature dimensions for the probe (384 for dinov2 small, 4 layers)
    setattr(model, 'feat_dim', [384, 384, 384, 384])
    return model


def get_multilayers_from_model(model):
    """Compute multilayer indices (quarter, half, 3/4, last) from a ViT-like model.

    Tries common attribute names to determine the number of transformer blocks.
    """
    num_layers = None
    found_path = None
    for attr_path in [
        'blocks',
        'vit.blocks',
        'finetuned_model.blocks',
        'finetuned_model.model.blocks',
        'model.blocks',
    ]:
        try:
            obj = model
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            num_layers = len(obj)
            found_path = attr_path
            break
        except Exception as e:
            continue
    
    if num_layers is None:
        # fallback to 12 as a common ViT small depth
        num_layers = 12
        # logger.warning(f"⚠ Could not detect num_layers from model, using fallback: {num_layers}")
    # else:
        # logger.info(f"✓ Detected num_layers={num_layers} from model.{found_path}")
    
    multilayers = [
        num_layers // 4 - 1,
        num_layers // 2 - 1,
        num_layers // 4 * 3 - 1,
        num_layers - 1,
    ]
    # logger.info(f"  Multilayer indices: {multilayers}")
    return multilayers


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)
    
    # Log dataset sizes
    if rank == 0:
        logger.info(f"Dataset configuration:")
        logger.info(f"  Training instances: {len(trainval_loader.dataset)} (sliding windows, stride={cfg.dataset.get('n_dim', 4)})")
        logger.info(f"  Validation instances: {len(test_loader.dataset)} (sliding windows, stride=1 - every image evaluated)")
        logger.info(f"  Batch size: {cfg.batch_size}")
        logger.info(f"  Train iterations per epoch: {len(trainval_loader.dataset) // (cfg.batch_size * world_size)}")
        logger.info(f"  Validation iterations: {len(test_loader.dataset) // cfg.batch_size} (each image evaluated once)")


    # ===== Get models =====
    # Allow selecting base DINO via Hydra flag `eval_base=true` or FiT3D via `eval_fit3d=true`
    use_base = cfg.get('eval_base', False)
    use_fit3d = cfg.get('eval_fit3d', False)
    if use_base:
        model = load_base_dino_model(cfg)
        logger.info("Loaded base DINO model from config (eval_base=true)")
    elif use_fit3d:
        model = load_fit3d_model(cfg)
        logger.info("Loaded FiT3D model for evaluation (eval_fit3d=true)")
    else:
        # Try to use DINO3D loader if 3D model config is present, otherwise use Hydra instantiate
        if 'model_name' in cfg or hasattr(cfg, 'model_name'):
            model = load_dino3d_model(cfg)
            logger.info("Loaded DINO3D model from config")
        else:
            model = instantiate(cfg.backbone)
            logger.info("Loaded model using Hydra instantiate")

    if cfg.get('concat_with_base', False):
        base_model = load_base_dino_model(cfg)
        logger.info("Loaded base DINO model from config - concat with base")
    
    # model.feat_dim is already a list [feat_dim, feat_dim, feat_dim, feat_dim] when return_multilayer=True
    # If it's a single value, convert it to a list
    # Determine feature dims
    if cfg.get('concat_with_base', False):
        feat_dims = [768, 768, 768, 768]
    elif hasattr(model, 'feat_dim'):
        feat_dims = model.feat_dim if isinstance(model.feat_dim, list) else [model.feat_dim] * 4
    elif use_fit3d or use_base:
        # Default to 384 for dinov2 small-like models (multilayer list x4)
        feat_dims = [384, 384, 384, 384]
    else:
        feat_dims = [384, 384, 384, 384]
    
    # Validate multilayer setup
    if isinstance(feat_dims, list):
        logger.info(f"✓ Model configured for multilayer output with feat_dims: {feat_dims}")
        assert len(feat_dims) == 4, f"Expected 4 feature layers, got {len(feat_dims)}"
    else:
        logger.warning(f"⚠ Model is NOT multilayer, feat_dim is single value: {feat_dims}")
    
    probe = instantiate(cfg.probe, feat_dim=feat_dims)

    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    # Build robust model_info for naming/logging regardless of model type
    _ckpt_name = str(getattr(model, 'checkpoint_name', getattr(model, 'model_name', 'base_dino')))
    _patch_size = int(getattr(model, 'patch_size', cfg.get('stride', 14)))
    _layer = str(getattr(model, 'layer', 'N/A'))
    _output = str(getattr(model, 'output', cfg.get('model_output_type', 'dense')))
    model_info = [
        f"{_ckpt_name:40s}",
        f"{_patch_size:2d}",
        f"{_layer:5s}",
        f"{_output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{cfg.optimizer.probe_lr:4.2e}",
        f"{cfg.optimizer.model_lr:4.2e}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces
    if cfg.exp_name:
        exp_name = cfg.exp_name

    # ===== SETUP LOGGING =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"{cfg.exp_directory}/{exp_name}"
        if use_base:
            exp_path = Path(__file__).parent / f"{cfg.exp_directory}/{exp_name}_base"
        if use_fit3d:
            exp_path = Path(__file__).parent / f"{cfg.exp_directory}/{exp_name}_fit3d"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    if cfg.get('concat_with_base', False):
        base_model = base_model.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["images"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if cfg.get('concat_with_base', False):
            base_model = DDP(base_model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if cfg.get('concat_with_base', False):
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()


    lambda_fn = lambda epoch: cosine_decay_linear_warmup(
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        rank=rank,
        world_size=world_size,
        skip_camera_pe=(use_base or use_fit3d),
        use_fit3d=use_fit3d,
        base_model=base_model if cfg.get('concat_with_base', False) else None,
        train_first_view_only=cfg.get('train_first_view_only', False),  # New parameter
        rotate_pred_to_canonical=cfg.get('rotate_pred_to_canonical', False),
        # valid_loader=test_loader,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")
        test_loss, test_metrics = validate(
            model, probe, test_loader,
            skip_camera_pe=(use_base or use_fit3d), use_fit3d=use_fit3d, base_model=base_model if cfg.get('concat_with_base', False) else None,
            rotate_pred_to_canonical=cfg.get('rotate_pred_to_canonical', False),
        )
        logger.info(f"Final test loss       | {test_loss:.4f}")
        for metric in test_metrics:
            logger.info(f"Final test {metric:10s} | {test_metrics[metric]:.4f}")

        # result summary
        model_info = ", ".join(model_info)
        probe_info = ", ".join(probe_info)
        train_info = ", ".join(train_info)
        results = ", ".join([f"{test_metrics[_m]:.4f}" for _m in test_metrics])

        log = f"{timestamp}, {model_info}, {probe_info}, {train_info}, {results} \n"
        with open(f"snorm_results_{test_dset}.log", "a") as f:
            f.write(log)

        # save final model checkpoint (full)
        ckpt_path = exp_path / "ckpt.pth"
        checkpoint = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
            "probe": probe.module.state_dict() if world_size > 1 else probe.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved full checkpoint at {ckpt_path}")
        
        # save probe weights separately in probe3d directory
        probe3d_dir = Path(__file__).parent.parent / cfg.exp_directory / "probe3d"
        probe3d_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe3d_dir / f"{exp_name}_probe.pth"
        probe_checkpoint = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "probe": probe.module.state_dict() if world_size > 1 else probe.state_dict(),
            "test_metrics": test_metrics,
            "model_info": {
                "checkpoint_name": getattr(model, 'checkpoint_name', getattr(model, 'model_name', 'base_dino')),
                "feat_dims": feat_dims,
            }
        }
        torch.save(probe_checkpoint, probe_path)
        logger.info(f"Saved probe weights at {probe_path}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="snorm_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    # If user specified a config_file, load it and merge with cfg
    if 'config_file' in cfg and cfg.config_file:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / cfg.config_file
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            external_cfg = yaml.safe_load(f)
        # Set struct to False to allow new keys, merge, then convert back
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(external_cfg))
        logger.info(f"Merged config from {cfg.config_file}")

    cfg["batch_size"] = 2
    # cfg["n_dim"] = 8

    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
