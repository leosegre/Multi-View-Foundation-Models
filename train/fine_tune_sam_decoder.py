import os
import random
import sys
from pathlib import Path
import math
import contextlib

# Minimal fix: ensure project root is on sys.path so local packages (like `CLIP`)
# can be imported when running scripts from subfolders.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from dino3d.checkpointing import CheckPoint
from tqdm import tqdm
from dino3d.train.train import fix_random_seeds, get_dataloaders, get_train_dataloader, load_config, SafeNamespace
import torch.multiprocessing as mp
from SAM.segment_anything import sam_model_registry
# Original (unmodified) SAM baseline registry
import importlib
# Track BASE_SAM import state for better diagnostics
base_sam_model_registry = None
base_sam_pkg = None
base_sam_import_error = None
try:
    # Import BASE_SAM package (exists in repo root). Some files inside BASE_SAM
    # use absolute imports like `from segment_anything.modeling import ...`.
    base_pkg = importlib.import_module("BASE_SAM")
    base_sam_pkg = base_pkg
    # Provide a stable alias so absolute imports inside BASE_SAM resolve to the
    # loaded package under the name `segment_anything`.
    if "segment_anything" not in sys.modules:
        sys.modules["segment_anything"] = base_pkg
    base_sam_model_registry = getattr(base_pkg, "sam_model_registry", None)
    if base_sam_model_registry is None:
        raise ImportError("BASE_SAM does not expose 'sam_model_registry'")
    # Save package file path for debugging
    base_sam_pkg_file = getattr(base_pkg, "__file__", None)
    print(f"[Baseline SAM] BASE_SAM imported from: {base_sam_pkg_file}")
except Exception as e:
    # If the import failed because BASE_SAM's files do absolute imports like
    # `from segment_anything.modeling import ...`, the initial import can fail
    # (ModuleNotFoundError for 'segment_anything'). Try loading BASE_SAM's
    # __init__.py as the module name 'segment_anything' directly and alias it
    # to 'BASE_SAM'. This ensures absolute imports inside BASE_SAM resolve.
    base_sam_model_registry = None
    base_sam_pkg = None
    base_sam_import_error = e
    err_str = str(e)
    if isinstance(e, ModuleNotFoundError) and "segment_anything" in err_str:
        try:
            import importlib.util
            base_init = os.path.join(str(project_root), "BASE_SAM", "__init__.py")
            spec = importlib.util.spec_from_file_location("segment_anything", base_init)
            module = importlib.util.module_from_spec(spec)
            # Pre-insert into sys.modules so any absolute imports during module
            # execution resolve to this module name
            sys.modules["segment_anything"] = module
            sys.modules["BASE_SAM"] = module
            spec.loader.exec_module(module)
            base_sam_pkg = module
            base_sam_model_registry = getattr(module, "sam_model_registry", None)
            base_sam_pkg_file = getattr(module, "__file__", None)
            print(f"[Baseline SAM] Loaded BASE_SAM by importing as 'segment_anything' from: {base_sam_pkg_file}")
        except Exception as e2:
            base_sam_import_error = e2
            base_sam_model_registry = None
            base_sam_pkg = None
            print(f"[Baseline SAM] Secondary import attempt failed: {e2}")
    else:
        print(f"[Baseline SAM] Import failed ({e}). Falling back to modified SAM for baseline.")
import torchvision
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset
from dino3d.datasets.colmapdata import COLMAPBuilder

def monkey_patch_decoder_for_batching():
    """
    Monkey patch SAM's MaskDecoder.predict_masks to support batched image embeddings.
    
    Original implementation assumes batch=1 and uses repeat_interleave to expand
    for multiple prompts. This breaks when we want to batch across multiple images.
    
    The patched version handles the case where:
    - image_embeddings: [B, C, H, W] (multiple images)
    - sparse_prompt_embeddings: [B*num_prompts, num_sparse, C] (prompts for all images)
    - dense_prompt_embeddings: [B*num_prompts, C, H, W] (dense prompts for all images)
    
    It reshapes to group prompts by image, processes through transformer, then flattens back.
    """
    from SAM.segment_anything.modeling.mask_decoder import MaskDecoder
    import torch
    from typing import Tuple, List
    
    original_predict_masks = MaskDecoder.predict_masks
    
    def batched_predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks with support for batched image embeddings."""
        
        # Detect if we're in batched mode
        B_img = image_embeddings.shape[0]  # Number of images
        B_total = sparse_prompt_embeddings.shape[0]  # Total prompts across all images
        
        # If batch sizes match, use original implementation (single image case)
        if B_img == B_total or B_img == 1:
            return original_predict_masks(
                self, image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings
            )

        # Batched case: multiple images with multiple prompts each
        # Assume prompts are evenly distributed: num_prompts = B_total // B_img
        assert B_total % B_img == 0, f"Total prompts ({B_total}) must be divisible by num images ({B_img})"
        num_prompts_per_image = B_total // B_img

        # Reshape inputs to [B_img, num_prompts, ...]
        sparse_prompt_embeddings = sparse_prompt_embeddings.view(
            B_img, num_prompts_per_image, *sparse_prompt_embeddings.shape[1:]
        )
        dense_prompt_embeddings = dense_prompt_embeddings.view(
            B_img, num_prompts_per_image, *dense_prompt_embeddings.shape[1:]
        )

        # Process each image separately (could be further optimized with grouped batching)
        all_masks = []
        all_iou_pred = []

        for i in range(B_img):
            # Get embeddings for this image
            img_emb = image_embeddings[i:i+1]  # [1, C, H, W]
            sparse_emb = sparse_prompt_embeddings[i]  # [num_prompts, num_sparse, C]
            dense_emb = dense_prompt_embeddings[i]  # [num_prompts, C, H, W]

            # Use original implementation for single image
            masks, iou_pred = original_predict_masks(
                self, img_emb, image_pe, sparse_emb, dense_emb
            )

            all_masks.append(masks)
            all_iou_pred.append(iou_pred)
        masks = torch.cat(all_masks, dim=0)
        iou_pred = torch.cat(all_iou_pred, dim=0)
        
        return masks, iou_pred
    
    # Apply the patch
    MaskDecoder.predict_masks = batched_predict_masks
    print("✓ Monkey patched MaskDecoder.predict_masks for batched processing")


def get_sam_model(args, device='cpu', **kwargs):
    """Initialize SAM model with optional 3D features (spatial adapters and plucker embeddings)"""
    spatial_reduction = kwargs.get('spatial_reduction', getattr(args, 'spatial_reduction', False))
    use_sa_ffn = kwargs.get('use_sa_ffn', getattr(args, 'use_sa_ffn', False))
    plucker_emb_dim = kwargs.get('plucker_emb_dim', getattr(args, 'plucker_emb_dim', None))
    disable_plucker = kwargs.get('disable_plucker', getattr(args, 'disable_plucker', False))
    
    model = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        spatial_reduction=spatial_reduction,
        use_sa_ffn=use_sa_ffn,
        plucker_emb_dim=plucker_emb_dim,
        disable_plucker=disable_plucker,
    )

    def image_features(images, plucker=None):
        return model.image_encoder(images, plucker)

    model.image_features = image_features
    model = model.to(device)
    return model


def get_original_sam_model(args, device='cpu'):
    """Load the original SAM model from BASE_SAM (unmodified baseline).

    Falls back to modified SAM if BASE_SAM isn't available.
    Always kept in float32 and does not support multi-view (expects [B, C, H, W]).
    """
    # Determine which registry we'll use for the baseline and optionally fail-fast
    if base_sam_model_registry is None:
        registry = sam_model_registry
        registry_source = "repo_SAM (modified)"
    else:
        registry = base_sam_model_registry
        registry_source = "BASE_SAM (original)"

    # If user requested strict behavior, fail if original BASE_SAM is not available
    if getattr(args, 'require_original_baseline', False) and registry_source != "BASE_SAM (original)":
        # Include the original import exception (if any) to aid debugging
        cause = getattr(globals().get('base_sam_import_error'), 'args', globals().get('base_sam_import_error'))
        raise RuntimeError(f"--require_original_baseline set but BASE_SAM could not be imported. Cause: {cause}")

    model_type = getattr(args, 'base_model_type', getattr(args, 'model_type', 'vit_b'))
    checkpoint = getattr(args, 'base_checkpoint', getattr(args, 'checkpoint', None))
    if checkpoint is None:
        raise ValueError("Baseline SAM requires --base_checkpoint or --checkpoint.")
    base_model = registry[model_type](checkpoint=checkpoint)
    base_model = base_model.to(device)
    # Ensure eval + freeze
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    # Also print which module file provided the registry when available
    pkg_file = None
    if registry_source == "BASE_SAM (original)":
        pkg_file = getattr(base_sam_pkg, "__file__", None)
    else:
        # repo SAM package file
        try:
            import SAM.segment_anything as repo_sam_pkg
            pkg_file = getattr(repo_sam_pkg, "__file__", None)
        except Exception:
            pkg_file = None
    print(f"[Baseline SAM] Loaded baseline from {registry_source}: model_type={model_type} checkpoint={checkpoint} (module_file={pkg_file})")
    return base_model


def setup_decoder_training_model(model, freeze_prompt_encoder: bool = False):
    """
    Prepare 3D SAM for decoder fine-tuning.

    - Always freeze the image encoder.
    - Mask decoder is trainable.
    - Prompt encoder trainability is configurable (default trainable).

    Args:
        model: The 3D SAM model instance.
        freeze_prompt_encoder: If True, also freeze the prompt encoder.

    Returns:
        The model with updated requires_grad flags.
    """
    # Freeze entire encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    # Prompt encoder: configurable
    for p in model.prompt_encoder.parameters():
        p.requires_grad = not freeze_prompt_encoder

    # Unfreeze the decoder
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    return model


def _count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def log_and_verify_trainable_params(model, optimizer, prefix: str = "Model", writer: SummaryWriter = None, step: int = 0):
    """Print and sanity-check which params are trainable and in the optimizer.

    - Ensures baseline params are not present (we pass only 3D model to the optimizer).
    - Ensures no encoder params are in the optimizer.
    - Logs counts to console and TensorBoard.
    """
    enc_total, enc_train = _count_params(model.image_encoder)
    dec_total, dec_train = _count_params(model.mask_decoder)
    pe_total, pe_train = _count_params(model.prompt_encoder)

    print("\n" + "-" * 60)
    print(f"{prefix} trainable parameter check:")
    print("-" * 60)
    print(f"Image Encoder:   {enc_train:>12,} / {enc_total:>12,} trainable (should be 0 trainable)")
    print(f"Mask Decoder:    {dec_train:>12,} / {dec_total:>12,} trainable")
    print(f"Prompt Encoder:  {pe_train:>12,} / {pe_total:>12,} trainable")

    # Optimizer parameter set
    opt_param_ids = set()
    for i, g in enumerate(optimizer.param_groups):
        params = g.get('params', [])
        count = sum(p.numel() for p in params)
        print(f"Param group {i}: lr={g.get('lr', None)} wd={g.get('weight_decay', None)} params={count:,}")
        opt_param_ids.update(id(p) for p in params)

    trainable_ids = set(id(p) for p in model.parameters() if p.requires_grad)
    enc_ids = set(id(p) for p in model.image_encoder.parameters())

    extra_in_optim = opt_param_ids - trainable_ids
    missing_in_optim = trainable_ids - opt_param_ids
    enc_in_optim = opt_param_ids & enc_ids

    if enc_in_optim:
        print(f"[WARN] Found {len(enc_in_optim)} encoder parameters present in optimizer! They should be frozen.")
    if extra_in_optim:
        print(f"[WARN] Optimizer contains {len(extra_in_optim)} params that are not marked trainable.")
    if missing_in_optim:
        print(f"[WARN] {len(missing_in_optim)} trainable params are missing from optimizer.")
    if not (enc_in_optim or extra_in_optim or missing_in_optim):
        print("✓ Optimizer contains exactly the trainable params, encoder excluded.")

    if writer is not None:
        writer.add_scalar(f"{prefix}/encoder_trainable_frac", float(enc_train) / float(max(1, enc_total)), step)
        writer.add_scalar(f"{prefix}/decoder_trainable_frac", float(dec_train) / float(max(1, dec_total)), step)
        writer.add_scalar(f"{prefix}/prompt_trainable_frac", float(pe_train) / float(max(1, pe_total)), step)

    return {
        'enc_total': enc_total, 'enc_train': enc_train,
        'dec_total': dec_total, 'dec_train': dec_train,
        'pe_total': pe_total, 'pe_train': pe_train,
        'enc_in_optim': len(enc_in_optim),
        'extra_in_optim': len(extra_in_optim),
        'missing_in_optim': len(missing_in_optim),
    }


class MaskConsistencyLoss(nn.Module):
    """
    Loss for comparing masks generated by base model vs 3D model.
    Uses feature-based prompts to generate masks and compares them.
    """
    def __init__(self, loss_type='dice_bce'):
        super().__init__()
        self.loss_type = loss_type
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Compute Dice loss between predicted and target masks"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def bce_loss(self, pred, target):
        """Binary cross-entropy loss"""
        # Target masks coming from the base model are logits as well.
        # BCEWithLogits expects targets in [0, 1] (probabilities/soft labels).
        # Convert target logits to probabilities to avoid invalid/negative losses.
        target_prob = torch.sigmoid(target)
        return F.binary_cross_entropy_with_logits(pred, target_prob)
    
    def mse_loss(self, pred, target):
        """MSE loss between logits (direct comparison without sigmoid)"""
        return F.mse_loss(pred, target)
    
    def forward(self, pred_masks, target_masks):
        """
        Args:
            pred_masks: masks from 3D model [B, N, H, W]
            target_masks: masks from base model [B, N, H, W]
        """
        if self.loss_type == 'dice':
            return self.dice_loss(torch.sigmoid(pred_masks), torch.sigmoid(target_masks))
        elif self.loss_type == 'bce':
            return self.bce_loss(pred_masks, target_masks)
        elif self.loss_type == 'dice_bce':
            dice = self.dice_loss(torch.sigmoid(pred_masks), torch.sigmoid(target_masks))
            bce = self.bce_loss(pred_masks, target_masks)
            return dice + bce
        elif self.loss_type == 'mse':
            return self.mse_loss(pred_masks, target_masks)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def generate_masks_from_features(model, images, image_embeddings, num_prompts=10, device='cuda', chunk_size=None, return_points: bool = False, autocast_ctx=None):
    """Generate masks from image & encoder features using point prompts.

    Now uses monkey-patched decoder that supports batched processing across
    multiple images with multiple prompts each.

    Args:
        model: SAM model (with patched decoder)
        images: Tensor [B, C, H, W] or [B, S, C, H, W]
        image_embeddings: Encoder features matching images
        num_prompts: Number of point prompts (laid out on sqrt grid)
        device: CUDA/CPU device
        chunk_size: Unused (kept for API compatibility)

    Returns:
        masks: Tensor [B_flat, num_prompts, H_mask, W_mask]
    """
    # Flatten multi-view dimension if present
    if len(images.shape) == 5:
        B, S, C, H, W = images.shape
        images = images.reshape(B * S, C, H, W)
        if len(image_embeddings.shape) == 5:
            image_embeddings = image_embeddings.reshape(B * S, *image_embeddings.shape[2:])
    B_flat = images.shape[0]

    # Setup autocast context default
    if autocast_ctx is None:
        autocast_ctx = contextlib.nullcontext()

    # Resize encoder features to match the spatial size expected by the prompt encoder's dense PE.
    # This avoids shape mismatches when image size/config changes (e.g., 512->1024 or already 1024).
    # Ensure get_dense_pe runs under autocast so its internal matmuls use the same
    # dtype as the model buffers/parameters (avoids float vs bfloat16 mismatches).
    with autocast_ctx:
        pe = model.prompt_encoder.get_dense_pe()
    pe_h, pe_w = pe.shape[-2], pe.shape[-1]
    emb_h, emb_w = image_embeddings.shape[-2], image_embeddings.shape[-1]
    if (emb_h, emb_w) != (pe_h, pe_w):
        image_embeddings_resized = F.interpolate(
            image_embeddings,
            size=(pe_h, pe_w),
            mode='bilinear',
            align_corners=False,
        )
    else:
        image_embeddings_resized = image_embeddings

    # Build a grid with enough points (ceil(sqrt(N))) and use only the first N.
    # IMPORTANT: prompt coordinates passed to the PromptEncoder.forward_with_coords
    # must be in the pixel coordinate system expected by the prompt encoder
    # (prompt_encoder.input_image_size). Use that size to build the grid so
    # prompts align with the encoder's positional encoding. Later, when
    # returning `used_points`, rescale them back to the original image pixel
    # coordinates so visualization overlays align with the input images.
    side_prompts = int(math.ceil(num_prompts ** 0.5))
    orig_img_h, orig_img_w = images.shape[-2], images.shape[-1]

    # Get the input image size expected by the prompt encoder (H, W).
    try:
        enc_input_h, enc_input_w = model.prompt_encoder.input_image_size
    except Exception:
        # Fallback to the original image size if the prompt encoder does not
        # expose input_image_size (shouldn't happen for SAM implementations).
        enc_input_h, enc_input_w = orig_img_h, orig_img_w

    h_step = max(1, enc_input_h // side_prompts)
    w_step = max(1, enc_input_w // side_prompts)
    grid_coords = [(i, j) for i in range(side_prompts) for j in range(side_prompts)][:num_prompts]

    # Detect model dtype (could be float32 or bfloat16)
    model_dtype = next(model.parameters()).dtype
    
    # Generate point coords for all prompts: [num_prompts, 1, 2]
    # Use same dtype as model to avoid dtype mismatch in prompt encoder
    point_coords_list = []
    for i, j in grid_coords:
        # Coordinates are generated in the prompt encoder's input-image pixel
        # coordinate system (enc_input_w x enc_input_h).
        point_coords_list.append([j * w_step + w_step // 2, i * h_step + h_step // 2])
    point_coords_grid = torch.tensor(point_coords_list, dtype=model_dtype, device=device)  # [num_prompts, 2]
    point_coords_grid = point_coords_grid.unsqueeze(1)  # [num_prompts, 1, 2]

    # Replicate prompts for all images: [B_flat, num_prompts, 1, 2]
    point_coords_batch = point_coords_grid.unsqueeze(0).expand(B_flat, -1, -1, -1)
    point_labels_batch = torch.ones((B_flat, num_prompts, 1), dtype=torch.long, device=device)
    
    # Flatten to [B_flat * num_prompts, 1, 2] for decoder
    point_coords_flat = point_coords_batch.reshape(B_flat * num_prompts, 1, 2)
    point_labels_flat = point_labels_batch.reshape(B_flat * num_prompts, 1)
    
    # Replicate image embeddings for each prompt: [B_flat * num_prompts, C, H, W]
    image_embeddings_repeated = image_embeddings_resized.unsqueeze(1).repeat(1, num_prompts, 1, 1, 1)
    image_embeddings_flat = image_embeddings_repeated.reshape(B_flat * num_prompts, *image_embeddings_resized.shape[1:])
    
    # Forward pass through prompt encoder (batch = B_flat * num_prompts)
    # Forward pass through prompt encoder (batch = B_flat * num_prompts)
    # Some SAM implementations (the patched repo SAM) support a batched
    # mask decoder that accepts image_embeddings for multiple images and
    # flattened prompt embeddings for all prompts. The original BASE_SAM
    # decoder expects a single image at a time. Detect which one we're
    # dealing with and adapt accordingly.
    decoder_fn_name = getattr(model.mask_decoder.predict_masks, "__name__", "predict_masks")
    if decoder_fn_name == "batched_predict_masks":
        # Batched decoder available: compute embeddings for all prompts at once
        # Compute prompt embeddings and decoder predictions under autocast
        with autocast_ctx:
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords_flat, point_labels_flat),
                boxes=None,
                masks=None,
            )

            # Forward pass through patched mask decoder (handles batched images)
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings_resized,  # [B_flat, C, H, W]
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,  # [B_flat * num_prompts, ...]
                dense_prompt_embeddings=dense_embeddings,  # [B_flat * num_prompts, ...]
                multimask_output=False,
            )

        # Reshape from [B_flat * num_prompts, 1, H, W] -> [B_flat, num_prompts, H, W]
        masks = low_res_masks[:, 0].reshape(B_flat, num_prompts, *low_res_masks.shape[2:])
    else:
        # Original (unpatched) decoder: run per-image to avoid the repeat_interleave mismatch
        masks_list = []
        for bi in range(B_flat):
            # Build per-image point coords: [num_prompts, 1, 2]
            pts = point_coords_grid.clone().to(image_embeddings_resized.device)
            pts = pts.unsqueeze(0).expand(1, -1, -1, -1).reshape(num_prompts, 1, 2)
            labs = torch.ones((num_prompts, 1), dtype=torch.long, device=image_embeddings_resized.device)

            with autocast_ctx:
                sparse_emb_i, dense_emb_i = model.prompt_encoder(points=(pts, labs), boxes=None, masks=None)

                # Call decoder with single image embedding
                low_res_i, _ = model.mask_decoder(
                    image_embeddings=image_embeddings_resized[bi:bi+1],
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb_i,
                    dense_prompt_embeddings=dense_emb_i,
                    multimask_output=False,
                )

            # low_res_i: [num_prompts, 1, H, W] -> convert to [1, num_prompts, H, W]
            masks_i = low_res_i[:, 0].unsqueeze(0)
            masks_list.append(masks_i)

        masks = torch.cat(masks_list, dim=0)
    
    if return_points:
        # Return exact point coordinates used for each flattened image: [B_flat, num_prompts, 2]
        used_points = point_coords_batch.squeeze(2)  # remove the singleton num_points dim
        # used_points are currently in the prompt-encoder input-image pixel
        # coordinates (enc_input_h, enc_input_w). Rescale to original image
        # pixel coordinates so overlays line up with `images` supplied by the
        # caller.
        if (enc_input_h, enc_input_w) != (orig_img_h, orig_img_w):
            scale_x = float(orig_img_w) / float(enc_input_w)
            scale_y = float(orig_img_h) / float(enc_input_h)
            used_points = used_points.clone().to(torch.float32)
            used_points[..., 0] = used_points[..., 0] * scale_x
            used_points[..., 1] = used_points[..., 1] * scale_y
        return masks, used_points
    return masks


def run_decoder_iter(model_3d, base_model, data, mask_loss_fn, device='cuda', num_prompts=10, dtype=torch.float32):
    """
    Run one iteration of decoder training.
    
    Pipeline:
    - Base model: 1024x1024 image -> encoder -> features (e.g., 64x64) -> decoder
    - 3D model: 1024x1024 image -> downsample to 512x512 -> encoder -> features (e.g., 32x32)
                -> upsample features x2 (to 64x64) -> decoder gets same resolution as base
    - Both models: decoder and prompt encoder work with 1024x1024 image coordinates
    
    Args:
        model_3d: 3D SAM model with pre-trained encoder
        base_model: base SAM model (frozen)
        data: batch of data
        mask_loss_fn: mask comparison loss function
        device: device to run on
        num_prompts: number of prompts for mask generation
        dtype: data type for computation (float32 or bfloat16)
    
    Returns:
        loss: computed loss value
    """
    # Keep input tensors in float32 and rely on autocast to cast ops to
    # bfloat16 where supported. Feeding float32 inputs into autocast is
    # the recommended pattern and avoids LayerNorm / parameter dtype
    # mismatches while still allowing most ops to run in bfloat16.
    images = data['images'].to(device, dtype=torch.float32)

    # Get plucker embeddings if available. Keep as float32 for encoder.
    plucker = data.get('plucker', None)
    if plucker is not None:
        plucker = plucker.to(device, dtype=torch.float32)
        
    # Forward pass through base model encoder (also expects [B, S, C, H, W])
    # Note: Base model encoder was also modified to handle multi-view format
    # Decide which autocast context to use (CUDA vs CPU). If bfloat16 was
    # requested for training, use autocast to run the encoder in mixed
    # precision. Otherwise use a nullcontext.
    use_autocast = (dtype == torch.bfloat16)
    torch_dev = torch.device(device)
    if use_autocast:
        if torch_dev.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast'):
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            # CPU autocast may exist on newer PyTorch versions as
            # torch.cpu.amp.autocast; fall back to nullcontext if not available.
            cpu_amp = getattr(torch, 'cpu', None)
            if cpu_amp and hasattr(cpu_amp, 'amp') and hasattr(cpu_amp.amp, 'autocast'):
                autocast_ctx = torch.cpu.amp.autocast(dtype=torch.bfloat16)
            else:
                autocast_ctx = contextlib.nullcontext()
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.no_grad():
        # Pipeline for 3D model:
        # 1. Downsample 1024x1024 images to 512x512 for encoder
        # 2. Run encoder to get features (e.g., 32x32)
        # 3. Upsample features x2 to match base model's feature resolution (e.g., 64x64)
        
        # Downsample images to 512x512 for 3D encoder
        if images.ndim == 5:
            B, S, C, H, W = images.shape
            images_flat_3d = images.view(B * S, C, H, W)
        else:
            images_flat_3d = images
        
        # Downsample to 512x512 (half resolution)
        images_512 = F.interpolate(
            images_flat_3d,
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back to multi-view if needed for encoder
        if images.ndim == 5:
            images_512 = images_512.view(B, S, C, 512, 512)
        
        # Run 3D encoder on 512x512 images
        with autocast_ctx:
            image_embeddings_3d_lowres = model_3d.image_encoder(images_512, plucker)
        
        # Upsample encoder features x2 to match base model's feature resolution
        # Flatten multi-view if present for upsampling
        if len(image_embeddings_3d_lowres.shape) == 5:
            B_emb, S_emb, C_emb, H_emb, W_emb = image_embeddings_3d_lowres.shape
            embeddings_flat = image_embeddings_3d_lowres.view(B_emb * S_emb, C_emb, H_emb, W_emb)
        else:
            embeddings_flat = image_embeddings_3d_lowres
            C_emb, H_emb, W_emb = image_embeddings_3d_lowres.shape[1:]
        
        # Upsample features x2 (e.g., 32x32 -> 64x64)
        image_embeddings_3d = F.interpolate(
            embeddings_flat,
            scale_factor=2.0,
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back to multi-view if needed
        if len(image_embeddings_3d_lowres.shape) == 5:
            image_embeddings_3d = image_embeddings_3d.view(
                B_emb, S_emb, C_emb, H_emb * 2, W_emb * 2
            )
        
        # print(f"[Pipeline] 3D encoder: {images_512.shape} -> features {image_embeddings_3d_lowres.shape} -> upsampled {image_embeddings_3d.shape}")
        
        # Flatten multi-view into batch for baseline if necessary: [B,S,C,H,W] -> [B*S,C,H,W]
        if images.ndim == 5:
            B, S, C, H, W = images.shape
            base_images = images.view(B * S, C, H, W)
        else:
            base_images = images
        # Simplified baseline path: upsample images by x2 (e.g., 512->1024)
        # and feed directly into the baseline encoder without extra preprocessing.
        # Keep float32 for encoder inputs to avoid dtype issues.
        base_images = base_images.to(device=base_images.device, dtype=torch.float32)
        preprocessed = F.interpolate(
            base_images, scale_factor=2.0, mode='bilinear', align_corners=False
        )
        # Run baseline encoder under the same autocast context to keep
        # behavior consistent when using bfloat16.
        with autocast_ctx:
            image_embeddings_base = base_model.image_encoder(preprocessed)
        
        # print(f"[Pipeline] Base encoder: {preprocessed.shape} -> features {image_embeddings_base.shape}")
    
    # NOTE: We keep encoder outputs in original multi-view shape and only
    # flatten inside generate_masks_from_features, which handles per-sample iteration.
    # IMPORTANT: Pass original 1024x1024 images to generate_masks_from_features so
    # the decoder and prompt encoder work at 1024x1024 resolution, even though
    # the 3D encoder saw 512x512 images (features were upsampled to match).
    masks_3d_all, points_3d = generate_masks_from_features(
        model_3d,
        images,  # Original 1024x1024 images
        image_embeddings_3d,  # Upsampled features (same resolution as base model)
        num_prompts=num_prompts,
        device=device,
        return_points=True,
        autocast_ctx=autocast_ctx,
    )
    # If multi-view, reshape to [B,S,P,H,W] then flatten views to align with baseline batch [B*S,P,H,W]
    if images.ndim == 5:
        B, S = images.shape[0], images.shape[1]
        P = masks_3d_all.shape[1]
        Hm, Wm = masks_3d_all.shape[2], masks_3d_all.shape[3]
        masks_3d_all = masks_3d_all.view(B, S, P, Hm, Wm)
        masks_3d = masks_3d_all.view(B * S, P, Hm, Wm)
    else:
        masks_3d = masks_3d_all

    with torch.no_grad():
        # Baseline model expects single-view images & its own embeddings
        masks_base, points_base = generate_masks_from_features(
            base_model,
            base_images,
            image_embeddings_base,
            num_prompts=num_prompts,
            device=device,
            return_points=True,
            autocast_ctx=autocast_ctx,
        )
    
    # Sanity check: verify prompts match (should be identical grids) - run once at start
    if not hasattr(run_decoder_iter, '_debug_done'):
        run_decoder_iter._debug_done = True
        print(f"\n[Prompt Verification]")
        print(f"  3D model prompts shape: {points_3d.shape}")
        print(f"  Base model prompts shape: {points_base.shape}")
        print(f"  3D prompts first image, first 3: {points_3d[0, :3]}")
        print(f"  Base prompts first image, first 3: {points_base[0, :3]}")
        if not torch.allclose(points_3d, points_base, atol=1.0):
            print(f"  ⚠️  WARNING: Prompts don't match!")
            print(f"  Max diff: {(points_3d - points_base).abs().max().item()}")
        else:
            print(f"  ✓ Prompts match exactly")
        
        # Also verify mask shapes match
        print(f"  3D masks shape: {masks_3d.shape}")
        print(f"  Base masks shape: {masks_base.shape}")
        if masks_3d.shape != masks_base.shape:
            print(f"  ⚠️  ERROR: Mask shapes don't match!")
        else:
            print(f"  ✓ Mask shapes match")
    
    # Compute loss
    loss = mask_loss_fn(masks_3d, masks_base)
    
    return loss, masks_3d, masks_base, points_3d, points_base


def load_checkpoint_with_resize(checkpointer, model, optimizer, lr_scheduler):
    """
    Load checkpoint with special handling for pos_embed size mismatch.
    Returns the loaded model, global_step, and epoch for resuming training.
    
    This loads the ENTIRE model (encoder + decoder + prompt_encoder),
    not just the encoder. The decoder is the main component being trained.
    """
    latest_path = os.path.join(checkpointer.dir, "latest.pth")
    global_step = 0
    epoch = 0
    
    if os.path.exists(latest_path):
        print("Loading checkpoint from fine-tuning...")
        states = torch.load(latest_path)
        
        # Handle pos_embed size mismatch if present
        if "model" in states:
            state_dict = states["model"]
            
            if 'image_encoder.pos_embed' in state_dict:
                checkpoint_pos_embed = state_dict['image_encoder.pos_embed']
                model_pos_embed = model.image_encoder.pos_embed
                
                if checkpoint_pos_embed.shape != model_pos_embed.shape:
                    print(f"Resizing pos_embed from {checkpoint_pos_embed.shape} to {model_pos_embed.shape}")
                    # Interpolate position embedding to match current model size
                    checkpoint_pos_embed_resized = F.interpolate(
                        checkpoint_pos_embed.permute(0, 3, 1, 2),
                        size=(model_pos_embed.shape[1], model_pos_embed.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    state_dict['image_encoder.pos_embed'] = checkpoint_pos_embed_resized
            
            # Load entire model state (encoder + decoder + prompt_encoder)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded full model state dict with {len(state_dict)} keys")
            
            # Print what components were loaded
            encoder_keys = [k for k in state_dict.keys() if k.startswith('image_encoder')]
            decoder_keys = [k for k in state_dict.keys() if k.startswith('mask_decoder')]
            prompt_keys = [k for k in state_dict.keys() if k.startswith('prompt_encoder')]
            print(f"  - Image Encoder: {len(encoder_keys)} keys")
            print(f"  - Mask Decoder: {len(decoder_keys)} keys")
            print(f"  - Prompt Encoder: {len(prompt_keys)} keys")
        
        if "n" in states:
            global_step = states["n"]
            print(f"✓ Resuming from step {global_step}")
        
        # Note: epoch might not be in checkpoint, that's ok
        if "epoch" in states:
            epoch = states["epoch"] + 1  # Start from next epoch
            print(f"✓ Resuming from epoch {epoch}")
        
        if "optimizer" in states:
            try:
                optimizer.load_state_dict(states["optimizer"])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")
        
        if "lr_scheduler" in states:
            try:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
                print("✓ Loaded lr_scheduler state")
            except Exception as e:
                print(f"Failed to load lr_scheduler state: {e}")
    else:
        print("No checkpoint found, starting from scratch")
    
    return model, global_step, epoch


def visualize_masks_to_tensorboard(writer, images, masks_3d, masks_base, global_step, prefix='train', max_images=4, max_prompts=4, point_coords_3d: torch.Tensor = None, point_coords_base: torch.Tensor = None):
    """
    Visualize mask predictions from 3D and base models side-by-side in TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        images: Input images [B, S, C, H, W] or [B, C, H, W]
        masks_3d: Masks from 3D model [B_flat, num_prompts, H_mask, W_mask] (logits)
        masks_base: Masks from base model [B_flat, num_prompts, H_mask, W_mask] (logits)
        global_step: Current training step
        prefix: Prefix for tag (train/eval)
        max_images: Maximum number of images to visualize
        max_prompts: Maximum number of prompts to visualize per image
    """
    # Ensure all tensors are on the same device and in float32 for visualization
    common_device = masks_3d.device

    # Flatten images if multi-view
    if len(images.shape) == 5:
        B, S, C, H, W = images.shape
        images_flat = images.reshape(B * S, C, H, W)
    else:
        images_flat = images
    # Move images to common device and float32 for safe arithmetic
    images_flat = images_flat.to(common_device, dtype=torch.float32)
    masks_3d = masks_3d.to(common_device)
    masks_base = masks_base.to(common_device)
    
    B_flat = min(images_flat.shape[0], max_images)
    num_prompts = min(masks_3d.shape[1], max_prompts)
    
    # Convert logits to probabilities
    masks_3d_prob = masks_3d[:B_flat, :num_prompts].to(torch.float32)  # [B_flat, num_prompts, H, W]
    masks_base_prob = masks_base[:B_flat, :num_prompts].to(torch.float32)  # [B_flat, num_prompts, H, W]
    
    # Upsample masks to match image resolution for better visualization
    target_size = (images_flat.shape[-2], images_flat.shape[-1])  # Original image size
    # Upsample masks to image resolution using nearest neighbor to preserve
    # exact mask coverage (avoids small, blurred mask blobs). Then binarize
    # at 0.5 for crisp overlays.
    # Defensive handling: detect swapped H/W or unexpected mask spatial sizes
    # and correct before upsampling so overlays align with images.
    mask_h = masks_3d_prob.shape[2]
    mask_w = masks_3d_prob.shape[3]
    img_h, img_w = target_size

    if (mask_h, mask_w) != (img_h, img_w):
        # If H/W are swapped compared to the image, permute last two dims
        if (mask_h, mask_w) == (img_w, img_h):
            masks_3d_prob = masks_3d_prob.permute(0, 1, 3, 2)
            masks_base_prob = masks_base_prob.permute(0, 1, 3, 2)
            mask_h, mask_w = masks_3d_prob.shape[2], masks_3d_prob.shape[3]

    masks_3d_up = F.interpolate(
        masks_3d_prob.reshape(B_flat * num_prompts, 1, *masks_3d_prob.shape[2:]),
        size=target_size,
        mode='nearest'
    ).reshape(B_flat, num_prompts, *target_size)

    masks_base_up = F.interpolate(
        masks_base_prob.reshape(B_flat * num_prompts, 1, *masks_base_prob.shape[2:]),
        size=target_size,
        mode='nearest'
    ).reshape(B_flat, num_prompts, *target_size)

    # Binarize for overlay clarity (keep probabilities for the statistics below)
    masks_3d_up = (masks_3d_up > 0.).to(torch.float32)
    masks_base_up = (masks_base_up > 0.).to(torch.float32)
    
    # Normalize images for visualization [0, 1]
    images_vis = images_flat[:B_flat].clone()
    for i in range(3):  # Normalize each channel
        images_vis[:, i] = (images_vis[:, i] - images_vis[:, i].min()) / (images_vis[:, i].max() - images_vis[:, i].min() + 1e-8)
    
    # Helper: draw a small colored point (circle) on top of an image tensor [C,H,W]
    def _draw_point(img_chw: torch.Tensor, x: int, y: int, radius: int = 6, color=(0.0, 1.0, 0.0), alpha: float = 0.85):
        H, W = img_chw.shape[-2], img_chw.shape[-1]
        x = int(max(0, min(W - 1, x)))
        y = int(max(0, min(H - 1, y)))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return img_chw
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=img_chw.device),
            torch.arange(x0, x1, device=img_chw.device),
            indexing='ij'
        )
        mask = ((yy - y) ** 2 + (xx - x) ** 2) <= (radius ** 2)
        mask = mask.to(img_chw.dtype)
        color_t = torch.tensor(color, device=img_chw.device, dtype=img_chw.dtype).view(3, 1, 1)
        region = img_chw[:, y0:y1, x0:x1]
        img_chw[:, y0:y1, x0:x1] = torch.clamp(region * (1 - alpha * mask) + color_t * (alpha * mask), 0.0, 1.0)
        return img_chw

    # Create visualization grid for each prompt
    for prompt_idx in range(num_prompts):
        vis_rows = []
        # If we were not given explicit coords, build a fallback grid at current resolution
        fallback_pts = None
        if point_coords_3d is None and point_coords_base is None:
            side_prompts = int(math.ceil(num_prompts ** 0.5))
            h_step = max(1, target_size[0] // side_prompts)
            w_step = max(1, target_size[1] // side_prompts)
            grid_coords = [(i, j) for i in range(side_prompts) for j in range(side_prompts)][:num_prompts]
            gi, gj = grid_coords[prompt_idx]
            fallback_pts = (int(gj * w_step + w_step // 2), int(gi * h_step + h_step // 2))
        
        for img_idx in range(B_flat):
            img = images_vis[img_idx]  # [C, H, W]
            # masks_3d_up and masks_base_up have already been upsampled to the
            # image resolution (target_size) earlier in this function, so use
            # those for overlays to ensure masks and image match.
            mask_3d = masks_3d_up[img_idx, prompt_idx:prompt_idx+1]  # [1, H, W] at image res
            mask_base = masks_base_up[img_idx, prompt_idx:prompt_idx+1]  # [1, H, W] at image res

            # Create overlay: image with mask as red channel
            overlay_3d = img.clone()
            overlay_3d[0] = torch.clamp(overlay_3d[0] + mask_3d[0] * 0.5, 0, 1)  # Add red

            overlay_base = img.clone()
            overlay_base[0] = torch.clamp(overlay_base[0] + mask_base[0] * 0.5, 0, 1)  # Add red
            
            # Choose per-image points from provided coords (if any)
            if point_coords_3d is not None and img_idx < point_coords_3d.shape[0]:
                pt_x_3d = int(point_coords_3d[img_idx, prompt_idx, 0].item())
                pt_y_3d = int(point_coords_3d[img_idx, prompt_idx, 1].item())
            else:
                pt_x_3d, pt_y_3d = fallback_pts if fallback_pts is not None else (0, 0)
            if point_coords_base is not None and img_idx < point_coords_base.shape[0]:
                pt_x_base = int(point_coords_base[img_idx, prompt_idx, 0].item())
                pt_y_base = int(point_coords_base[img_idx, prompt_idx, 1].item())
            else:
                pt_x_base, pt_y_base = pt_x_3d, pt_y_3d

            # Draw points (green=3D, blue=baseline)
            # Points are in image pixel coordinates — draw them directly on the
            # full-resolution image and overlays.
            img_with_pt = _draw_point(img.clone(), pt_x_3d, pt_y_3d, color=(0.0, 1.0, 0.0))
            img_with_pt = _draw_point(img_with_pt, pt_x_base, pt_y_base, color=(0.0, 0.4, 1.0))
            overlay_base_with_pt = _draw_point(overlay_base.clone(), pt_x_base, pt_y_base, color=(0.0, 0.4, 1.0))
            overlay_3d_with_pt = _draw_point(overlay_3d.clone(), pt_x_3d, pt_y_3d, color=(0.0, 1.0, 0.0))

            # Create difference map (3D - base)
            diff = (mask_3d - mask_base).abs()
            diff_vis = diff.repeat(3, 1, 1)  # Convert to RGB
            
            # Stack: [image, base_overlay, 3d_overlay, mask_base, mask_3d, diff]
            row = torch.cat([
                img_with_pt,
                overlay_base_with_pt,
                overlay_3d_with_pt,
                mask_base.repeat(3, 1, 1),
                mask_3d.repeat(3, 1, 1),
                diff_vis
            ], dim=2)  # Concatenate horizontally
            
            vis_rows.append(row)
        
        # Stack all images vertically
        grid = torch.stack(vis_rows, dim=0)  # [B_flat, C, H, W*6]
        
        # Make grid
        grid_img = torchvision.utils.make_grid(grid, nrow=1, padding=2, normalize=False)
        # Move to CPU for TensorBoard per-prompt
        writer.add_image(f'{prefix}/masks_prompt_{prompt_idx}', grid_img.detach().cpu(), global_step)
    
    # Also log some statistics
    writer.add_scalar(f'{prefix}/mask_3d_mean', masks_3d_prob.mean().item(), global_step)
    writer.add_scalar(f'{prefix}/mask_base_mean', masks_base_prob.mean().item(), global_step)
    writer.add_scalar(f'{prefix}/mask_diff_mean', (masks_3d_prob - masks_base_prob).abs().mean().item(), global_step)


def train(args):
    fix_random_seeds(args.seed)
    # Determine device and dtype first
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Setup dtype for mixed precision
    use_bfloat16 = getattr(args, 'use_bfloat16', True)
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    if use_bfloat16:
        print("✓ Using bfloat16 for training")
    else:
        print("Using float32 for training")

    # Initialize baseline SAM (original) BEFORE monkey patch so it remains unmodified
    print("Initializing ORIGINAL baseline SAM model...")
    base_model = get_original_sam_model(args, device=device)
    # Cast only if user explicitly forces baseline to bfloat16 (optional future arg)
    if getattr(args, 'baseline_use_bfloat16', False):
        base_model = base_model.to(dtype=dtype)

    # Initialize 3D model with pre-trained encoder (modified SAM)
    print("Initializing 3D SAM model with pre-trained encoder...")
    model_3d = get_sam_model(args, device=device)
    if use_bfloat16:
        model_3d = model_3d.to(dtype=dtype)

    # Apply monkey patch AFTER baseline init so only modified SAM decoders are altered
    monkey_patch_decoder_for_batching()
    
    # Create experiment directory
    if not os.path.exists(args.exp_directory):
        os.makedirs(args.exp_directory)
    os.makedirs(f'{args.exp_directory}/{args.exp_name}', exist_ok=True)
    curr_exp_dir = f'{args.exp_directory}/{args.exp_name}'
    checkpoint_dir = f"{curr_exp_dir}/checkpoints/"
    writer = SummaryWriter(curr_exp_dir)

    # Dump args to yaml and save to experiment directory
    with open(f"{curr_exp_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    

    # Freeze baseline just to be explicit
    for p in base_model.parameters():
        p.requires_grad = False

    # Setup autocast context and GradScaler for mixed precision training.
    # Note: GradScaler's foreach unscale functions currently do not support
    # BFloat16 on some PyTorch versions/hardware. To avoid the
    # NotImplementedError, we enable the scaler only for float16 (FP16).
    use_autocast_globally = (dtype in (torch.bfloat16, torch.float16))
    if use_autocast_globally and torch.cuda.is_available():
        # Choose autocast dtype based on requested dtype
        if dtype == torch.float16:
            global_autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            # dtype == bfloat16: use autocast but disable GradScaler (not supported)
            global_autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        # No CUDA or no mixed precision requested: fallback to nullcontext and no scaler
        global_autocast_ctx = contextlib.nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    # Load pre-trained 3D encoder checkpoint if specified
    if hasattr(args, 'pretrained_3d_checkpoint') and args.pretrained_3d_checkpoint:
        print(f"Loading pre-trained 3D encoder from: {args.pretrained_3d_checkpoint}")
        checkpoint = torch.load(args.pretrained_3d_checkpoint, map_location=device)
        
        # Get the state dict - handle both checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded from 'model_state_dict' key")
        elif 'model' in checkpoint:
            # This is from encoder training checkpoint
            state_dict = checkpoint['model']
            print("Loaded from 'model' key")
        else:
            # Direct state dict
            state_dict = checkpoint
            print("Loaded as direct state dict")
        
        print(f"State dict has {len(state_dict)} keys")
        print(f"Sample keys: {list(state_dict.keys())[:5]}")
        
        # Verify it's actually a model state dict
        if 'model' in state_dict or 'optimizer' in state_dict or 'lr_scheduler' in state_dict:
            raise ValueError(
                f"The checkpoint appears to be improperly formatted. "
                f"Expected model state dict but got keys: {list(state_dict.keys())[:10]}"
            )
        
        # Handle pos_embed size mismatch (encoder was trained on 512x512 with 32x32 PE)
        # We KEEP the 32x32 pos_embed from checkpoint - it matches the encoder's training!
        if 'image_encoder.pos_embed' in state_dict:
            checkpoint_pos_embed = state_dict['image_encoder.pos_embed']
            model_pos_embed = model_3d.image_encoder.pos_embed
            
            if checkpoint_pos_embed.shape != model_pos_embed.shape:
                print(f"Overriding model pos_embed {model_pos_embed.shape} with checkpoint {checkpoint_pos_embed.shape}")
                # Use checkpoint's pos_embed directly (trained with 512x512 input, 32x32 PE)
                model_3d.image_encoder.pos_embed = torch.nn.Parameter(checkpoint_pos_embed)
                print(f"✓ Loaded pos_embed with shape {checkpoint_pos_embed.shape} (matches encoder training)")
        
        # Load the state dict (with strict=False to handle any other mismatches)
        missing_keys, unexpected_keys = model_3d.load_state_dict(state_dict)
        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint (expected for decoder/prompt_encoder): {len(missing_keys)} keys")
            # Only print first few for brevity
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
            else:
                print(f"  First 10: {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")
    
    # Setup for decoder training (freeze encoder, decoder trainable, prompt encoder per-flag)
    freeze_pe = getattr(args, 'freeze_prompt_encoder', False)
    model_3d = setup_decoder_training_model(model_3d, freeze_prompt_encoder=freeze_pe)
    model_3d.train()

    # Count and print trainable parameters
    encoder_params = sum(p.numel() for p in model_3d.image_encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model_3d.image_encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model_3d.mask_decoder.parameters())
    decoder_trainable = sum(p.numel() for p in model_3d.mask_decoder.parameters() if p.requires_grad)
    prompt_encoder_params = sum(p.numel() for p in model_3d.prompt_encoder.parameters())
    prompt_encoder_trainable = sum(p.numel() for p in model_3d.prompt_encoder.parameters() if p.requires_grad)
    
    learnable_params = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_3d.parameters())
    
    print("\n" + "="*60)
    print("Model Parameter Summary:")
    print("="*60)
    print(f"Image Encoder:   {encoder_trainable:>12,} / {encoder_params:>12,} trainable")
    print(f"Mask Decoder:    {decoder_trainable:>12,} / {decoder_params:>12,} trainable")
    print(f"Prompt Encoder:  {prompt_encoder_trainable:>12,} / {prompt_encoder_params:>12,} trainable")
    print("-"*60)
    print(f"Total:           {learnable_params:>12,} / {total_params:>12,} trainable")
    print(f"Percentage:      {100.0 * learnable_params / total_params:>11.2f}%")
    print("="*60 + "\n")

    # Loss function
    mask_loss_fn = MaskConsistencyLoss(loss_type=getattr(args, 'mask_loss_type', 'dice_bce'))

    # Optimizer - only decoder and prompt encoder parameters
    trainable_params = [p for p in model_3d.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), 
                                 weight_decay=getattr(args, 'weight_decay', 0.01), eps=1e-7)
    # Immediate verification that encoder is frozen and optimizer has correct params
    log_and_verify_trainable_params(model_3d, optimizer, prefix="Trainable", writer=None, step=0)
    
    # Checkpointing - load before creating lr_scheduler to get proper train_dataset_size
    checkpointer = CheckPoint(checkpoint_dir)
    # Use a placeholder for train_dataset_size initially
    print("LR scheduler disabled for this run (using static lr from args). To enable, pass --enable_scheduler")
    lr_scheduler = None

    # Load checkpoint if exists (loads entire model including decoder!)
    # model_3d, global_step, first_epoch = load_checkpoint_with_resize(
    #     checkpointer, model_3d, None, None
    # )
    
    # Data loading (after model initialization and checkpoint loading)
    print("\nLoading datasets...")
    dataloader, eval_dataloader, train_dataset_size = get_dataloaders(args)
    builder = COLMAPBuilder(colmap_root=args.colmap_path,
                            scene_names=[args.scene] if args.scene else None)
    dataset = builder.build_scenes(
        min_overlap=args.min_overlap,
        max_overlap=args.min_overlap,
        max_num_pairs=args.max_num_pairs,
        n_dim=args.n_dim,
        max_correspondences=args.max_correspondences,
        image_size=args.image_size,
        dino_output_type=args.model_output_type,
        dataset_size=args.dataset_size,
        split="test",
        debug_small_data=args.debug_small_data,
        normalize_plucker=args.normalize_plucker,
        cross_plucker=args.cross_plucker,
        clip_norm=args.clip_norm, # Change ImageNet normalization for CLIP
        )
    dataset = ConcatDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            persistent_workers=True,
            # collate_fn=collate_fn,
            pin_memory=True,
    )
    # Ensure LR starts sane; reset if it's accidentally tiny from a bad scheduler state
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < 1e-12:
        print(f"[LR Guard] Detected extremely low LR ({current_lr:.2e}). Resetting to args.lr={args.lr}")
        for g in optimizer.param_groups:
            g['lr'] = float(args.lr)

    print(f"Scheduler: {lr_scheduler} | Initial LR: {optimizer.param_groups[0]['lr']} | Scheduler class: {type(lr_scheduler).__name__ if lr_scheduler else 'None'}")

    num_prompts = getattr(args, 'num_mask_prompts', 20)
    avg_loss = None
    first_epoch = 0
    global_step = 0
    
    # Sanity check: run one batch to verify loss magnitude
    print("\n" + "="*60)
    print("Running initial sanity check...")
    print("="*60)
    try:
        sanity_data = next(iter(dataloader))
        with torch.no_grad():
            sanity_loss, sanity_pred, sanity_target, _, _ = run_decoder_iter(
                model_3d, base_model, sanity_data, mask_loss_fn,
                device=device, num_prompts=num_prompts, dtype=dtype
            )
        print(f"Initial loss value: {sanity_loss.item():.4f}")
        print(f"Pred masks - mean: {torch.sigmoid(sanity_pred).mean().item():.4f}, std: {torch.sigmoid(sanity_pred).std().item():.4f}")
        print(f"Target masks - mean: {torch.sigmoid(sanity_target).mean().item():.4f}, std: {torch.sigmoid(sanity_target).std().item():.4f}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Sanity check failed: {e}")
        print("="*60 + "\n")
    
    # Training loop
    for epoch in range(first_epoch, args.num_epochs):
        # Evaluation
        if epoch % args.validation_interval == 0 and not args.skip_eval:
            eval_loss = 0
            eval_masks_3d = None
            eval_masks_base = None
            eval_images = None
            
            print("Evaluating on validation set...")
            for eval_idx, eval_data in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    loss, masks_3d, masks_base, points_3d, points_base = run_decoder_iter(
                        model_3d,
                        base_model,
                        eval_data,
                        mask_loss_fn,
                        device=device,
                        num_prompts=num_prompts,
                        dtype=dtype,
                    )
                eval_loss += loss.item()
                
                # Save first batch for visualization
                if eval_idx == 0:
                    eval_masks_3d = masks_3d
                    eval_masks_base = masks_base
                    eval_images = eval_data['images']
                    eval_points_3d = points_3d
                    eval_points_base = points_base

            eval_loss /= len(eval_dataloader)
            writer.add_scalar("Eval/Mask Loss", eval_loss, global_step)
            print(f"Epoch {epoch} - Eval Mask Loss: {eval_loss:.4f}")
            
            # Visualize eval masks
            if eval_masks_3d is not None:
                visualize_masks_to_tensorboard(
                    writer, eval_images, eval_masks_3d, eval_masks_base, 
                    global_step, prefix='eval', max_images=8, max_prompts=3,
                    point_coords_3d=eval_points_3d, point_coords_base=eval_points_base,
                )

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            checkpointer.save(model_3d, optimizer, lr_scheduler, global_step, epoch, avg_loss)

        # Training
        pbar = tqdm(dataloader)
        epoch_loss = 0
        # Visualization interval (iterations)
        vis_interval = getattr(args, 'vis_interval', 1000)

        for train_idx, data in enumerate(pbar):
            optimizer.zero_grad()
            # Run the full forward (including prompt/decoder) under autocast
            with global_autocast_ctx:
                loss, pred_masks, target_masks, points_3d_train, points_base_train = run_decoder_iter(
                    model_3d,
                    base_model,
                    data,
                    mask_loss_fn,
                    device=device,
                    num_prompts=num_prompts,
                    dtype=dtype,
                )

            # Backprop: if scaler is enabled (CUDA mixed precision), use it
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # Unscale once here so grad norm logging (and any clipping) sees true grads
                scaler.unscale_(optimizer)
                # Optional: gradient norm logging to catch vanishing/exploding grads
                if args.log_interval and (global_step % args.log_interval == 0):
                    try:
                        total_norm = 0.0
                        for g in optimizer.param_groups:
                            params = [p for p in g['params'] if p.grad is not None]
                            if not params:
                                continue
                            param_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2.0)
                            total_norm = (total_norm + param_norm.item())
                        writer.add_scalar("Train/grad_norm_l2", total_norm, global_step)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient norm logging
                if args.log_interval and (global_step % args.log_interval == 0):
                    try:
                        total_norm = 0.0
                        for g in optimizer.param_groups:
                            params = [p for p in g['params'] if p.grad is not None]
                            if not params:
                                continue
                            param_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2.0)
                            total_norm = (total_norm + param_norm.item())
                        writer.add_scalar("Train/grad_norm_l2", total_norm, global_step)
                    except Exception:
                        pass
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}/{args.num_epochs-1} - Mask Loss: {loss.item():.4f}")
            global_step += 1

            # Log to tensorboard
            if global_step % args.log_interval == 0:
                writer.add_scalar("Train Loss/Mask Loss", loss.item(), global_step)
                writer.add_scalar("Train Params/lr", optimizer.param_groups[0]["lr"], global_step)
                
                # Log mask statistics for debugging
                with torch.no_grad():
                    pred_prob = torch.sigmoid(pred_masks)
                    target_prob = torch.sigmoid(target_masks)
                    writer.add_scalar("Train Stats/pred_mask_mean", pred_prob.mean().item(), global_step)
                    writer.add_scalar("Train Stats/target_mask_mean", target_prob.mean().item(), global_step)
                    writer.add_scalar("Train Stats/pred_mask_std", pred_prob.std().item(), global_step)
                    writer.add_scalar("Train Stats/target_mask_std", target_prob.std().item(), global_step)

            # Optional LR print/log for verification (console + TB). Set --lr_log_steps N to print every N steps.
            if getattr(args, 'lr_log_steps', 0) and args.lr_log_steps > 0 and (global_step % args.lr_log_steps == 0):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[LR] step {global_step} lr={current_lr:.6e}")
                try:
                    writer.add_scalar("Train Params/lr_print", current_lr, global_step)
                except Exception:
                    pass

            # Visualize training masks every vis_interval iterations using current batch
            if vis_interval > 0 and (global_step % vis_interval == 0):
                visualize_masks_to_tensorboard(
                    writer,
                    data['images'],
                    pred_masks.detach(),
                    target_masks.detach(),
                    global_step,
                    prefix='train',
                    max_images=2,
                    max_prompts=3,
                    point_coords_3d=points_3d_train.detach(),
                    point_coords_base=points_base_train.detach(),
                )

            # Step scheduler per batch only for OneCycleLR (batch-wise schedule)
            if lr_scheduler is not None and isinstance(lr_scheduler, OneCycleLR):
                lr_scheduler.step()
        
        # Epoch-end scheduler step for epoch-based schedules (CosineAnnealingLR, LinearLR)
        if lr_scheduler is not None and not isinstance(lr_scheduler, OneCycleLR):
            lr_scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Training Mask Loss: {avg_loss:.4f}")
        
        if args.shuffle_data:
            dataloader, train_dataset_size = get_train_dataloader(args)

    # Final checkpoint
    checkpointer.save(model_3d, optimizer, lr_scheduler, global_step)
    print("Training completed!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = ArgumentParser()
    parser.add_argument("--exp_directory", default='experiments', type=str, help="Name of the experiment for logging")
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment for logging")
    parser.add_argument("--config_name", default='train.yaml', type=str, help="YAML Configuration file")
    parser.add_argument("--log_interval", default=1, type=int, help="Interval for logging to TensorBoard")
    parser.add_argument("--validation_interval", default=1, type=int, help="Interval for calculating validation set")
    parser.add_argument("--checkpoint_interval", default=1, type=int, help="Interval for saving checkpoints")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--colmap_path", default='data/', type=str, help="Path to COLMAP data")
    parser.add_argument("--scene", default=None, type=str, help="Name of scene")
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation on validation set')
    parser.add_argument('--chunk_data', action='store_true', help='Use chunked data loading')
    parser.add_argument('--pretrained_3d_checkpoint', default=None, type=str, 
                       help='Path to pre-trained 3D SAM encoder checkpoint')
    parser.add_argument('--num_mask_prompts', default=10, type=int, 
                       help='Number of point prompts to use for mask generation')
    parser.add_argument('--mask_loss_type', default='dice_bce', type=str,
                       choices=['dice', 'bce', 'dice_bce', 'mse'],
                       help='Type of loss for mask comparison')
    parser.add_argument('--use_bfloat16', action='store_true',
                       help='Use bfloat16 precision for training (reduces memory usage)')
    parser.add_argument('--vis_interval', default=500, type=int,
                        help='Iterations between mask visualizations to TensorBoard (0 disables)')
    parser.add_argument('--enable_scheduler', action='store_true',
                        help='Enable LR scheduler (default: disabled). Use this to opt-in to scheduling.')
    parser.add_argument('--disable_scheduler', action='store_true',
                        help='(Deprecated) Disable scheduler - kept for backward compatibility with configs')
    parser.add_argument('--lr_log_steps', default=0, type=int,
                        help='Print LR to console and log to TensorBoard every N steps (0 disables)')
    parser.add_argument('--base_checkpoint', type=str, default=None,
                        help='Checkpoint path for ORIGINAL baseline SAM (BASE_SAM). If omitted uses --checkpoint.')
    parser.add_argument('--base_model_type', type=str, default=None,
                        help='Model type for baseline SAM (e.g., vit_b, vit_l). Defaults to --model_type if not set.')
    parser.add_argument('--require_original_baseline', action='store_true',
                        help='If set, abort when BASE_SAM cannot be imported (no fallback to modified SAM).')
    parser.add_argument('--freeze_prompt_encoder', action='store_true',
                        help='Freeze prompt encoder (only train mask decoder).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config file if specified)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (overrides config file if specified)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config file if specified)')
    # Deprecated: baseline flattens multi-view into batch automatically

    parsed_args, unknown = parser.parse_known_args()
    # Merge cfg with args
    config_path = os.path.join('configs', parsed_args.config_name or 'train.yaml')
    cfg = load_config(config_path)
    cfg_parser = ArgumentParser()
    for key, value in cfg.items():
        if key in vars(parsed_args):
            continue
        if isinstance(value, bool):
            cfg_parser.add_argument(f'--{key}', action='store_true', default=None)
        elif isinstance(value, int):
            cfg_parser.add_argument(f'--{key}', type=int, default=None)
        elif isinstance(value, float):
            cfg_parser.add_argument(f'--{key}', type=float, default=None)
        elif isinstance(value, str):
            cfg_parser.add_argument(f'--{key}', type=str, default=None)
        elif isinstance(value, list):
            if value and all(isinstance(x, int) for x in value):
                cfg_parser.add_argument(f'--{key}', nargs='*', type=int, default=None)
            elif value and all(isinstance(x, float) for x in value):
                cfg_parser.add_argument(f'--{key}', nargs='*', type=float, default=None)
    cfg_parser.set_defaults(**cfg)
    cfg_args = cfg_parser.parse_args(unknown)
    args = SafeNamespace(**vars(cfg_args))
    for key, value in vars(parsed_args).items():
        if value is not None and value is not False:
            setattr(args, key, value)
    print("Final parameters:")
    print(yaml.dump(vars(args), default_flow_style=False))
    train(args)
