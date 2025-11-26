import argparse

import numpy as np
import torch
import yaml
from einops import rearrange
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR
from dino3d.datasets.chunked_colmap_dataset import ChunkedCOLMAPDataset
from dino3d.datasets.colmapdata import COLMAPBuilder
from dino3d.losses.reg_loss import get_reg_loss
from dino3d.utils.visualization import merged_pca


class SafeNamespace(argparse.Namespace):
    def __getattr__(self, name):
        # Called only if 'name' not found in self.__dict__
        return False


def process_base_features(base_features, dino_output_type):
    # Features are dense-cls -> convert to args.dino_output_type
    # Features are [spatial_tokens, cls_token], concatenated along the channel dimension
    C, H, W = base_features.shape
    if dino_output_type == 'dense-cls':
        return base_features
    elif dino_output_type == 'dense':
        return base_features[: C // 2]
    elif dino_output_type == 'cls':
        return base_features[C // 2:]
    else:
        raise ValueError(f"Unknown dino_output_type: {dino_output_type}")

def run_iter(model, data, 
             consistency_loss, 
             device='cuda:0', 
             base_model=None, 
             concat_with_base=False, 
             separate_forward_regularization=False,
             clip=False,
             sam=False,
             dinov3=False,
             cross_plucker=False,
             cosine_similarity_loss=None):
    """
    Run a single training/eval iteration. By default this runs a single forward
    pass where all views are processed together and the regularization loss is
    computed from those features. If the model has the attribute
    `separate_forward_regularization=True` then a second forward pass is run
    where each view is evaluated separately (reshaping [B,S,C,H,W] ->
    [B*S,1,C,H,W]) and the regularization loss is computed from that pass.
    """
    data = move_to(data, device)
    epipolar_enabled = getattr(model, 'epipolar_enabled', False)

    if clip or sam or dinov3:
        cast_dtype = torch.bfloat16
    else:
        cast_dtype = torch.bfloat16
    # First (joint) forward pass - this is the default behaviour
    with torch.autocast("cuda", dtype=cast_dtype):
        if epipolar_enabled:
            features = model(data["images"],
                             data["plucker"],
                             data.get("intrinsics", None),
                             data.get("extrinsics", None))
        elif clip or sam:
            features = model.image_features(data["images"], data["plucker"])
        else:
            features = model(data["images"], data["plucker"])
            if cross_plucker:
                with torch.no_grad():
                    cross_features = model(data["images"], data["cross_plucker"])


        if base_model is not None:
            with torch.no_grad():
                if clip or sam:
                    base_features = base_model.image_features(data["images"])
                elif dinov3:
                    base_features = base_model(data["images"])
                else:
                    base_features = base_model(rearrange(data["images"], 'b s c h w -> (b s) c h w'))
                    B, S = data["images"].shape[0], data["images"].shape[1]
                    base_features = rearrange(base_features, '(b s) c h w -> b s c h w', b=B, s=S)
        else:
            base_features = data.get("base_features", None)

    if concat_with_base:
        features = torch.cat([features, base_features], dim=2)

    dense_loss, sparse_loss = consistency_loss(features,
                                               data['image_ids'],
                                               data['pairwise_correspondence'],
                                               data["images"]
                                               )

    if cross_plucker:
        # Calculate cosine similarity between features and cross_features
        # print(features.shape, cross_features.shape)
        cross_plucker_loss = get_reg_loss(features, cross_features.detach())
    else:
        cross_plucker_loss = torch.tensor(0.0, device=device)

    # By default compute regularization loss from the joint features. If
    # separate_forward_regularization is enabled, run a second forward pass
    # where each view is evaluated independently and compute the regularizer
    # from those features instead.
    if separate_forward_regularization:
        B, S = data["images"].shape[0], data["images"].shape[1]
        # reshape inputs so each view becomes its own batch item: (B*S, 1, C, H, W)
        images_sep = rearrange(data["images"], 'b s c h w -> (b s) 1 c h w')
        plucker_sep = rearrange(data["plucker"], 'b s c h w -> (b s) 1 c h w')

        intrinsics = data.get("intrinsics", None)
        extrinsics = data.get("extrinsics", None)
        intrinsics_sep = None
        extrinsics_sep = None
        if intrinsics is not None:
            intrinsics_sep = rearrange(intrinsics, 'b s ... -> (b s) 1 ...')
        if extrinsics is not None:
            extrinsics_sep = rearrange(extrinsics, 'b s ... -> (b s) 1 ...')

        if epipolar_enabled:
            features_sep = model(images_sep, plucker_sep, intrinsics_sep, extrinsics_sep)
        else:
            features_sep = model(images_sep, plucker_sep)

        # features_sep shape is ((B*S), 1, C, H, W) -> reshape back to (B, S, C, H, W)
        features_sep = rearrange(features_sep, '(b s) 1 c h w -> b s c h w', b=B, s=S)
        regularization_loss = get_reg_loss(features_sep, base_features)
    else:
        if concat_with_base:
            regularization_loss = torch.tensor(0.0, device=device)
        else:
            regularization_loss = get_reg_loss(features, base_features)

    return features, base_features, dense_loss, sparse_loss, regularization_loss, cross_plucker_loss


def get_train_dataloader(args):
    if args.chunk_data:
        dataset = ChunkedCOLMAPDataset(
        chunk_root=args.colmap_path,
        split="train",
        n_dim=args.n_dim,
        max_correspondences=args.max_correspondences,
        image_size=args.image_size,
        dino_output_type=args.model_output_type,
        dataset_size=args.dataset_size,
        no_features=False,           # or True to process online
        load_corrs=True,
    )
    else:
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
            split="train",
            debug_small_data=args.debug_small_data,
        )
        dataset = ConcatDataset(dataset)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            pin_memory=True,
    )
    return dataloader, dataset_size

def get_dataloaders(args):
    # Train dataset
    if args.chunk_data:
        dataset = ChunkedCOLMAPDataset(
        chunk_root=args.colmap_path,
        split="train",
        n_dim=args.n_dim,
        max_correspondences=args.max_correspondences,
        image_size=args.image_size,
        dino_output_type=args.model_output_type,
        dataset_size=args.dataset_size,
        no_features=False,           # or True to process online
        load_corrs=True,
    )
    else:
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
            split="train",
            debug_small_data=args.debug_small_data,
            normalize_plucker=args.normalize_plucker,
            cross_plucker=args.cross_plucker,
            clip_norm=args.clip_norm, # Change ImageNet normalization for CLIP
        )
        dataset = ConcatDataset(dataset)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            persistent_workers=True,
            # collate_fn=collate_fn,
            pin_memory=True,
    )

    # Eval dataset
    if args.chunk_data:
        eval_dataset = ChunkedCOLMAPDataset(
            chunk_root=args.colmap_path,
            split="eval",
            n_dim=args.n_dim,
            max_correspondences=args.max_correspondences,
            image_size=args.image_size,
            dino_output_type=args.model_output_type,
            dataset_size=args.dataset_size // 10,  # Smaller eval dataset
            no_features=False,
            load_corrs=True,
        )
    else:
        eval_builder = COLMAPBuilder(colmap_root=args.colmap_path, scene_names=[args.scene] if args.scene else None)
        eval_dataset = eval_builder.build_scenes(
            min_overlap=args.min_overlap,
            max_overlap=args.min_overlap,
            max_num_pairs=args.max_num_pairs,
            n_dim=args.n_dim,
            max_correspondences=args.max_correspondences,
            image_size=args.image_size,
            dino_output_type=args.model_output_type,
            dataset_size=args.dataset_size // 10,  # Smaller eval dataset
            split="eval",
            debug_small_data=args.debug_small_data,
            cross_plucker=args.cross_plucker,
            clip_norm=args.clip_norm, # Change ImageNet normalization for CLIP
        )
        eval_dataset = ConcatDataset(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=True,
            # collate_fn=collate_fn,
            pin_memory=True,
    )
    return dataloader, eval_dataloader, dataset_size

def log_pca(writer, global_step, title, images, features, base_features, batch_idx=0):
    features_3d, features_2d = merged_pca(features[batch_idx], base_features[batch_idx])
    writer.add_images(f"{title} PCA/Images", images[batch_idx], global_step)
    writer.add_images(f"{title} PCA/3D features", features_3d, global_step)
    writer.add_images(f"{title} PCA/2D features", features_2d, global_step)

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj

def collate_fn(batch):
    out = {
        "images": torch.stack([b["images"] for b in batch]),
        "plucker": torch.stack([b["plucker"] for b in batch]),
        # "identifiers": [b["identifiers"] for b in batch],
        "image_ids": torch.stack([b["image_ids"] for b in batch]),
        "pairwise_correspondence": torch.stack([b["pairwise_correspondence"] for b in batch]),
    }
    # Add intrinsics/extrinsics/base_features if present
    out["intrinsics"] = torch.stack([b["intrinsics"] for b in batch])
    out["extrinsics"] = torch.stack([b["extrinsics"] for b in batch])
    # out["base_features"] = torch.stack([b["base_features"] for b in batch])
    return out


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read())


def get_scheduler(optimizer, train_dataset_size, batch_size, args):
    lr_scheduler = None
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer,
                                             T_max=train_dataset_size//batch_size,
                                             eta_min=1e-6)
        if args.scheduler == 'one_cycle':
            lr_scheduler = OneCycleLR(optimizer,
                                      max_lr=args.lr,
                                      steps_per_epoch=train_dataset_size,
                                      epochs=args.num_epochs)
        if args.scheduler == 'linear':
            lr_scheduler = LinearLR(optimizer,
                                    start_factor=0.1,
                                    total_iters=args.num_epochs)
    return lr_scheduler

def get_scheduler(optimizer, train_dataset_size, batch_size, args):
    lr_scheduler = None
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer,
                                             T_max=train_dataset_size//batch_size,
                                             eta_min=1e-6)
        if args.scheduler == 'one_cycle':
            lr_scheduler = OneCycleLR(optimizer,
                                      max_lr=args.lr,
                                      steps_per_epoch=train_dataset_size,
                                      epochs=args.num_epochs)
        if args.scheduler == 'linear':
            lr_scheduler = LinearLR(optimizer,
                                    start_factor=0.1,
                                    total_iters=args.num_epochs)
    return lr_scheduler