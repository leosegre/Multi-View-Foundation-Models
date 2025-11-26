import argparse
import os
import json
import torch
import yaml
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from einops import rearrange
from collections import defaultdict

from CLIP.clip.utils import get_clip_model
from dino3d.datasets.colmapdata import COLMAPBuilder
from dino3d.models.dino3d import DINO3D
from dino3d.models.base_dino import DINO
from dino3d.losses.corr_loss import ConsistencyLoss
from dino3d.losses.corr_map_model import Correlation2Displacement
from dino3d.checkpointing import CheckPoint
from pathlib import Path
import types
from FiT3D.utils import get_intermediate_layers
from dino3d.models.utils.utils import get_dino3d_model
from dino3d.train.train import fix_random_seeds, get_dataloaders, get_train_dataloader, run_iter, log_pca, \
    get_scheduler, load_config, SafeNamespace
from eval.linear_evaluate_fit3d import FiT3D
from torch.utils.data import ConcatDataset
from OpenClip.create_open_clip import open_clip_load
from save_images import save_everything


class SafeNamespace(argparse.Namespace):
    def __getattr__(self, name):
        # Called only if 'name' not found in self.__dict__
        return False


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read())


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    else:
        return obj


def compute_viewpoint_angle(data):
    """
    Compute the angle between camera viewpoints for pairs of images.
    For n_dims=2, we have pairs of images with shape (B, 2, ...).

    Returns angle in degrees between the two camera poses.
    """
    # Extract camera information from plucker coordinates
    # Plucker coordinates encode ray directions, but we need camera poses
    # We'll compute the angle from the ray directions

    plucker = data["plucker"]  # Shape: (B, 2, 6, H, W)
    B, S = plucker.shape[0], plucker.shape[1]

    if S != 2:
        raise ValueError(f"Viewpoint analysis only works with n_dims=2 (pairs), got {S} images")

    # Extract direction vectors from center pixels
    # Plucker coordinates: first 3 are direction, last 3 are moment
    H, W = plucker.shape[3], plucker.shape[4]
    center_h, center_w = H // 2, W // 2

    # Get ray directions at image center for both views
    dir1 = plucker[:, 0, :3, center_h, center_w]  # (B, 3)
    dir2 = plucker[:, 1, :3, center_h, center_w]  # (B, 3)

    # Normalize directions
    dir1 = dir1 / (torch.norm(dir1, dim=1, keepdim=True) + 1e-8)
    dir2 = dir2 / (torch.norm(dir2, dim=1, keepdim=True) + 1e-8)

    # Compute angle using dot product
    cos_angle = torch.sum(dir1 * dir2, dim=1)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg.cpu().numpy()  # Return as numpy array


def get_angle_bin(angle, angle_bins):
    """
    Get the bin index for a given angle.
    Returns the index of the bin that the angle falls into.
    """
    for i in range(len(angle_bins) - 1):
        if angle >= angle_bins[i] and angle < angle_bins[i + 1]:
            return i
    return len(angle_bins) - 2  # Last bin


@torch.no_grad()
def evaluate(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fix_random_seeds(args.seed)

    # Validate viewpoint analysis settings
    if args.viewpoint_analysis and args.n_dim != 2:
        raise ValueError("Viewpoint analysis only works with n_dims=2 (image pairs)")

    # Initialize storage for per-angle metrics (1-degree resolution)
    if args.viewpoint_analysis:
        print("Recording metrics at 1-degree angle resolution")

        # Initialize storage for per-angle metrics
        angle_metrics = defaultdict(lambda: {
            "count": 0,
            "3d_location_error": 0.0,
            "3d_cosine_similarity": 0.0,
            "2d_location_error": 0.0,
            "2d_cosine_similarity": 0.0,
            "3d_cosine_similarity_to_base": 0.0,
            "fit3d_location_error": 0.0,
            "fit3d_cosine_similarity": 0.0,
            "fit3d_cosine_similarity_to_base": 0.0,
            "fit3d_with_base_location_error": 0.0,
            "fit3d_with_base_cosine_similarity": 0.0,
        })

    # Load models
    if args.model_type == "dinov2":
        model = get_dino3d_model(args).to(device)
    elif args.model_type == "clip":
        model = open_clip_load(args).to(device)
    elif args.model_type == "dinov3":
        from train.train_dinov3 import get_dinoV33d_model
        model = get_dinoV33d_model(args).to(device)
    model.eval()

    # Load checkpoint
    checkpoint_dir = os.path.join(args.exp_directory, args.exp_name, "checkpoints/")
    print(f"Loading model from {checkpoint_dir}")
    checkpointer = CheckPoint(checkpoint_dir)
    model = checkpointer.load_model(model, checkpoint_name=args.checkpoint_name)

    # Init base model
    if args.model_type == "dinov2":
        base_model = DINO(
            model_name=args.model_name,
            output=args.model_output_type,
            return_multilayer=False,
            stride=args.stride,
        )
    elif args.model_type == "clip":
        base_model = open_clip_load(args,
                                    spatial_reduction=False,
                                    use_sa_ffn=False,
                                    plucker_emb_dim=0)
    elif args.model_type == "dinov3":
        base_model = get_dinoV33d_model(args,
                                        device=device,
                                        spatial_reduction=False,
                                        use_sa_ffn=False,
                                        plucker_emb_dim=0)
    base_model = base_model.to(device)
    base_model.eval()

    # Load FiT3D model if specified
    if args.fit3d:
        fit3d_model = FiT3D(args.fit3d_backbone_type)
        fit3d_model.eval()
        fit3d_model.to(device)

    # Load dataset
    builder = COLMAPBuilder(colmap_root=args.colmap_path, scene_names=[args.scene] if args.scene else None)
    test_dataset = builder.build_scenes(
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        max_num_pairs=args.max_num_pairs,
        n_dim=args.n_dim,
        max_correspondences=args.max_correspondences,
        image_size=args.image_size,
        dino_output_type=args.model_output_type,
        split=args.split,
    )

    test_dataset = ConcatDataset(test_dataset)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,  # assumes only one scene
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Loss module
    if args.model_type == "dinov2":
        feature_size = 1 + (args.image_size[0] - model.patch_size) // model.stride
    elif args.model_type == "clip":
        feature_size = args.image_size[0] // model.vision_patch_size
    elif args.model_type == "dinov3":
        feature_size = args.image_size[0] // model.patch_size
    correlation_module = Correlation2Displacement(feature_size=feature_size, calc_soft_argmax=False)
    consistency_loss = ConsistencyLoss(correlation_module, optim_strategy='similarity', loss_type='L1')

    total_3d_location_error, total_3d_cosine_similarity, total_2d_location_error, total_2d_cosine_similarity = 0, 0, 0, 0
    total_3d_cosine_similarity_to_base = 0
    if args.fit3d:
        fit3d_total_location_error, fit3d_total_cosine_similarity = 0, 0
        fit3d_with_base_total_location_error, fit3d_with_base_total_cosine_similarity = 0, 0
        fit3d_total_cosine_similarity_to_base = 0
    n_batches = 0

    for i, data in enumerate(tqdm(dataloader, desc="Evaluating")):
        data = move_to(data, device)

        # Compute viewpoint angle if analysis is enabled
        if args.viewpoint_analysis:
            angles = compute_viewpoint_angle(data)  # Shape: (B,)

        # 3D model
        if args.model_type == "dinov2" or args.model_type == "dinov3":
            features_3d = model(data["images"], data["plucker"])
        elif args.model_type == "clip":
            features_3d = model.image_features(data["images"], data["plucker"])

        # 2D model
        if args.model_type == "dinov2":
            features_2d = base_model(rearrange(data["images"], 'b s c h w -> (b s) c h w'))
            B, S = data["images"].shape[0], data["images"].shape[1]
            features_2d = rearrange(features_2d, '(b s) c h w -> b s c h w', b=B, s=S)
        elif args.model_type == "clip":
            features_2d = base_model.image_features(data["images"])
        elif args.model_type == "dinov3":
            features_2d = base_model(data["images"])

        if args.fit3d:
            if args.model_type == "dinov2":
                # Adjust the stride to fit our model
                fit3d_model.vit.patch_embed.proj.stride = (model.stride, model.stride)
                fit3d_model.finetuned_model.patch_embed.proj.stride = (model.stride, model.stride)
            # B, S = data["images"].shape[:2]
            # images_flat = rearrange(data["images"], "b s c h w -> (b s) c h w")
            # fit3d_features = fit3d_model.get_intermediate_layers(images_flat, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
            #                         return_class_token=False, norm=True)
            fit3d_features_dict = fit3d_model(data["images"], return_channel_first=True, return_dict=True)
            fit3d_features = fit3d_features_dict["raw_vit_feats_fine"]

        if args.save_images:
            data_to_save = {
                "images": data["images"],  # (B=2, S=4, C=3, H=256, W=256)
                "plucker": data["plucker"],  # (B=2, S=4, C=6, H, W)
                "features_3d": features_3d,  # (B=2, S=4, C=64, H, W)
                "features_2d": features_2d,
                "fit3d_features": fit3d_features if args.fit3d else None,
            }
            save_everything(data_to_save, out_root=f"exports/{i}")

        # print(data["image_ids"].shape, data["pairwise_correspondence"].shape, data["images"].shape)

        dense_3d, sparse_3d = consistency_loss(
            features_3d,
            data["image_ids"],
            data["pairwise_correspondence"],
            data["images"],
        )


        dense_2d, sparse_2d = consistency_loss(
            features_2d,
            data["image_ids"],
            data["pairwise_correspondence"],
            data["images"],
        )

        if args.compare_to_base:
            # Calculate cosine similarity of features_3d to features_2d
            batch_3d_cosine_similarity_to_base = torch.nn.functional.cosine_similarity(features_3d, features_2d,
                                                                                       dim=2).mean()

        # If FiT3D model is provided, compute its losses
        if args.fit3d:
            # fit3d_features = rearrange(fit3d_features, "(b s) c h w -> b s c h w", b=B, s=S)
            fit3d_dense, fit3d_sparse = consistency_loss(
                fit3d_features,
                data["image_ids"],
                data["pairwise_correspondence"],
                data["images"],
            )

            fit3d_features_with_base = fit3d_features_dict["out_feat"]
            fit3d_dense_with_base, fit3d_sparse_with_base = consistency_loss(
                fit3d_features_with_base,
                data["image_ids"],
                data["pairwise_correspondence"],
                data["images"],
            )

            if args.compare_to_base:
                # Calculate cosine similarity of fit3d_features to features_2d
                batch_fit3d_cosine_similarity_to_base = torch.nn.functional.cosine_similarity(fit3d_features,
                                                                                              features_2d, dim=2).mean()

        # Accumulate metrics
        if args.viewpoint_analysis:
            # Accumulate per-angle metrics (round to nearest degree)
            B = data["images"].shape[0]
            for b in range(B):
                angle = int(round(angles[b]))  # Round to nearest degree

                angle_metrics[angle]["count"] += 1
                angle_metrics[angle]["3d_location_error"] += dense_3d.item() / B
                angle_metrics[angle]["3d_cosine_similarity"] += (1 - sparse_3d.item()) / B
                angle_metrics[angle]["2d_location_error"] += dense_2d.item() / B
                angle_metrics[angle]["2d_cosine_similarity"] += (1 - sparse_2d.item()) / B

                if args.compare_to_base:
                    angle_metrics[angle][
                        "3d_cosine_similarity_to_base"] += batch_3d_cosine_similarity_to_base.item() / B

                if args.fit3d:
                    angle_metrics[angle]["fit3d_location_error"] += fit3d_dense.item() / B
                    angle_metrics[angle]["fit3d_cosine_similarity"] += (1 - fit3d_sparse.item()) / B
                    angle_metrics[angle]["fit3d_with_base_location_error"] += fit3d_dense_with_base.item() / B
                    angle_metrics[angle]["fit3d_with_base_cosine_similarity"] += (1 - fit3d_sparse_with_base.item()) / B
                    if args.compare_to_base:
                        angle_metrics[angle][
                            "fit3d_cosine_similarity_to_base"] += batch_fit3d_cosine_similarity_to_base.item() / B
        else:
            # Regular accumulation
            total_3d_location_error += dense_3d.item()
            total_3d_cosine_similarity += 1 - sparse_3d.item()
            total_2d_location_error += dense_2d.item()
            total_2d_cosine_similarity += 1 - sparse_2d.item()
            if args.compare_to_base:
                total_3d_cosine_similarity_to_base += batch_3d_cosine_similarity_to_base.item()

            if args.fit3d:
                fit3d_total_location_error += fit3d_dense.item()
                fit3d_total_cosine_similarity += 1 - fit3d_sparse.item()
                fit3d_with_base_total_location_error += fit3d_dense_with_base.item()
                fit3d_with_base_total_cosine_similarity += 1 - fit3d_sparse_with_base.item()
                if args.compare_to_base:
                    fit3d_total_cosine_similarity_to_base += batch_fit3d_cosine_similarity_to_base.item()

        n_batches += 1

    # Save results
    if args.viewpoint_analysis:
        # Compute averages per angle
        avg_results = {
            "exp_name": args.exp_name,
            "scene": args.scene,
            "viewpoint_analysis": True,
            "angle_results": {}
        }

        print("\nEvaluation Results by Viewpoint Angle:")
        print(f"{'Angle':<8} {'Count':<8} {'3D Loc Err':<12} {'3D Cos Sim':<12} {'2D Loc Err':<12} {'2D Cos Sim':<12}")
        print("-" * 80)

        # Sort angles for cleaner output
        for angle in sorted(angle_metrics.keys()):
            count = angle_metrics[angle]["count"]

            if count > 0:
                angle_avg = {
                    "count": count,
                }

                # Average all metrics
                for key in angle_metrics[angle]:
                    if key != "count":
                        angle_avg[key] = angle_metrics[angle][key] / count

                avg_results["angle_results"][str(angle)] = angle_avg

                # Print summary
                print(f"{angle:<8} {count:<8} {angle_avg['3d_location_error']:<12.4f} "
                      f"{angle_avg['3d_cosine_similarity']:<12.4f} "
                      f"{angle_avg['2d_location_error']:<12.4f} "
                      f"{angle_avg['2d_cosine_similarity']:<12.4f}")

        print("\nAdditional metrics saved in JSON file.")
        if args.compare_to_base:
            print("- 3D to base cosine similarity")
        if args.fit3d:
            print("- FiT3D metrics")
    else:
        avg_results = {
            "exp_name": args.exp_name,
            "scene": args.scene,
            "total_3d_location_error": total_3d_location_error / n_batches,
            "total_3d_cosine_similarity": total_3d_cosine_similarity / n_batches,
        }

        if args.compare_to_base:
            avg_results["total_3d_cosine_similarity_to_base"] = total_3d_cosine_similarity_to_base / n_batches

        avg_results["total_2d_location_error"] = total_2d_location_error / n_batches
        avg_results["total_2d_cosine_similarity"] = total_2d_cosine_similarity / n_batches

        if args.fit3d:
            avg_results["fit3d_total_location_error"] = fit3d_total_location_error / n_batches
            avg_results["fit3d_total_cosine_similarity"] = fit3d_total_cosine_similarity / n_batches
            if args.compare_to_base:
                avg_results["fit3d_total_cosine_similarity_to_base"] = fit3d_total_cosine_similarity_to_base / n_batches
            avg_results["fit3d_with_base_total_location_error"] = fit3d_with_base_total_location_error / n_batches
            avg_results["fit3d_with_base_total_cosine_similarity"] = fit3d_with_base_total_cosine_similarity / n_batches

        print("\nEvaluation Results:")
        for k, v in avg_results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    os.makedirs(args.results_dir, exist_ok=True)
    suffix = "_viewpoint" if args.viewpoint_analysis else ""
    out_path = f"{args.results_dir}/test_{args.exp_name}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(avg_results, f, indent=4)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_directory", default="experiments", type=str)
    parser.add_argument("--exp_name", default=None, type=str, help="Name of the experiment for logging")
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--colmap_path", default='data/', type=str, help="Path to COLMAP data")
    parser.add_argument("--fit3d", action="store_true", help="Use FiT3D model for feature extraction")
    parser.add_argument("--fit3d_backbone_type", default="dinov2_reg_small", type=str,
                        help="Backbone type for FiT3D model",
                        choices=["dinov2_small", "dinov2_base", "dinov2_reg_small", "clip_base", "mae_base",
                                 "deit3_base"])
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--split", default="test", type=str, help="Dataset split to evaluate on")
    parser.add_argument("--model_type", default="dinov2", type=str, help="Name of the model",
                        choices=["dinov2", "clip", "dinov3"])
    parser.add_argument("--results_dir", default="results", type=str, help="Directory to save results")
    parser.add_argument("--compare_to_base", action="store_true", help="Compare 3D features to base 2D features")
    parser.add_argument("--checkpoint_name", default=None, type=str, help="Name of the checkpoint file to load")
    parser.add_argument("--save_images", action="store_true", help="Show images with correspondences")
    parser.add_argument("--n_dims", default=4, type=int, help="Number of feature dimensions")
    parser.add_argument("--viewpoint_analysis", action="store_true",
                        help="Analyze results as a function of viewpoint angle (only works with n_dims=2)")
    parser.add_argument("--image_size", default=518, type=int, help="Image size")
    args, _ = parser.parse_known_args()

    # Load args.yaml from exp_directory
    args_yaml_path = os.path.join(args.exp_directory, args.exp_name, "args.yaml")
    if not os.path.exists(args_yaml_path):
        raise FileNotFoundError(f"args.yaml not found at {args_yaml_path}")
    yaml_args = load_config(args_yaml_path)

    # Override image_size
    yaml_args["image_size"] = (args.image_size, args.image_size)
    if args.image_size > 518:
        print("Warning: image_size > 518 may cause out-of-memory issues. Batch down to 2")
        yaml_args["batch_size"] = 2
    # Override scene
    yaml_args["scene"] = args.scene
    # Override colmap_path
    yaml_args["colmap_path"] = args.colmap_path
    # Override exp_directory
    yaml_args["exp_directory"] = args.exp_directory
    # Override seed
    yaml_args["seed"] = args.seed
    # Override n_dims
    yaml_args["n_dim"] = args.n_dims
    # Add fit3d
    yaml_args["fit3d"] = args.fit3d
    # Add backbone type for fit3d
    yaml_args["fit3d_backbone_type"] = args.fit3d_backbone_type
    # Add split
    yaml_args["split"] = args.split
    # Add model_type
    yaml_args["model_type"] = args.model_type
    # Add results_dir
    yaml_args["results_dir"] = args.results_dir
    # Add compare_to_base
    yaml_args["compare_to_base"] = args.compare_to_base
    # Add checkpoint_name
    yaml_args["checkpoint_name"] = args.checkpoint_name
    # Add save_images
    yaml_args["save_images"] = args.save_images
    # Add viewpoint_analysis
    yaml_args["viewpoint_analysis"] = args.viewpoint_analysis

    if yaml_args["save_images"]:
        # Override stride to 1 and batch size to 1 for better visualization
        yaml_args["stride"] = 2
        yaml_args["batch_size"] = 1

    # Merge all: full config → yaml args → CLI args
    args = SafeNamespace(**yaml_args)

    evaluate(args)