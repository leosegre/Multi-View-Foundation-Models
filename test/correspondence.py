import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dino3d.checkpointing import CheckPoint
from dino3d.datasets.colmapdata import COLMAPBuilder
from dino3d.models.dino3d import DINO3D
from dino3d.models.utils.utils import get_dino3d_model, get_lora_dino_model
from dino3d.train import move_to
import types
from einops import rearrange
from FiT3D.utils import get_intermediate_layers
from matplotlib.patches import Circle, Rectangle
from sklearn.decomposition import PCA
from dino3d.models.base_dino import DINO
from FiT3D.linear_evaluate_fit3d import FiT3D


class SafeNamespace(argparse.Namespace):
    def __getattr__(self, name):
        # Called only if 'name' not found in self.__dict__
        return False

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read())


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))
    return torch.stack(result_list, dim=2)


def shared_pca_projection(support_feats_dict):
    """Compute shared PCA across all support features, return per-feature RGB maps."""
    feat_list = []
    feat_shapes = {}
    for name, feat in support_feats_dict.items():
        C, H, W = feat.shape
        feat_flat = feat.view(C, -1).T.cpu().numpy()  # (H*W, C)
        feat_list.append(feat_flat)
        feat_shapes[name] = (H, W)

    # Stack and fit PCA
    all_feats = np.concatenate(feat_list, axis=0)  # (H*W*num_feats, C)
    pca = PCA(n_components=3)
    pca.fit(all_feats)

    # Transform each feature using the shared PCA
    pca_images = {}
    for name, feat in support_feats_dict.items():
        C, H, W = feat.shape
        feat_flat = feat.view(C, -1).T.cpu().numpy()
        feat_pca = pca.transform(feat_flat)
        feat_pca -= feat_pca.min(0, keepdims=True)
        feat_pca /= feat_pca.max(0, keepdims=True) + 1e-5
        pca_images[name] = feat_pca.reshape(H, W, 3)

    return pca_images

def show_similarity_interactive(
    support_img,
    query_imgs,
    support_feats,     # dict: {"base": feat_tensor, "ours": feat_tensor, ...}
    query_feats,       # list of dicts: [{"base": feat_tensor, "ours": ...}, ...]
    feat_names         # list of feature names: ["base", "ours", ...]
):
    support_img_np = support_img.permute(1, 2, 0).cpu().numpy()
    query_imgs_np = [q.permute(1, 2, 0).cpu().numpy() for q in query_imgs]


    # Prepare per-feature shapes
    support_shapes = {k: v.shape[1:] for k, v in support_feats.items()}  # {feat_name: (H, W)}
    support_channels = {k: v.shape[0] for k, v in support_feats.items()}  # {feat_name: C}
    num_queries = len(query_imgs)
    num_feats = len(feat_names)

    fig, axes = plt.subplots(1 + num_feats, num_queries + 1, figsize=(4 * (num_queries + 1), 4 * (1 + num_feats)))
    for axi in axes.ravel():
        axi.set_axis_off()

    radius = 14 // 2

    # Assign unique colors per feature type
    color_map = {
        name: color for name, color in zip(
            feat_names,
            [(1, 0, 0, 0.75), (0, 1, 0, 0.75), (0, 0, 1, 0.75), (1, 1, 0, 0.75), (1, 0, 1, 0.75)]
        )
    }

    # Row 0: support and queries
    axes[0, 0].imshow(support_img_np)
    axes[0, 0].set_title("Support")
    for i, img in enumerate(query_imgs_np):
        axes[0, i+1].imshow(img)
        axes[0, i+1].set_title(f"Query {i+1}")

    # Unified PCA projections of support features
    # concatenate support and query into a checkerboard pattern of 2x2
    checkerboard = {}
    for key in support_feats.keys():
        # Interpolate all features to the support feature's shape for this key
        H, W = support_shapes[key]
        def interp(feat):
            if feat.shape[1:] != (H, W):
                return torch.nn.functional.interpolate(feat.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            return feat
        top = torch.cat([support_feats[key], interp(query_feats[0][key])], dim=1)
        bottom = torch.cat([interp(query_feats[1][key]), interp(query_feats[2][key])], dim=1)
        checkerboard[key] = torch.cat([top, bottom], dim=2)
    support_pca_imgs = shared_pca_projection(checkerboard)

    # Compute similarities
    similarities_by_feat = {}
    query_shapes_by_feat = {}
    for feat_name in feat_names:
        C = support_channels[feat_name]
        H, W = support_shapes[feat_name]
        support_feat = support_feats[feat_name]
        # Interpolate query features to support feature's shape if needed
        query_feat_list = []
        query_feat_shapes = []
        for qf in query_feats:
            q_feat = qf[feat_name]
            if q_feat.shape[1:] != (H, W):
                q_feat = torch.nn.functional.interpolate(q_feat.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            query_feat_list.append(q_feat)
            query_feat_shapes.append(q_feat.shape[1:])
        # Now permute and reshape for similarity
        support_feat_ = support_feat.permute(1, 2, 0)[None, None]
        query_feat_list_ = [qf.permute(1, 2, 0)[None, None] for qf in query_feat_list]
        sims = []
        for q_feat, (qh, qw) in zip(query_feat_list_, query_feat_shapes):
            sim = chunk_cosine_sim(
                support_feat_.view(1, 1, H * W, C),
                q_feat.view(1, 1, qh * qw, C)
            )[0, 0].view(H, W, qh * qw)
            sims.append(sim)
        similarities_by_feat[feat_name] = sims
        query_shapes_by_feat[feat_name] = query_feat_shapes

    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    load_size = (support_img_np.shape[0], support_img_np.shape[1])

    while len(pts) == 1:
        for axi in axes.ravel():
            axi.clear()
            axi.set_axis_off()

        axes[0, 0].imshow(support_img_np)
        for i, img in enumerate(query_imgs_np):
            axes[0, i+1].imshow(img)

        # Use support feature's H, W for support location
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        # For each feature type, use its support H, W for support location, and query H, W for reshaping
        center = (x_coor, y_coor)
        axes[0, 0].add_patch(Circle(center, radius, color=(0, 0, 0, 0.4)))

        for row_idx, feat_name in enumerate(feat_names):
            color = color_map[feat_name]
            H, W = support_shapes[feat_name]
            # Map click to support feature coordinates
            y_descs_coor = int(y_coor / load_size[0] * H)
            x_descs_coor = int(x_coor / load_size[1] * W)

            # Highlight support location on support RGB
            axes[0, 0].add_patch(Circle(center, radius, color=color))

            # Show PCA of support feature
            axes[row_idx+1, 0].imshow(support_pca_imgs[feat_name])
            axes[row_idx+1, 0].set_title(f"Support {feat_name}", color=color[:-1])
            # axes[row_idx+1, 0].add_patch(Rectangle((0, 0), W, H, linewidth=2, edgecolor=color, facecolor='none'))
            # axes[row_idx+1, 0].add_patch(Circle(center, radius, color=color))

            for i in range(num_queries):
                axes[0, i+1].imshow(query_imgs_np[i])
                qh, qw = query_shapes_by_feat[feat_name][i]
                sim = similarities_by_feat[feat_name][i][y_descs_coor, x_descs_coor]
                sim = torch.softmax(sim, dim=0).view(qh, qw)
                sim_img = sim.cpu().numpy()

                axes[row_idx+1, i+1].imshow(sim_img, cmap='jet', interpolation='nearest', alpha=0.5, extent=(0, qw, 0, qh))
                axes[row_idx+1, i+1].set_title(f"{feat_name} Q{i+1}")

                # Best match in query i
                sim_vec = similarities_by_feat[feat_name][i][y_descs_coor, x_descs_coor]  # (qh*qw,)
                max_idx = int(torch.argmax(sim_vec))

                # 2D indices in the query feature grid
                yq_feat = max_idx // qw
                xq_feat = max_idx % qw

                # Map feature-grid cell center -> query image pixels
                Hq_img, Wq_img = query_imgs_np[i].shape[:2]
                yq_img = (yq_feat + 0.5) * (Hq_img / max(qh, 1))
                xq_img = (xq_feat + 0.5) * (Wq_img / max(qw, 1))

                axes[0, i + 1].add_patch(Circle((xq_img, yq_img), radius, color=color))

        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", default='data/', type=str, help="Path to COLMAP data")
    parser.add_argument("--scene", default='pikachu', type=str, help="Name of scene")
    parser.add_argument("--exp_directory", default="experiments_runai", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--fit3d", action="store_true", help="Use FiT3D model for feature extraction")
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint instead of best")
    parser.add_argument("--lora", action="store_true", help="Use FiT3D model for feature extraction")
    parser.add_argument("--lora_exp_name", default=None, type=str)
    parser.add_argument("--stride_override", default=None, type=int, help="DINO model stride")
    parser.add_argument(
        "--override_image_size",
        nargs=2,  # expect exactly two values
        type=int,  # convert each to int
        metavar=("W", "H"),
        default=None,
        help="Override image size as W H (e.g., 640 480)",
    )
    parser.add_argument("--split", default="test", type=str, help="Dataset split to use")
    # ------------------------------------------------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------------------------------------------------
    args, _ = parser.parse_known_args()
    split_json_path = os.path.join(args.colmap_path, "splits.json")
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"split.yaml not found at {split_json_path}")
    with open(split_json_path) as json_file:
        split_data = json.load(json_file)

    if args.scene not in split_data[args.split]:
        raise ValueError(f"Scene {args.scene} not found in split.yaml under {args.split} split, try one of {split_data[args.split]}")

    args_yaml_path = os.path.join(args.exp_directory, args.exp_name, "args.yaml")
    if not os.path.exists(args_yaml_path):
        raise FileNotFoundError(f"args.yaml not found at {args_yaml_path}")
    yaml_args = load_config(args_yaml_path)
    yaml_args["scene"] = args.scene
    merged_args = {**yaml_args, **vars(args)}
    args = SafeNamespace(**merged_args)

    if args.stride_override is not None:
        args.stride = args.stride_override
    if args.override_image_size is not None:
        args.image_size = args.override_image_size
    # print("args.image_size:", args.image_size)
    # ------------------------------------------------------------------------------------------------------------
    # Load the model
    # ------------------------------------------------------------------------------------------------------------
    model = get_dino3d_model(args).to(device)
    model.eval()

    checkpoint_dir = os.path.join(args.exp_directory, args.exp_name, "checkpoints/")
    checkpointer = CheckPoint(checkpoint_dir)
    model = checkpointer.load_model(model, latest=args.latest)

    base_model = DINO(
        model_name=args.model_name,
        output=args.model_output_type,
        return_multilayer=False,
        stride=args.stride,
    )
    base_model = base_model.to(device)
    base_model.eval()

    if args.fit3d:
        fit3d_model = FiT3D("dinov2_reg_small")
        fit3d_model.eval()
        fit3d_model.to(device)
        fit3d_model.vit.patch_embed.proj.stride = (model.stride, model.stride)

    if args.lora:
        lora_model = get_lora_dino_model(args).to(device)
        lora_checkpoint_dir = os.path.join(args.exp_directory, args.lora_exp_name, "checkpoints/")
        lora_checkpointer = CheckPoint(lora_checkpoint_dir)
        lora_model = checkpointer.load_model(lora_model)
        lora_model.eval()

    # ------------------------------------------------------------------------------------------------------------
    # Build the dataset
    # ------------------------------------------------------------------------------------------------------------
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
    )[0]

    rand_img_idx = random.randint(0, len(test_dataset) - 1)
    data = test_dataset[rand_img_idx]
    data = move_to(data, device)
    intrinsics = data["intrinsics"][None] if "intrinsics" in data else None
    extrinsics = data["extrinsics"][None] if "extrinsics" in data else None

    # ------------------------------------------------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        features_3d = model(data["images"][None], data["plucker"][None], intrinsics=intrinsics, extrinsics=extrinsics)[0]
        # If FiT3D model is provided, extract features from it
        if args.fit3d or args.lora:
            B, S = data["images"][None].shape[:2]
            images_flat = rearrange(data["images"][None], "b s c h w -> (b s) c h w")
            if args.lora:
                additional_features = lora_model(images_flat)[0]
            else:
                additional_features = fit3d_model(data["images"], return_channel_first=True, return_dict=True)
                additional_features = additional_features["raw_vit_feats_fine"].squeeze(0)
        features_2d = base_model(data["images"])

    # Choose 4 out of args.n_dim
    if args.n_dim < 4:
        raise ValueError("Need at least 4 images in the batch for this visualization.")
    selected_indices = random.sample(range(args.n_dim), 4)
    support_idx = selected_indices[0]
    query_indices = selected_indices[1:]

    # Back to normal images from imagenet normalized images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    data["images"] = data["images"] * std + mean
    support_image = data["images"][support_idx]
    query_images = [data["images"][i] for i in query_indices]
    support_feature = features_3d[support_idx]
    query_features = [features_3d[i] for i in query_indices]
    support_base_feature = features_2d[support_idx]
    query_base_features = [features_2d[i] for i in query_indices]
    if args.fit3d or args.lora:
        print(additional_features.shape, support_idx, query_indices)
        support_additional_feature = additional_features[support_idx]
        query_additional_features = [additional_features[i] for i in query_indices]

    # Organize it into dicts for visualization
    support_feats = {
        "base": support_base_feature,
        "ours": support_feature,
    }
    query_feats = []
    for i in range(len(query_images)):
        query_feats.append({
            "base": query_base_features[i],
            "ours": query_features[i],
        })

    if args.fit3d or args.lora:
        name = "lora" if args.lora else "fit3d"
        support_feats[name] = support_additional_feature
        for i in range(len(query_images)):
            query_feats[i][name] = query_additional_features[i]


    show_similarity_interactive(
        support_image,
        query_images,
        support_feats,
        query_feats,
        feat_names=list(support_feats.keys())
    )