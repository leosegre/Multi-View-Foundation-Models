import argparse
import json
import os
import random
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from dino3d.checkpointing import CheckPoint
from dino3d.datasets.colmapdata import COLMAPBuilder
from dino3d.models.utils.utils import get_dino3d_model
from dino3d.train import move_to
# Try to import huggingface_hub, install if missing
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("📦 Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download


pretrained_checkpoints = {
    "dinov2_reg": "Leoseg/dinov2_reg",
    "dinov2_reg_no_plucker": "Leoseg/dinov2_reg_no_plucker",
    "dinov3": "Leoseg/dinov3",
    "clip": "Leoseg/clip",
    "sam": "Leoseg/sam",
}


class SafeNamespace(argparse.Namespace):
    def __getattr__(self, name):
        return False


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read())


def shared_pca_projection(feat_list):
    """Compute shared PCA across all features for visual consistency."""
    C, H, W = feat_list[0].shape
    all_feats = torch.stack(feat_list).permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(all_feats)

    # Normalize globally to [0, 1]
    pca_results -= pca_results.min(0, keepdims=True)
    pca_results /= (pca_results.max(0, keepdims=True) + 1e-5)

    return pca_results.reshape(len(feat_list), H, W, 3)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", default='data/', type=str)
    parser.add_argument("--scene", default='pikachu', type=str)
    parser.add_argument("--exp_directory", default="experiments_runai", type=str)
    parser.add_argument("--exp_name", default=None, type=str, help="Experiment name to load - for pretrained models, use --load_pretrained and exp_name={dinov2_reg, dinov2_reg_no_plucker, dinov3, clip, sam}")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--stride_override", default=7, type=int)
    parser.add_argument("--override_image_size", nargs=2, type=int, metavar=("W", "H"), default=None)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--limit_corrs", default=None, type=int)
    parser.add_argument("--load_pretrained", action="store_true")



    # 1. Parse and Merge Args using the SafeNamespace method
    args, _ = parser.parse_known_args()

    if args.load_pretrained:
        # --- Step 1: Download Model ---
        MODEL_REPO = pretrained_checkpoints[args.exp_name]
        print(f"\n⬇️  Downloading Model: {MODEL_REPO}...")
        # This downloads into {experiments_dir}/{experiment_name}/
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=args.exp_directory,
            local_dir_use_symlinks=False  # Download real files, not links
        )


    # Check scene in splits.json
    split_json_path = os.path.join(args.colmap_path, "splits.json")
    with open(split_json_path) as json_file:
        split_data = json.load(json_file)
    if args.scene not in split_data[args.split]:
        raise ValueError(f"Scene {args.scene} not in {args.split} split.")

    # Load and merge YAML
    args_yaml_path = os.path.join(args.exp_directory, args.exp_name, "args.yaml")
    yaml_args = load_config(args_yaml_path)
    yaml_args["scene"] = args.scene
    merged_args = {**yaml_args, **vars(args)}
    args = SafeNamespace(**merged_args)

    # Apply overrides
    if args.stride_override is not None:
        args.stride = args.stride_override
    if args.override_image_size is not None:
        args.image_size = args.override_image_size

    # 2. Load Model
    model = get_dino3d_model(args).to(device)
    checkpoint_dir = os.path.join(args.exp_directory, args.exp_name, "checkpoints/")
    model = CheckPoint(checkpoint_dir).load_model(model, latest=args.latest)
    model.eval()

    # 3. Build Dataset
    builder = COLMAPBuilder(colmap_root=args.colmap_path, scene_names=[args.scene])
    test_dataset = builder.build_scenes(
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        max_num_pairs=args.max_num_pairs,
        n_dim=args.n_dim,
        max_correspondences=args.max_correspondences,
        image_size=args.image_size,
        dino_output_type=args.model_output_type,
        split=args.split,
        limit_corrs=args.limit_corrs,
        dataset_size=1,
    )[0]

    data = move_to(test_dataset[random.randint(0, len(test_dataset) - 1)], device)

    # 4. Extract Features
    with torch.no_grad():
        features_3d = model(
            data["images"][None],
            data["plucker"][None],
            intrinsics=data.get("intrinsics")[None] if "intrinsics" in data else None,
            extrinsics=data.get("extrinsics")[None] if "extrinsics" in data else None
        )[0]

    # 5. Visualization (Top: RGB, Bottom: Ours PCA)
    num_views = min(4, args.n_dim)
    pca_images = shared_pca_projection([features_3d[i] for i in range(num_views)])

    fig, axes = plt.subplots(2, num_views, figsize=(num_views * 4, 8))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    for i in range(num_views):
        # Image row
        rgb = (data["images"][i] * std + mean).permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"View {i}")
        axes[0, i].axis('off')

        # PCA row
        axes[1, i].imshow(pca_images[i])
        axes[1, i].set_title("Ours (PCA Features)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()