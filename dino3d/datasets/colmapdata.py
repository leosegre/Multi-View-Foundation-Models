import os
import time

import numpy as np
import torch
from PIL import Image

from dino3d.utils import get_tuple_transform_ops, get_depth_tuple_transform_ops
from ..models.utils.geometry import get_plucker_coordinates
from ..utils.colmap_utils import read_cameras_binary, read_images_binary

import json
from pathlib import Path
from itertools import combinations
from matplotlib import pyplot as plt
import sys
from safetensors.torch import load_file as load_st
from collections import defaultdict


MAX_INT = 2**31
UINT64_MAX = 18446744073709551615


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def visualize_correspondences(img1, img2, matches, title=None, dpi=100):
    """
    Visualizes matches between two images.

    Args:
        img1: Tensor or numpy array (3,H,W) or (H,W,3)
        img2: Tensor or numpy array (3,H,W) or (H,W,3)
        matches: Tensor of shape (N,6) → (id1, x1, y1, id2, x2, y2)
        title: Optional figure title
        dpi: Resolution for the plot
    """

    matches = matches[:24, ...]

    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()  # (C,H,W) → (H,W,C)
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()

    # Normalize if necessary
    if img1.max() > 1.5:
        img1 = img1 / 255.0
    if img2.max() > 1.5:
        img2 = img2 / 255.0

    # Concatenate images side-by-side
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    canvas = np.ones((canvas_height, canvas_width, 3))
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.imshow(canvas)
    ax.axis('off')

    # Plot lines between matching points
    for match in matches:
        x1, y1 = match[1:3].cpu().numpy()
        x2, y2 = match[4:6].cpu().numpy()

        x1 *= w1
        y1 *= h1
        x2 *= w2
        y2 *= h2

        color = np.random.rand(3,)
        ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=1)
        ax.scatter(x1, y1, color=color, s=5)
        ax.scatter(x2 + w1, y2, color=color, s=5)

    if title:
        ax.set_title(title)

    plt.show()

# Manual conversion from quaternion to rotation matrix
# (used instead of pycolmap.qvec2rotmat which may not exist)
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])


class COLMAPScene(torch.utils.data.Dataset):
    def __init__(
        self,
        colmap_root,
        image_dir,
        n_dim=4,
        max_correspondences=20,
        image_size=(224, 224),
        dino_output_type='dense',
        dataset_size=1000,
        use_epipolar_attention=False,
        use_register_features=True,
        load_corrs = False,
        normalize_plucker=False,
        cross_plucker=False,
        clip_norm=False,
        **kwargs
    ):
        print(f"Loading COLMAP scene from '{colmap_root}'")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = n_dim
        self.max_correspondences = max_correspondences
        self.dataset_size = 1000 if isinstance(dataset_size, bool) else dataset_size
        self.image_size = image_size

        self.colmap_root = colmap_root
        self.image_dir = image_dir
        self.matches_root = os.path.join(colmap_root, "matches")
        self.dino_output_type = dino_output_type
        self.use_epipolar_attention = use_epipolar_attention
        self.use_register_features = use_register_features
        self.load_corrs = load_corrs
        self.normalize_plucker = normalize_plucker
        self.cross_plucker = cross_plucker
        # Normalize and resize to DINO input size
        self.im_transform_ops = get_tuple_transform_ops(normalize=True, resize=image_size, clip_norm=clip_norm)


        # Load COLMAP cameras and images from .bin files
        self.cameras = read_cameras_binary(os.path.join(colmap_root, 'cameras.bin'))
        self.images = read_images_binary(os.path.join(colmap_root, 'images.bin'))



        # Initialize lists and maps for scene metadata
        self.image_list = []
        self.image_ids = []
        self.intrinsics = []
        self.image_hw = []
        self.poses = []
        image_id_to_idx = {}
        image_name_to_idx = {}
        image_to_2D3D = {}  # image index → {point3D_id: (x, y)}

        # Parse each image entry and build metadata
        for idx, (image_id, im) in enumerate(self.images.items()):
            image_id_to_idx[image_id] = idx
            image_name_to_idx[im['name']] = idx
            self.image_list.append(im['name'])
            self.image_ids.append(image_id)

            cam = self.cameras[im['camera_id']]
            K = np.array([[cam['params'][0], 0, cam['params'][2]],
                          [0, cam['params'][1], cam['params'][3]],
                          [0, 0, 1]])
            self.intrinsics.append(K)
            self.image_hw.append((cam['height'], cam['width']))

            pose = self._build_pose(im['qvec'], im['tvec'])
            self.poses.append(pose)



        self.intrinsics = np.stack(self.intrinsics)
        self.poses = np.stack(self.poses)


        # Convert list to numpy to prevent memory leaks
        self.image_hw = np.array(self.image_hw)
        self.image_list = np.array(self.image_list)
        self.image_ids = np.array(self.image_ids)

        # Load matches metadata
        mapping_path = os.path.join(colmap_root, "matches_mapping.json")
        assert os.path.exists(mapping_path), f"Missing matches_mapping.json at {mapping_path}"

        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        self.num_matches = mapping['num_matches']
        self.scene_name = mapping['scene_name']
        matches_mapping = mapping['matches']  # the { "id1,id2": {"path": path, "num_matches": num_matches} }


        # Array of chosen indexes for the dataset, used for sampling. size of [self.__len__(), N]
        self.chosen_idx = np.array([
            np.random.choice(len(self.image_list), size=self.N, replace=False)
            for _ in range(self.__len__())
        ])

        # Check that all matches have at least max_correspondences
        chosen_idx_ok = np.zeros((self.__len__(),), dtype=bool)
        while not np.all(chosen_idx_ok):
            for idx in np.argwhere(~chosen_idx_ok).flatten():
                num_matches = 0
                chosen_ids = self.image_ids[self.chosen_idx[idx]]
                for id1, id2 in combinations(chosen_ids, 2):
                    num_matches += self.get_num_matches_pair(id1, id2, matches_mapping)
                    if num_matches >= self.max_correspondences:
                        break
                if num_matches >= self.max_correspondences:
                    chosen_idx_ok[idx] = True
                else:
                    # If not ok, resample
                    self.chosen_idx[idx] = np.random.choice(len(self.image_list), size=self.N, replace=False)



    def _build_K(self, cam):
        # Build 3x3 intrinsics matrix from COLMAP camera parameters
        fx, fy, cx, cy = cam['params'][:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def _build_pose(self, qvec, tvec):
        # Convert COLMAP quaternion and translation to 4x4 transformation matrix
        # from pycolmap import qvec2rotmat  # Use COLMAP's own utility
        R = qvec2rotmat(qvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        return T  # c2w

    def get_num_matches_pair(self, id1, id2, matches_mapping):
        key = f"{id1},{id2}"
        if key not in matches_mapping:
            key = f"{id2},{id1}"
            if key not in matches_mapping:
                return 0
        return int(matches_mapping[key]["num_matches"])

    # def load_match(self, id1, id2, matches_mapping):
    #     key = f"{id1},{id2}"
    #     if key not in matches_mapping:
    #         key = f"{id2},{id1}"  # Try flipping
    #         if key not in matches_mapping:
    #             return torch.empty((0, 6))  # No match
    #     path = os.path.join(self.colmap_root, matches_mapping[key]["path"])
    #     matches = torch.load(path)
    #     # Normalize mathces to [0, 1] by dividing 224 -> a hard coded value from the script generate_corrs.py
    #     matches[:, 1:3] = matches[:, 1:3] / 224
    #     matches[:, 4:6] = matches[:, 4:6] / 224
    #     return matches

    def load_match(self, id1, id2):
        # Try the original order first
        filename = f"{id1}_{id2}.pt"
        path = os.path.join(self.matches_root, filename)

        if not os.path.exists(path):
            # Try flipping the order
            filename = f"{id2}_{id1}.pt"
            path = os.path.join(self.matches_root, filename)

            if not os.path.exists(path):
                return torch.empty((0, 6))  # No match

        matches = torch.load(path)
        # Normalize matches to [0, 1] by dividing 224 -> a hard coded value from the script generate_corrs.py
        matches[:, 1:3] = matches[:, 1:3] / 224
        matches[:, 4:6] = matches[:, 4:6] / 224
        return matches

    def get_batch_matches(self, image_indices, image_to_2D3D):
        # Build inverted index: point_id -> [(image_id, x, y), ...]
        point_to_observations = defaultdict(list)

        for image_id in image_indices:
            id_to_uv = image_to_2D3D[image_id]
            for pid, (x, y) in id_to_uv.items():
                point_to_observations[pid].append((image_id, x, y))

        # Filter to only points observed in 2+ images
        multi_view_points = {pid: obs for pid, obs in point_to_observations.items() if len(obs) >= 2}

        if not multi_view_points:
            return torch.empty((0, 6), dtype=torch.float32)

        # Strategy 1: Sample points first (good when max_correspondences << total_matches)
        total_possible_matches = sum(len(obs) * (len(obs) - 1) // 2 for obs in multi_view_points.values())

        if hasattr(self, 'max_correspondences') and self.max_correspondences < total_possible_matches:
            # Random sampling approach
            all_matches = []

            # Collect all potential matches with their point weights
            point_items = list(multi_view_points.items())

            # Shuffle points for random selection
            np.random.shuffle(point_items)

            matches_collected = 0
            for pid, observations in point_items:
                if matches_collected >= self.max_correspondences:
                    break

                # Convert to numpy array for vectorized operations
                obs_array = np.array(observations, dtype=np.float32)
                n_obs = len(observations)

                # Calculate how many matches this point contributes
                point_matches = n_obs * (n_obs - 1) // 2
                remaining_needed = self.max_correspondences - matches_collected

                if point_matches <= remaining_needed:
                    # Take all matches from this point
                    for i in range(n_obs):
                        for j in range(i + 1, n_obs):
                            all_matches.append([
                                obs_array[i, 0], obs_array[i, 1], obs_array[i, 2],
                                obs_array[j, 0], obs_array[j, 1], obs_array[j, 2]
                            ])
                    matches_collected += point_matches
                else:
                    # Randomly sample from this point's matches
                    point_match_pairs = []
                    for i in range(n_obs):
                        for j in range(i + 1, n_obs):
                            point_match_pairs.append((i, j))

                    # Randomly select subset
                    selected_pairs = np.random.choice(len(point_match_pairs),
                                                      remaining_needed, replace=False)

                    for idx in selected_pairs:
                        i, j = point_match_pairs[idx]
                        all_matches.append([
                            obs_array[i, 0], obs_array[i, 1], obs_array[i, 2],
                            obs_array[j, 0], obs_array[j, 1], obs_array[j, 2]
                        ])
                    matches_collected = self.max_correspondences
                    break

            if all_matches:
                matches = np.array(all_matches, dtype=np.float32)
            else:
                matches = np.empty((0, 6), dtype=np.float32)
        else:
            # Original approach when we want all matches
            total_matches = total_possible_matches
            matches = np.empty((total_matches, 6), dtype=np.float32)
            match_idx = 0

            for pid, observations in multi_view_points.items():
                obs_array = np.array(observations, dtype=np.float32)
                n_obs = len(observations)
                for i in range(n_obs):
                    for j in range(i + 1, n_obs):
                        matches[match_idx] = [
                            obs_array[i, 0], obs_array[i, 1], obs_array[i, 2],
                            obs_array[j, 0], obs_array[j, 1], obs_array[j, 2]
                        ]
                        match_idx += 1

            matches = matches[:match_idx]

        return torch.from_numpy(matches)

    def calc_filtered_corr(self, idx):
        chosen_ids = self.image_ids[self.chosen_idx[idx]]
        images = {int(i): self.images[int(i)] for i in chosen_ids if int(i) in self.images}

        image_to_2D3D = {}

        for image_id, im in images.items():
            cam = self.cameras[im['camera_id']]

            # Get valid points (vectorized filtering)
            xys = np.array(im['xys'], dtype=np.float32)
            point_ids = np.array(im['point3D_ids'])

            # Filter out invalid points
            valid_mask = (point_ids != -1) & (point_ids < UINT64_MAX)
            valid_xys = xys[valid_mask]
            valid_pids = point_ids[valid_mask]

            # Normalize coordinates (vectorized)
            normalized_xys = valid_xys / np.array([cam['width'], cam['height']], dtype=np.float32)

            # Build mapping efficiently
            id_to_uv = dict(zip(valid_pids, normalized_xys))
            image_to_2D3D[image_id] = id_to_uv

        pairwise_corr = self.get_batch_matches_reservoir(chosen_ids, image_to_2D3D)

        return pairwise_corr, chosen_ids

    # Alternative: Ultra-efficient reservoir sampling approach
    def get_batch_matches_reservoir(self, image_indices, image_to_2D3D):
        """Most efficient when max_correspondences << total_matches"""
        # Build inverted index
        point_to_observations = defaultdict(list)

        for image_id in image_indices:
            id_to_uv = image_to_2D3D[image_id]
            for pid, (x, y) in id_to_uv.items():
                point_to_observations[pid].append((image_id, x, y))

        # Filter to multi-view points
        multi_view_points = {pid: obs for pid, obs in point_to_observations.items() if len(obs) >= 2}

        if not multi_view_points:
            return torch.empty((0, 6), dtype=torch.float32)

        if not hasattr(self, 'max_correspondences'):
            # Fall back to original method
            return self._get_all_matches(multi_view_points)

        # Reservoir sampling for fixed-size random sample
        reservoir = []
        total_seen = 0

        for pid, observations in multi_view_points.items():
            obs_array = np.array(observations, dtype=np.float32)
            n_obs = len(observations)

            # Generate all pairs for this point
            for i in range(n_obs):
                for j in range(i + 1, n_obs):
                    match = [obs_array[i, 0], obs_array[i, 1], obs_array[i, 2],
                             obs_array[j, 0], obs_array[j, 1], obs_array[j, 2]]

                    if len(reservoir) < self.max_correspondences:
                        # Still filling reservoir
                        reservoir.append(match)
                    else:
                        # Randomly replace existing match
                        replace_idx = np.random.randint(0, total_seen + 1)
                        if replace_idx < self.max_correspondences:
                            reservoir[replace_idx] = match

                    total_seen += 1

        if reservoir:
            matches = np.array(reservoir, dtype=np.float32)
        else:
            matches = np.empty((0, 6), dtype=np.float32)

        return torch.from_numpy(matches)

    def _get_all_matches(self, multi_view_points):
        """Helper for getting all matches when no limit"""
        total_matches = sum(len(obs) * (len(obs) - 1) // 2 for obs in multi_view_points.values())
        matches = np.empty((total_matches, 6), dtype=np.float32)
        match_idx = 0

        for pid, observations in multi_view_points.items():
            obs_array = np.array(observations, dtype=np.float32)
            n_obs = len(observations)
            for i in range(n_obs):
                for j in range(i + 1, n_obs):
                    matches[match_idx] = [
                        obs_array[i, 0], obs_array[i, 1], obs_array[i, 2],
                        obs_array[j, 0], obs_array[j, 1], obs_array[j, 2]
                    ]
                    match_idx += 1

        return torch.from_numpy(matches[:match_idx])


    def get_filtered_corr(self, idx):
        chosen_ids = self.image_ids[self.chosen_idx[idx]]
        # Load matches for the chosen image pairs
        pairwise_corr = []
        for id1, id2 in combinations(chosen_ids, 2):
            match = self.load_match(id1, id2)
            if match.shape[0] > 0:
                pairwise_corr.append(match)

        # Concatenate all matches
        pairwise_corr = torch.cat(pairwise_corr, dim=0) if len(pairwise_corr) > 0 else torch.empty((0, 6))

        return pairwise_corr, chosen_ids

    def process_base_features(self, base_features):
        # Features are dense-cls -> convert to args.dino_output_type
        # Features are [spatial_tokens, cls_token], concatenated along the channel dimension
        C, H, W = base_features.shape
        if self.dino_output_type == 'dense-cls':
            return base_features
        elif self.dino_output_type == 'dense':
            return base_features[: C // 2]
        elif self.dino_output_type == 'cls':
            return base_features[C // 2:]
        else:
            raise ValueError(f"Unknown dino_output_type: {self.dino_output_type}")

    def __len__(self):
        # return (int(self.num_matches) // self.max_correspondences) // 35
        return self.dataset_size

    def __getitem__(self, idx):
        t0 = time.time()
        # Note that seed is curently not used, can be used if N is set to the scene and all permutations are precomputed
        assert self.N <= len(self.image_list), f"Requested N={self.N}, but only {len(self.image_list)} images are available in {self.scene_name}."

        # # Load matches metadata
        # mapping_path = os.path.join(self.colmap_root, "matches_mapping.json")
        # assert os.path.exists(mapping_path), f"Missing matches_mapping.json at {mapping_path}"
        # with open(mapping_path, 'r') as f:
        #     mapping = json.load(f)
        #     matches_mapping = mapping['matches']

        # Randomly sample N unique image indices

        if self.load_corrs:
            pairwise_corr, chosen_ids = self.get_filtered_corr(idx)
        else:
            pairwise_corr, chosen_ids = self.calc_filtered_corr(idx)


        chosen_idx = self.chosen_idx[idx]

        # # make sure that pairwise_corr >= max_correspondences, else choose new ids until it is (or up to 10 times)
        # while pairwise_corr.shape[0] < self.max_correspondences:
        #     pairwise_corr, chosen_ids = self.get_filtered_corr(idx)

        choice = np.random.choice(pairwise_corr.shape[0],
                                  min(pairwise_corr.shape[0], self.max_correspondences),
                                  replace=False)
        pairwise_corr = pairwise_corr[choice]
        # if len(pairwise_corr) < self.max_correspondences:
        #     # Pad with nans
        #     pad = torch.full((self.max_correspondences - len(pairwise_corr), 6), np.nan)
        #     pairwise_corr = torch.vstack((pairwise_corr, pad))

        images = []
        # base_features = []
        intrs = []
        extrs = []
        hw = []

        for i in chosen_idx:
            name = self.image_list[i]
            path = os.path.join(self.image_dir, name)
            im = Image.open(path).convert('RGB')
            im, = self.im_transform_ops((im,))

            # feat_name = os.path.splitext(name)[0]
            # if self.use_register_features:
            #     feat_folder = os.path.join(Path(self.image_dir).parent, "features_register")
            # else:
            #     feat_folder = os.path.join(Path(self.image_dir).parent, "features")
            #
            # feat_path = os.path.join(feat_folder, f"{feat_name}.pt")
            # feat = torch.load(feat_path)
            #
            # Create dummy features
            # feat = torch.zeros((768, 37, 37) , dtype=torch.float32)
            #
            # feat = self.process_base_features(feat)

            K = torch.tensor(self.intrinsics[i], dtype=torch.float32)
            T = torch.tensor(self.poses[i], dtype=torch.float32)
            h, w = im.shape[-2:]
            # K = self._scale_K(K, w, h)
            K = self._scale_K(K, 1, 1) # Intrinsics should be in normalized image coordinates

            images.append(im)
            # base_features.append(feat)
            intrs.append(K)
            extrs.append(T)
            hw.append((h, w))


        images = torch.stack(images)
        intrs = torch.stack(intrs)
        extrs = torch.stack(extrs)
        # base_features = torch.stack(base_features)

        # visualize_correspondences(images[0], images[1], pairwise_corr, title=f"num matches: {len(pairwise_corr)}")

        # Use Plücker rays (optional)

        camera_scale = 2.0  # Scale cameras to roughly fit in a unit sphere
        # c2w = extrs
        c2w = torch.linalg.inv(extrs)

        # camera centering
        ref_c2ws = c2w
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)

        # camera normalization
        camera_dists = c2w[:, :3, 3].clone()
        translation_scaling_factor = (
            camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]),
                torch.zeros(1),
                atol=1e-5,
            ).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= translation_scaling_factor
        c2w[:, :3, 3] *= translation_scaling_factor
        
        intrinsics_src = intrs[0][None]  # Use source pose intrinsics
        plucker = get_plucker_coordinates(
            w2c[0], w2c, intrinsics_src.float().clone(), target_size=hw[0], normalize_plucker=self.normalize_plucker
        )

        if self.cross_plucker:
            # Randomly choose a source view for each target
            i = np.random.randint(1, self.N)
            cross_plucker = get_plucker_coordinates(
            w2c[i], w2c, intrinsics_src.float().clone(), target_size=hw[0], normalize_plucker=self.normalize_plucker
            )
        else:
            cross_plucker = torch.tensor([0])

        sample = {
            "images": images,
            # "base_features": base_features,
            "plucker": plucker,
            "cross_plucker": cross_plucker,
            "image_ids": torch.from_numpy(chosen_ids),
            # "image_idx": torch.from_numpy(chosen_idx),
            "pairwise_correspondence": pairwise_corr,
        }
        # Epipolar attention integration: always include intrinsics/extrinsics for compatibility
        # This is safe for backward compatibility: downstream code can ignore these if not needed
        sample["intrinsics"] = intrs
        sample["extrinsics"] = extrs
        return sample

    def _scale_K(self, K, w, h):
        sx, sy = w / self.image_hw[0][1], h /self.image_hw[0][0]
        scale = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=K.dtype)
        return scale @ K

class COLMAPBuilder:
    def __init__(self, colmap_root, scene_names=None) -> None:
        self.colmap_root = colmap_root
        self.scene_names = scene_names

    def build_scenes(self, split="train", **kwargs):
        # Load split info from JSON
        if "debug_small_data" in kwargs and kwargs["debug_small_data"]:
            splits_path = "splits_debug.json"
        else:
            splits_path = "splits.json"
        split_file = Path(self.colmap_root) / splits_path
        assert split_file.exists(), f"Missing split file: {split_file}"

        with open(split_file, 'r') as f:
            splits = json.load(f)

        assert split in splits, f"Split '{split}' not found in split file. Available: {list(splits.keys())}"


        scenes = []
        for dataset_name in sorted([d.name for d in Path(self.colmap_root).iterdir() if d.is_dir()]):
            scene_names = self.scene_names if self.scene_names is not None else sorted([d.name for d in (Path(self.colmap_root) / dataset_name).iterdir() if d.is_dir()])
            # scene_names = self.scene_names if self.scene_names is not None else np.array(sorted(os.listdir(self.colmap_root + dataset_name)))
            # Filter scene names based on the split
            scene_names = [scene for scene in scene_names if scene in splits[split]]
            for scene_name in scene_names:
                scene_path = os.path.join(self.colmap_root, dataset_name, scene_name, "colmap", "sparse", "0")
                # Load an image and check for the side, check the largest downscale with H and W larger than 224, options are 2, 4, 8
                image_path = os.path.join(self.colmap_root, dataset_name, scene_name, "images")

                if not (os.path.exists(image_path)):
                    print(f"[WARN] Skipping scene '{dataset_name}/{scene_name}' due to missing images files.")
                    continue


                first_image = os.path.join(image_path, os.listdir(image_path)[0])
                im = Image.open(first_image)
                w, h = im.size
                del im
                # Check for downscale
                min_size = torch.min(torch.tensor([w, h]))
                max_downscale = min_size // min(kwargs["image_size"])
                if max_downscale >= 8:
                    downscale = 8
                elif max_downscale >= 4:
                    downscale = 4
                elif max_downscale >= 2:
                    downscale = 2
                else:
                    downscale = 1
                if downscale > 1:
                    image_path = os.path.join(self.colmap_root, dataset_name, scene_name, "images_" + str(downscale))
                    if not os.path.exists(image_path):
                        print(f"[WARN] Skipping scene '{scene_name}' due to missing downscaled images.")
                        continue

                cameras_file = os.path.join(scene_path, "cameras.bin")
                images_file = os.path.join(scene_path, "images.bin")

                if not (os.path.exists(cameras_file) and os.path.exists(images_file)):
                    print(f"[WARN] Skipping scene '{scene_name}' due to missing COLMAP files.")
                    continue

                print("[build] Creating new COLMAPScene...")
                scene = COLMAPScene(
                    colmap_root=scene_path,
                    image_dir=image_path,
                    **kwargs
                )
                # print(f"[build] Scene '{scene_name}' loaded with {len(scene.image_list)} images and {scene.num_matches} matches.")
                # print(f"Scene size is: ", get_size(scene) / (1024*1024), "MB")
                # print(f"Scene matches_mapping dict size is: ", get_size(scene.matches_mapping) / (1024*1024), "MB")

                # batch = scene[0]
                #
                #
                # match = batch['pairwise_correspondence']  # (N,6)
                # id1, id2 = batch['image_ids']  # (2,)
                # idx1, idx2 = batch['image_idx']  # (2,)
                # if id2 < id1:
                #     id1, id2 = id2, id1
                #     idx1, idx2 = idx2, idx1
                #
                # # Load images manually:
                # img1_path = os.path.join(scene.image_dir, scene.image_list[idx1])
                # img2_path = os.path.join(scene.image_dir, scene.image_list[idx2])
                #
                # # Print the images names
                # print(f"Image 1: {scene.image_list[idx1]}, ID: {id1}")
                # print(f"Image 2: {scene.image_list[idx2]}, ID: {id2}")
                #
                # img1 = Image.open(img1_path).convert('RGB')
                # img2 = Image.open(img2_path).convert('RGB')
                #
                # transform = get_tuple_transform_ops(normalize=False, resize=(224, 224))  # your transform
                # img1, = transform((img1,))
                # img2, = transform((img2,))
                #
                # visualize_correspondences(img1, img2, match)

                scenes.append(scene)

        return scenes
