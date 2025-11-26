import os
import time
import json
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import io

import numpy as np
import torch
from PIL import Image

from dino3d.utils import get_tuple_transform_ops
from ..models.utils.geometry import get_plucker_coordinates
from ..utils.colmap_utils import read_cameras_binary, read_images_binary

import sys
from itertools import combinations


class ChunkedCOLMAPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        chunk_root: str,
        split: str = "train",
        n_dim: int = 4,
        max_correspondences: int = 20,
        image_size: tuple = (518, 518),
        dino_output_type: str = 'dense',
        dataset_size: int = 1000,
        use_epipolar_attention: bool = False,
        use_register_features: bool = True,
        load_corrs: bool = True,
        max_cached_chunks: int = 8,
        no_features: bool = True,
        preload_chunks: bool = False,
        preload_decode_images: bool = False,
        **kwargs,
    ):
        print(f"Loading Chunked COLMAP dataset from '{chunk_root}' split '{split}'")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = n_dim
        self.max_correspondences = max_correspondences
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.dino_output_type = dino_output_type
        self.use_epipolar_attention = use_epipolar_attention
        self.use_register_features = use_register_features
        self.load_corrs = load_corrs
        self.no_features = no_features
        self.preload_chunks_flag = preload_chunks
        self.preload_decode_images = preload_decode_images

        self.chunk_root = Path(chunk_root) / split
        
        # Auto-detect available data type by checking directory structure
        base_path = Path(chunk_root)
        split_path = base_path / split
        
        # Check for different data types in order of preference
        if (base_path / f"{split}_no_features").exists():
            self.chunk_root = base_path / f"{split}_no_features"
            self.detected_no_features = True
            self.detected_use_register = False
        elif (base_path / f"{split}_register").exists():
            self.chunk_root = base_path / f"{split}_register"
            self.detected_no_features = False
            self.detected_use_register = True
        elif split_path.exists():
            self.chunk_root = split_path
            self.detected_no_features = False
            self.detected_use_register = False
        else:
            raise ValueError(f"No valid data directory found in {chunk_root} for split {split}")
        
        # Override detected settings with user preferences if they conflict
        if hasattr(self, 'no_features') and self.no_features != self.detected_no_features:
            print(f"Warning: Requested no_features={self.no_features} but detected {self.detected_no_features} in data")
        if hasattr(self, 'use_register_features') and self.use_register_features != self.detected_use_register:
            print(f"Warning: Requested use_register_features={self.use_register_features} but detected {self.detected_use_register} in data")
        
        self.index_file = self.chunk_root / "index.json"

        assert self.index_file.exists(), f"Missing index file: {self.index_file}"

        with open(self.index_file, 'r') as f:
            self.index = json.load(f)

        self.keys = list(self.index.keys())
        print(f"Found {len(self.keys)} scenes in index.")

        # Cache for loaded chunks (LRU)
        self.max_cached_chunks = max_cached_chunks
        self.chunk_cache: OrderedDict = OrderedDict()
        # Per-chunk decoded image cache (maps chunk_rel -> {idx: tensor})
        self.decoded_cache = {}

        # Normalize and resize to DINO input size
        self.im_transform_ops = get_tuple_transform_ops(normalize=False, resize=image_size)

        # Optionally preload all chunks into memory
        if self.preload_chunks_flag:
            # Important: if you're using DataLoader with multiple workers and spawn/forkserver,
            # preloading will duplicate memory per worker process. Prefer num_workers=0 or
            # preload inside each worker carefully.
            print(f"Preloading all chunks into memory (decode_images={self.preload_decode_images})...")
            self.preload_chunks(decode=self.preload_decode_images)

    def load_chunk(self, key: str):
        # lookup chunk path for this key
        chunk_rel = self.index[key]
        chunk_path = self.chunk_root / chunk_rel

        # If the chunk is cached, use it and return the example
        if chunk_rel in self.chunk_cache:
            # Move to end to mark as recently used
            self.chunk_cache.move_to_end(chunk_rel)
            chunk_map = self.chunk_cache[chunk_rel]
            if key in chunk_map:
                return chunk_map[key]

        # Load chunk from disk and build a map key->example for O(1) lookup
        chunk = torch.load(chunk_path)
        chunk_map = {example["key"]: example for example in chunk}

        # Insert into LRU cache
        self.chunk_cache[chunk_rel] = chunk_map
        # Initialize decoded cache for this chunk
        self.decoded_cache[chunk_rel] = {}

        # Evict oldest if over budget
        while len(self.chunk_cache) > self.max_cached_chunks:
            old_chunk_rel, _ = self.chunk_cache.popitem(last=False)
            # remove decoded cache as well
            if old_chunk_rel in self.decoded_cache:
                try:
                    del self.decoded_cache[old_chunk_rel]
                except Exception:
                    pass

        if key not in chunk_map:
            raise ValueError(f"Key {key} not found in chunk {chunk_path}")
        return chunk_map[key]

    def get_batch_matches(self, image_indices, image_to_2D3D):
        matches = []
        for i, id1 in enumerate(image_indices):
            map1 = image_to_2D3D[id1]
            for j in range(i + 1, len(image_indices)):
                id2 = image_indices[j]
                map2 = image_to_2D3D[id2]
                shared = set(map1.keys()) & set(map2.keys())
                for pid in shared:
                    x1, y1 = map1[pid]
                    x2, y2 = map2[pid]
                    matches.append([id1, x1, y1, id2, x2, y2])
        if matches:
            return torch.tensor(matches, dtype=torch.float32)  # (A, 6)
        else:
            return torch.empty((0, 6), dtype=torch.float32)

    def calc_filtered_corr(self, scene, chosen_ids):
        image_to_2D3D = {}

        for idx, image_id in enumerate(chosen_ids):
            # Get the index in the scene
            scene_idx = (scene["image_ids"] == image_id).nonzero(as_tuple=True)[0].item()
            xys = scene["xys"][scene_idx]
            point3D_ids = scene["point3D_ids"][scene_idx]
            width = scene["widths"][scene_idx]
            height = scene["heights"][scene_idx]

            # Build mapping: 3D point ID: (u, v)
            id_to_uv = {}
            for xy, pid in zip(xys, point3D_ids):
                if pid == -1 or pid >= 18446744073709551615:  # UINT64_MAX
                    continue
                xy_norm = (xy[0] / width, xy[1] / height)
                id_to_uv[int(pid)] = tuple(xy_norm)
            image_to_2D3D[image_id] = id_to_uv

        pairwise_corr = self.get_batch_matches(chosen_ids, image_to_2D3D)

        return pairwise_corr

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        t0 = time.time()

        # Deterministic scene selection by wrapping index to number of keys
        scene_key = self.keys[int(idx) % len(self.keys)]
        scene = self.load_chunk(scene_key)

        num_images = len(scene["images"])

        if num_images < self.N:
            raise ValueError(f"Scene {scene['key']} has only {num_images} images, but N={self.N} requested.")

        # Sample N unique image indices using a per-worker seeded torch.Generator so
        # multiple DataLoader workers produce different samples deterministically.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        gen = torch.Generator()
        gen.manual_seed(int((idx + 1) + worker_id * 100000))
        perm = torch.randperm(num_images, generator=gen)
        chosen_img_idx = perm[: self.N].tolist()

        images = []
        intrs = []
        extrs = []
        hw = []
        chosen_ids = []

        # Compatibility: scene may contain 'extrinsics' or 'poses' (colmapdata.py uses poses)
        extr_key = "extrinsics" if "extrinsics" in scene else ("poses" if "poses" in scene else None)
        if extr_key is None:
            raise KeyError(f"Chunk for scene {scene.get('key', '<unknown>')} missing 'extrinsics' or 'poses' field")
        if "intrinsics" not in scene:
            raise KeyError(f"Chunk for scene {scene.get('key', '<unknown>')} missing 'intrinsics' field")

        for ii in chosen_img_idx:
            i = int(ii)
            # Decode image from raw bytes (use per-chunk decoded cache)
            img = self._get_decoded_image(scene, i)

            K = scene["intrinsics"][i]
            T = scene[extr_key][i]
            h, w = img.shape[-2:]
            K = self._scale_K(K, w, h)

            images.append(img)
            intrs.append(K)
            extrs.append(T)
            hw.append((h, w))
            chosen_ids.append(int(scene["image_ids"][i]))

        images = torch.stack(images)
        intrs = torch.stack(intrs)
        extrs = torch.stack(extrs)

        # Get pairwise correspondences
        pairwise_corr = []
        if self.load_corrs:
            for a, id1 in enumerate(chosen_ids):
                for b in range(a + 1, len(chosen_ids)):
                    id2 = chosen_ids[b]
                    key = (int(id1), int(id2))
                    if key in scene["matches"]:
                        match = scene["matches"][key]
                        pairwise_corr.append(match)
                    elif (int(id2), int(id1)) in scene["matches"]:
                        match = scene["matches"][(int(id2), int(id1))]
                        # Flip the matches
                        flipped_match = match.clone()
                        flipped_match[:, [0, 3]] = match[:, [3, 0]]
                        flipped_match[:, [1, 4]] = match[:, [4, 1]]
                        flipped_match[:, [2, 5]] = match[:, [5, 2]]
                        pairwise_corr.append(flipped_match)

        pairwise_corr = torch.cat(pairwise_corr, dim=0) if len(pairwise_corr) > 0 else torch.empty((0, 6))

        # If no matches found and load_corrs is True, calculate from xys and point3D_ids
        if pairwise_corr.shape[0] == 0 and self.load_corrs:
            pairwise_corr = self.calc_filtered_corr(scene, chosen_ids)

        # Sample max_correspondences
        if pairwise_corr.shape[0] > self.max_correspondences:
            choice = np.random.choice(pairwise_corr.shape[0], self.max_correspondences, replace=False)
            pairwise_corr = pairwise_corr[choice]

        # Use Plücker rays
        intrinsics_src = intrs[0][None]  # Use source pose intrinsics
        plucker = get_plucker_coordinates(
            extrs[0][None], extrs, intrinsics_src, target_size=hw[0]
        )

        sample = {
            "images": images,
            "plucker": plucker,
            "image_ids": torch.tensor(chosen_ids),
            "image_idx": torch.tensor(chosen_img_idx),
            "pairwise_correspondence": pairwise_corr,
        }

        # Optionally load and process base features (dense or dense-cls)
        if self.detected_no_features:
            base_features = None
            proc_base = None
        else:
            base_features = []
            for _, img_idx in enumerate(chosen_img_idx):
                if self.detected_use_register:
                    feat = scene.get("features_register", [None] * len(scene["images"]))[int(img_idx)]
                else:
                    feat = scene.get("features", [None] * len(scene["images"]))[int(img_idx)]
                base_features.append(feat if feat is not None else None)

            if all(f is not None for f in base_features):
                base_features = torch.stack([torch.as_tensor(f) if not isinstance(f, torch.Tensor) else f for f in base_features])
                C = base_features.shape[1]
                if self.dino_output_type == 'dense-cls':
                    proc_base = base_features
                elif self.dino_output_type == 'dense':
                    proc_base = base_features[:, : C // 2]
                elif self.dino_output_type == 'cls':
                    proc_base = base_features[:, C // 2:]
                else:
                    proc_base = base_features
            else:
                proc_base = None

        # Epipolar attention integration
        sample["intrinsics"] = intrs
        sample["extrinsics"] = extrs
        sample["base_features"] = proc_base
        return sample

    def _get_decoded_image(self, scene: dict, idx: int):
        """Return decoded and transformed image tensor for a given scene and view index.
        Uses a per-chunk decoded cache to avoid repeated decoding work.
        """
        chunk_rel = self.index[scene["key"]]
        # Initialize cache structures if needed
        if chunk_rel not in self.decoded_cache:
            self.decoded_cache[chunk_rel] = {}

        if idx in self.decoded_cache[chunk_rel]:
            return self.decoded_cache[chunk_rel][idx]

        img_bytes = scene["images"][idx]
        img = Image.open(io.BytesIO(img_bytes.numpy().tobytes())).convert('RGB')
        img, = self.im_transform_ops((img,))

        # store in decoded cache
        self.decoded_cache[chunk_rel][idx] = img
        return img

    def _scale_K(self, K, w, h):
        sx, sy = w / self.image_size[1], h / self.image_size[0]
        scale = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=K.dtype)
        return scale @ K

    def preload_chunks(self, decode: bool = False):
        """Preload all chunks into the LRU cache. If decode=True, also decode and transform images.

        Use with caution: will consume memory approximately equal to the sum of chunk file sizes
        plus decoded image memory. Not recommended with multiple DataLoader workers unless
        using shared memory / careful forking semantics.
        """
        # Load each chunk file from disk
        base_path = self.chunk_root
        chunk_files = sorted([p for p in base_path.iterdir() if p.suffix == '.torch'])
        for chunk_path in chunk_files:
            rel = str(chunk_path.name)
            # Avoid reloading if cached
            if rel in self.chunk_cache:
                continue
            chunk = torch.load(chunk_path)
            # Build map key->example
            chunk_map = {ex['key']: ex for ex in chunk}
            self.chunk_cache[rel] = chunk_map
            # Optionally decode and cache images per chunk
            if decode:
                self.decoded_cache[rel] = {}
                for key, ex in chunk_map.items():
                    imgs = ex.get('images', [])
                    for i, img_bytes in enumerate(imgs):
                        try:
                            img = Image.open(io.BytesIO(img_bytes.numpy().tobytes())).convert('RGB')
                            img_t, = self.im_transform_ops((img,))
                            self.decoded_cache[rel][i] = img_t
                        except Exception:
                            # keep decoding best-effort
                            pass
            # Evict if cache bigger than limit
            while len(self.chunk_cache) > self.max_cached_chunks:
                old_rel, _ = self.chunk_cache.popitem(last=False)
                if old_rel in self.decoded_cache:
                    try:
                        del self.decoded_cache[old_rel]
                    except Exception:
                        pass

