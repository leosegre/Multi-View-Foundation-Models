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
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
    build_pose,
    scale_K,
)
from dino3d.models.utils.geometry import get_plucker_coordinates
from torchvision.transforms.functional import to_tensor


class NAVI3D(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        model="all",
        image_mean="imagenet",
        augment_train=False,
        rotateflip=False,
        bbox_crop=True,
        pair_dataset=False,
        max_angle=120,
        relative_depth=False,
        n_dim=4,
        world_space_normals=True,  # Use world-space normals for multi-view consistency (Dino3D)
    ):
        super().__init__()

        # generate splits based on collections and subparts
        if split == "train":
            collection = "multiview"
            subpart = "train"
        elif split == "valid":
            collection = "multiview"
            subpart = "test"
        elif split == "trainval":
            collection = "multiview"
            subpart = "train"
        elif split == "test":
            collection = "multiview"
            subpart = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data_root = Path(path)
        self.bbox_crop = bbox_crop
        self.relative_depth = relative_depth
        self.max_depth = 1.0
        self.n_dim = n_dim
        self.world_space_normals = world_space_normals  # Transform normals to world space for 3D models

        self.name = f"NAVI_{collection}_{subpart}"
        if relative_depth:
            self.name = self.name + "_reldepth"

        # parse dataset
        self.data_dict = self.parse_dataset()
        self.define_instances_split(model, collection, subpart)

        # get transforms
        augment = augment_train and "train" in split
        image_size = (518, 518)
        t_fns = get_navi_transforms(
            image_mean,
            image_size=image_size,
            augment=augment,
            rotateflip=rotateflip,
            additional_targets={
                "depth": "image",
                "snorm": "image",
                "xyz_grid": "image",
                "plucker": "image",
            },
        )
        self.image_transform, self.target_transform, self.shared_transform = t_fns
        
        self.pair_dataset = pair_dataset
        self.max_angle = max_angle
        if self.pair_dataset:
            self.pair_indices = self.generate_instance_pairs(self.instances)

        # Removed [::4] subsampling - now we have proper sliding windows
        print(f"NAVI {collection} {subpart} {model}: {len(self.instances)} instances (sliding windows)")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        obj_id, scene_id, window_views = self.instances[index]
        
        # For wild_set (None window_views), use random sampling as before
        # print(window_views)
        if window_views is None:
            views = self.data_dict[obj_id][scene_id]["views"]
            chosen_img_ids = np.random.choice(views, size=min(self.n_dim, len(views)), replace=False)
        else:
            # For multiview, use the pre-computed sliding window
            chosen_img_ids = window_views

        images = []
        depths = []
        snorms = []
        intrinsics = []
        extrinsics = []
        plucker = []
        plucker_intrinsics = []
        plucker_extrinsics = []

        # Calculate plucker in advance
        for img_id in chosen_img_ids:
            intr, extr = self.get_intrinsics_extrinsics(obj_id, scene_id, img_id)
            plucker_intrinsics.append(intr)
            plucker_extrinsics.append(torch.tensor(extr, dtype=torch.float32))

        plucker_intrinsics = torch.stack(plucker_intrinsics)
        plucker_extrinsics = torch.stack(plucker_extrinsics)

        # Compute plucker
        c2w = torch.linalg.inv(plucker_extrinsics)
        # c2w = plucker_extrinsics
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
            2.0
            if torch.isclose(
                torch.norm(camera_dists[0]),
                torch.zeros(1),
                atol=1e-5,
            ).any()
            else (2.0 / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= translation_scaling_factor
        c2w[:, :3, 3] *= translation_scaling_factor

        intrinsics_src = plucker_intrinsics[0:1]
        plucker_before_tranforms = get_plucker_coordinates(
            w2c[0], w2c, intrinsics_src.float().clone(), target_size=(518, 518), normalize_plucker=False
        )
        cross_plucker = torch.tensor([0])


        # Reference pose for relative coordinate system (same as Plucker)
        reference_c2w = c2w[0]  # View 0 as reference
        
        for i, img_id in enumerate(chosen_img_ids):
            # print(obj_id, scene_id, img_id)
            # Pass the normalized c2w AND reference for consistent world-space normal computation
            # Normals must be in the SAME coordinate system as Plucker (view-0-relative)
            normalized_c2w = c2w[i]  # Use normalized camera pose, not original annotation
            normalized_w2c = w2c[i]  # Corresponding w2c (camera-to-world inverse)
            data = self.get_single(obj_id, scene_id, img_id, plucker_before_tranforms[i], normalized_c2w, reference_c2w)
            images.append(data["image"])
            depths.append(data["depth"])
            snorms.append(data["snorm"])
            intrinsics.append(data["intrinsics"])
            extrinsics.append(normalized_w2c)  # Use normalized extrinsics, not original
            plucker.append(data["plucker"])

        images = torch.stack(images)
        depths = torch.stack(depths)
        snorms = torch.stack(snorms)
        intrinsics = torch.stack(intrinsics)
        extrinsics = torch.stack(extrinsics)
        plucker = torch.stack(plucker)


        return {
            "images": images,
            "depths": depths,
            "snorms": snorms,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "plucker": plucker,
            "cross_plucker": cross_plucker,
        }

    def get_intrinsics_extrinsics(self, obj_id, scene_id, img_id):
        obj_number = self.objects[obj_id]
        anno = self.data_dict[obj_id][scene_id]["annotations"][img_id]

        # get scene path
        prefix = "downsampled_"
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{prefix}{img_id}.jpg"

        # get image
        image = read_image(image_path)
        image = self.image_transform(image) # Resize and then center crop. The focl should be adjusted to the same value fx and fy.

        #  === construct xyz at full image size and apply all transformations ===
        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)
        # intrnsics for augmented image
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = 0.5 * image_w  # cx = center in width dimension
        intrinsics[1, 2] = 0.5 * image_h  # cy = center in height dimension

        intrinsics = scale_K(intrinsics, image_w, image_h) # Intrinsics should be in normalized image coordinates

        # extract pose and change translation unit from mm to meters
        Rt = build_pose(anno)
        Rt[:3, 3] = Rt[:3, 3]

        return intrinsics, Rt



    def get_single(self, obj_id, scene_id, img_id, plucker, normalized_c2w=None, reference_c2w=None):
        obj_number = self.objects[obj_id]
        anno = self.data_dict[obj_id][scene_id]["annotations"][img_id]

        # get scene path
        prefix = "downsampled_"
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{prefix}{img_id}.jpg"
        depth_path = scene_path / f"depth/{prefix}{img_id}.png"

        # get image
        image = read_image(image_path)
        image = self.image_transform(image)

        # Assert that image size is same as plucker spatial, for example (3,512,512) -> (6,512,512)
        assert image.shape[1:] == plucker.shape[1:], \
            f"Spatial dims mismatch: {image.shape[1:]} vs {plucker.shape[1:]}"

        # get depth -- move from millimeter to meters
        depth = read_depth(str(depth_path)) / 1000
        min_depth = depth[depth > 0].min()
        depth = self.target_transform(depth)

        #  === construct xyz at full image size and apply all transformations ===
        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)

        # intrnsics for augmented image
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = 0.5 * image_w  # cx = center in width dimension
        intrinsics[1, 2] = 0.5 * image_h  # cy = center in height dimension

        # make grid
        grid = get_grid(image_h, image_w)
        uvd_grid = depth * grid
        xyz = intrinsics.inverse() @ uvd_grid.view(3, image_h * image_w)
        xyz_grid = xyz.view(3, image_h, image_w)

        if self.bbox_crop:
            image, depth, xyz_grid, plucker = bbox_crop(image, depth, xyz_grid, plucker)

        bbox_h, bbox_w = image.shape[1:]
        snorm = compute_normal(depth.clone(), aug_fx)
        
        # Transform normals to world space for multi-view consistency (if enabled)
        # CRITICAL: Normals must be in the SAME coordinate system as Plucker embeddings!
        # Plucker uses view-0-relative coordinates, so normals must also be relative to view 0
        if self.world_space_normals:
            if normalized_c2w is not None and reference_c2w is not None:
                # Compute RELATIVE transformation: current view relative to reference (view 0)
                # This matches how Plucker computes: extrinsics_rel = w2c @ inv(w2c_ref)
                #                                                   = w2c @ c2w_ref
                # For normals, we want to transform from camera space to the view-0-relative frame
                
                # Step 1: Transform from current camera space to absolute world space
                R_c2w = normalized_c2w[:3, :3]  # Current camera to world
                
                # Step 2: Transform from absolute world to view-0's camera space (reference frame)
                # This is R_ref_w2c = R_ref_c2w^T (transpose for rotation inverse)
                R_ref_w2c = reference_c2w[:3, :3].T
                
                # Combined: current camera -> world -> view-0 camera frame
                # R_relative = R_ref_w2c @ R_c2w
                R_relative = R_ref_w2c @ R_c2w
                
                # Transform normals using the relative rotation
                snorm_flat = snorm.view(3, -1)
                snorm_relative = R_relative @ snorm_flat
                
                # Re-normalize for numerical stability
                snorm_relative = torch.nn.functional.normalize(snorm_relative, p=2, dim=0, eps=1e-6)
                snorm = snorm_relative.view(3, bbox_h, bbox_w)
                
            elif normalized_c2w is not None:
                # Fallback: absolute world-space (old behavior, not aligned with Plucker)
                R_c2w = normalized_c2w[:3, :3]
                snorm_flat = snorm.view(3, -1)
                snorm_world = R_c2w @ snorm_flat
                snorm_world = torch.nn.functional.normalize(snorm_world, p=2, dim=0, eps=1e-6)
                snorm = snorm_world.view(3, bbox_h, bbox_w)
            else:
                # Double fallback: extract from original annotation (for backward compatibility)
                c2w_temp = camera_matrices_from_annotation(anno)
                R_c2w = c2w_temp[:3, :3]
                snorm_flat = snorm.view(3, -1)
                snorm_world = R_c2w @ snorm_flat
                snorm_world = torch.nn.functional.normalize(snorm_world, p=2, dim=0, eps=1e-6)
                snorm = snorm_world.view(3, bbox_h, bbox_w)

        if self.shared_transform is not None:
            transformed = self.shared_transform(
                image=image.permute(1, 2, 0).numpy(),
                depth=depth.permute(1, 2, 0).numpy(),
                snorm=snorm.permute(1, 2, 0).numpy(),
                xyz_grid=xyz_grid.permute(1, 2, 0).numpy(),
                plucker=plucker.permute(1, 2, 0).numpy(),
            )

            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            xyz_grid = torch.tensor(transformed["xyz_grid"]).float().permute(2, 0, 1)
            plucker = torch.tensor(transformed["plucker"]).float().permute(2, 0, 1)

        # -- use min() to handle center cropping
        final_h, final_w = image.shape[1:]
        final_fx = aug_fx * min(final_h, final_w) / min(bbox_h, bbox_w)
        intrinsics = torch.eye(3)
        intrinsics[:2] = final_fx * intrinsics[:2]

        # remove weird depth artifacts from averaging
        depth[depth < min_depth] = 0

        # extract pose and change translation unit from mm to meters
        Rt = camera_matrices_from_annotation(anno)
        Rt[:3, 3] = Rt[:3, 3] / 1000.0

        if self.relative_depth:
            max_depth = depth.max()
            zero_mask = depth == 0

            # normalize depth between 0 and 1
            depth = (depth - min_depth) / max(0.01, max_depth - min_depth)
            depth = depth * 0.99 + 0.01

            # set zero deoth to zero
            depth[zero_mask] = 0

        # mask = (depth > 0).float()
        # image = (image * mask) + torch.ones_like(image) * (1 - mask)

        return {
            "image": image,
            "depth": depth,
            "class_id": obj_number,
            "intrinsics": intrinsics,
            "snorm": snorm,
            "plucker": plucker,
            "Rt": Rt,
            "anno": anno,
            "xyz_grid": xyz_grid,
        }

    def parse_dataset(self):
        """
        Parses the directory for instances.
        Input: data_dict -- sturcture  <object_id>/<collection>/<instances>

        Output: all dataset instances
        """
        data_dict = {}

        # get all image folders
        all_collections = []
        all_collections += glob.glob(str(self.data_root / "*/multiview_*"))
        all_collections += glob.glob(str(self.data_root / "*/wild_set"))

        # parse image folders
        for collection_path in all_collections:
            object_id, collection_id = collection_path.split("/")[-2:]

            # get image ids
            img_files = os.listdir(os.path.join(collection_path, "images"))
            img_ids = [_file.split(".")[0] for _file in img_files if "jpg" in _file]

            # VERY hacky | remove small_
            img_ids = [_id for _id in img_ids if "_" not in _id]

            # load annotations and convert them to a dictionary
            with open(os.path.join(collection_path, "annotations.json")) as f:
                annotations = json.load(f)
                annotations = {
                    _an["filename"].split(".")[0]: _an for _an in annotations
                }

            # update data_dict
            if object_id not in data_dict:
                data_dict[object_id] = {}

            data_dict[object_id][collection_id] = {
                "views": img_ids,
                "annotations": annotations,
            }

        return data_dict

    def define_instances_split(self, model, collection, subpart):
        # get a list of object names
        if model == "all":
            object_names = list(self.data_dict.keys())
        else:
            object_names = [model]

        assert collection in ["multiview", "wild"]
        assert subpart in ["train", "test", "all"]

        self.instances = []
        self.objects = []

        for obj_id in object_names:
            scenes = list(self.data_dict[obj_id].keys())
            if "wild_set" not in scenes or len(scenes) == 1:
                print(f"Skipping object {obj_id}.")
                continue
            else:
                self.objects.append(obj_id)

            # print(collection, subpart)
            if collection == "wild":
                image_ids = self.data_dict[obj_id]["wild_set"]["views"]
                image_ann = self.data_dict[obj_id]["wild_set"]["annotations"]
                assert len(image_ids) > 1

                # Apply stride=1 to wild_set for proper validation (evaluate every image)
                for start_idx in range(len(image_ids)):
                    # Get n_dim views, wrapping around if necessary
                    window_views = []
                    for i in range(self.n_dim):
                        view_idx = (start_idx + i) % len(image_ids)
                        window_views.append(image_ids[view_idx])
                    self.instances.append((obj_id, "wild_set", window_views))
            else:
                scenes = [_s for _s in scenes if "multiview" in _s]

                # int floors -> at least 1 validation scene
                train_split = int(0.9 * len(scenes))

                if subpart == "train":
                    scenes = scenes[:train_split]
                elif subpart == "test":
                    scenes = scenes[train_split:]
                elif subpart == "all":
                    scenes = scenes
                else:
                    assert collection == "multiview", f"collection was {collection}."

                if len(scenes) == 0:
                    continue

                # Create sliding windows of images within each scene
                # For training (subpart="train" or "all"): stride=n_dim (non-overlapping)
                # For validation (subpart="test"): stride=1 (every image as anchor)
                is_validation = (subpart == "test")
                stride = 1 if is_validation else self.n_dim
                
                for scene in scenes:
                    views = self.data_dict[obj_id][scene]["views"]
                    views.sort()

                    for start_idx in range(0, len(views), stride):
                        # Get n_dim views, wrapping around if necessary (for validation)
                        window_views = []
                        for i in range(self.n_dim):
                            view_idx = (start_idx + i) % len(views)
                            window_views.append(views[view_idx])
                        
                        # For training, only include complete non-wrapped windows
                        # For validation, include all (with wrapping)
                        if is_validation or start_idx + self.n_dim <= len(views):
                            self.instances.append((obj_id, scene, window_views))

        # create object -> class mapping
        self.objects.sort()
        self.objects = {_obj: _id for _id, _obj in enumerate(self.objects)}

    def generate_instance_pairs(self, instances):
        torch.manual_seed(8)
        inst_dict = {}
        for ins in instances:
            obj_id, coll_id, img_id = ins
            if obj_id not in inst_dict:
                inst_dict[obj_id] = {coll_id: [img_id]}
            elif coll_id not in inst_dict[obj_id]:
                inst_dict[obj_id][coll_id] = [img_id]
            else:
                inst_dict[obj_id][coll_id].append(img_id)

        pair_dict = {}
        for obj_id in inst_dict:
            pair_dict[obj_id] = {}
            for col_id in inst_dict[obj_id]:
                pair_dict[obj_id][col_id] = {}
                rots = []
                img_ids = []
                for img_id in inst_dict[obj_id][col_id]:
                    anno = self.data_dict[obj_id][col_id]["annotations"][img_id]
                    Rt = camera_matrices_from_annotation(anno)
                    rots.append(Rt[:3, :3])
                    img_ids.append(img_id)
                rots = torch.stack(rots, dim=0)

                # for each image find a pair between 0 and max_angle degrees
                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    rots_i = rots[i, None].repeat(len(rots), 1, 1)
                    rots_ij = torch.bmm(rots_i, rots.permute(0, 2, 1))
                    rots_tr = rots_ij[:, 0, 0] + rots_ij[:, 1, 1] + rots_ij[:, 2, 2]
                    rel_ang_rad = (0.5 * rots_tr - 0.5).clamp(min=-1, max=1).acos()
                    rel_ang_deg = rel_ang_rad * 180 / np.pi

                    # only sample from deg(0, max_angle)
                    rel_ang_deg[rel_ang_deg > self.max_angle] = 0
                    rel_ang_deg[rel_ang_deg > 0] = 1

                    # sample an element
                    pair_i = torch.multinomial(rel_ang_deg, 1).item()
                    pair_dict[obj_id][col_id][img_id] = img_ids[pair_i]

        return pair_dict
