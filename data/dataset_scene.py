# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
import zipfile
from io import BytesIO



class Dataset(Dataset):
    def __init__(self, image_size, dataset_path, num_input_views, num_target_views, num_ss_views, num_ood_target_views, min_dist, max_dist, inference=False):
        super().__init__()

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.num_ss_views = num_ss_views
        self.num_ood_target_views = num_ood_target_views
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.inference = inference
        
        # Check if dataset_path is a zip file
        self.is_zip = dataset_path.endswith('.zip') and os.path.isfile(dataset_path)
        self.zip_path = dataset_path if self.is_zip else None

        # list all scenes in the dataset
        if self.is_zip:
            print(f"Loading zip file: {dataset_path}")
            with zipfile.ZipFile(dataset_path, 'r') as zf:
                # Get all scene directories (folders containing scene_info.json)
                all_files = zf.namelist()
                scene_dirs = set()
                for file_path in all_files:
                    if 'scene_info.json' in file_path:
                        # Extract the scene directory path
                        scene_dir = os.path.dirname(file_path)
                        scene_dirs.add(scene_dir)
                all_scene_paths = sorted(list(scene_dirs))
        else:
            scenes = os.listdir(dataset_path)
            all_scene_paths = [os.path.join(dataset_path, scene) for scene in scenes]

        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            view_idx_list = dict()
            view_idx_fp = f"data/evaluation_index_re10k_{num_ss_views}ss_{num_ood_target_views}ood_target_dist{min_dist}to{max_dist}.json"
            assert os.path.exists(view_idx_fp), f"View index file {view_idx_fp} does not exist, please run scripts/gen_index.py to generate it first."
            with open(view_idx_fp, 'r') as f:
                view_idx_list = json.load(f)
                # filter out None values, i.e. scenes that don't have specified input and targetviews
                view_idx_list_filtered = [k for k, v in view_idx_list.items() if v is not None]
                filtered_scene_paths = []
                for scene in all_scene_paths:
                    if self.is_zip:
                        # For zip files, scene path doesn't have dataset_path prefix
                        scene_name = os.path.basename(scene)
                    else:
                        scene_name = scene.split("/")[-1]
                    if scene_name in view_idx_list_filtered:
                        filtered_scene_paths.append(scene)

            print(f"Found {len(view_idx_list_filtered)} scenes in index file, {len(filtered_scene_paths)} scenes exist in the dataset.")
            all_scene_paths = filtered_scene_paths
            
            # prevent memory leaking by converting dict to numpy array
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
            input_idx_list_np, target_idx_list_np, ss_idx_list_np, ood_target_idx_list_np = [], [], [], []
            
            for scene_path in all_scene_paths:
                scene_name = os.path.basename(scene_path)
                assert scene_name in view_idx_list, f"Scene {scene_name} is not in the view idx list."
                input_idx_list_np.append(view_idx_list[scene_name]["input"])
                target_idx_list_np.append(view_idx_list[scene_name]["target"])
                ss_idx_list_np.append(view_idx_list[scene_name]["ss"])
                ood_target_idx_list_np.append(view_idx_list[scene_name]["ood_target"])
            self.input_idx_list_np = np.array(input_idx_list_np).astype(np.int32)
            self.target_idx_list_np = np.array(target_idx_list_np).astype(np.int32)
            self.ss_idx_list_np = np.array(ss_idx_list_np).astype(np.int32)
            self.ood_target_idx_list_np = np.array(ood_target_idx_list_np).astype(np.int32)
        
        print(f"Using {len(all_scene_paths)} scenes")
        # prevent memory leaking by converting string list to numpy array
        self.all_scene_paths = np.array(all_scene_paths).astype(np.bytes_)
        
        # For zip files, extract the base directory name (first component of scene path)
        # This is needed because image_paths in JSON are relative to dataset root,
        # but zip file stores them with base directory prefix
        if self.is_zip and len(all_scene_paths) > 0:
            # Get the first scene path and extract base directory
            first_scene = all_scene_paths[0]
            # scene_path is like "test/b12a86f213f4b4f7", we need "test"
            # Zip files always use forward slashes
            parts = first_scene.split('/')
            if len(parts) > 0:
                self.zip_base_dir = parts[0]
            else:
                self.zip_base_dir = ""
        else:
            self.zip_base_dir = None
        
        # For zip files, we'll open a ZipFile object per worker process
        # This is initialized lazily in __getitem__ to work with multiprocessing
        self._zip_file = None
        self._worker_id = None


    def __len__(self):
        return len(self.all_scene_paths)

    
    def _get_zip_file(self):
        """Get or create a ZipFile object for the current worker process."""
        if not self.is_zip:
            return None
        
        # Check if we're in a different worker process
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        # Create a new ZipFile if we're in a new worker or don't have one yet
        if self._zip_file is None or self._worker_id != worker_id:
            if self._zip_file is not None:
                self._zip_file.close()
            self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
            self._worker_id = worker_id
        
        return self._zip_file


    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws


    def view_selector(self, frames):
        if len(frames) < self.num_input_views + self.num_target_views + self.num_ss_views + self.num_ood_target_views:
            return None
        
        # TODO: remove this hardcode
        min_frame_dist = 25 
        max_frame_dist = 192
        if max_frame_dist <= min_frame_dist:
            return None
        
        # distance between input views
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        # sampled_frames = random.sample(range(start_frame + 1, end_frame), self.num_input_views + self.num_target_views - 2)
        # input views and target views are sampled in the same way as the original lvsm codebase
        input_indices = [start_frame, end_frame]
        target_indices = random.sample(range(start_frame + 1, end_frame), self.num_target_views)

        # distances between ss views and input views
        dist_left = random.randint(self.min_dist, self.max_dist) 
        dist_right = random.randint(self.min_dist, self.max_dist)
        ss_left = max(start_frame - dist_left, 0)
        ss_right = min(end_frame + dist_right, len(frames) - 1)
        
        assert self.num_ss_views == 2, "logic below are for num_ss_views == 2"
        ss_indices = [ss_left, ss_right]
        
        # sample ood target views on both sides
        num_ood_target_views_left = self.num_ood_target_views // 2
        num_ood_target_views_right = self.num_ood_target_views - num_ood_target_views_left
        ood_target_left_range = range(ss_left, start_frame + 1)
        ood_target_right_range = range(end_frame, ss_right + 1)
        if start_frame - ss_left + 1 >= num_ood_target_views_left:
            ood_target_left = random.sample(ood_target_left_range, num_ood_target_views_left)
        else:
            ood_target_left = ood_target_left_range * (num_ood_target_views_left // len(ood_target_left_range)) + random.sample(ood_target_left_range, num_ood_target_views_left % len(ood_target_left_range))
        if ss_right - end_frame + 1 >= num_ood_target_views_right:
            ood_target_right = random.sample(ood_target_right_range, num_ood_target_views_right)
        else:
            ood_target_right = ood_target_right_range * (num_ood_target_views_right // len(ood_target_right_range)) + random.sample(ood_target_right_range, num_ood_target_views_right % len(ood_target_right_range))
        ood_target_indices = sorted(ood_target_left + ood_target_right)

        image_indices = input_indices + target_indices + ss_indices + ood_target_indices
        return image_indices


    def __getitem__(self, idx):
        # try:
        scene_path = str(self.all_scene_paths[idx], encoding="utf-8").strip()
        json_file_path = os.path.join(scene_path, "scene_info.json")
        
        try:
            if self.is_zip:
                zf = self._get_zip_file()
                with zf.open(json_file_path) as f:
                    data_json = json.load(f)
            else:
                data_json = json.load(open(json_file_path, 'r'))
        except Exception as e:
            # zip file IO error is triggered, read the next scene
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference:
            input_indices = list(self.input_idx_list_np[idx])
            target_indices = list(self.target_idx_list_np[idx])
            ss_indices = list(self.ss_idx_list_np[idx])
            ood_target_indices = list(self.ood_target_idx_list_np[idx])
            assert self.num_input_views == len(input_indices), f"We have {len(input_indices)} input views, but we want to select {self.num_input_views} input views."
            assert self.num_target_views == len(target_indices), f"We have {len(target_indices)} target views, but we want to select {self.num_target_views} target views."
            assert self.num_ss_views == len(ss_indices), f"We have {len(ss_indices)} ss views, but we want to select {self.num_ss_views} ss views."
            assert self.num_ood_target_views == len(ood_target_indices), f"We have {len(ood_target_indices)} ood target views, but we want to select {self.num_ood_target_views} ood target views."
            image_indices = input_indices + target_indices + ss_indices + ood_target_indices
        else:
            # sample input and target views
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        frames_chosen = [frames[ic] for ic in image_indices]
        image_paths = [frame["image_path"] for frame in frames_chosen]
        c2ws = torch.tensor([frame["c2ws"] for frame in frames_chosen]).float()
        intrinsics = torch.tensor([frame["intrinsics"] for frame in frames_chosen]).float()

        # per-batch normalization
        c2ws = self.preprocess_poses(c2ws, scene_scale_factor=1.35)

        images = []
        if self.is_zip:
            zf = self._get_zip_file()
            for image_path in image_paths:
                # image_path from JSON is relative to dataset root (e.g., "b12a86f213f4b4f7/00000.png")
                # but zip file stores it with base directory prefix (e.g., "test/b12a86f213f4b4f7/00000.png")
                # Zip files always use forward slashes
                if self.zip_base_dir:
                    zip_image_path = f"{self.zip_base_dir}/{image_path}"
                else:
                    zip_image_path = image_path
                with zf.open(zip_image_path) as f:
                    # Open directly from the zip file handle to avoid duplicating bytes in memory
                    image = Image.open(f).convert('RGB')
                    assert image.size == (self.image_size, self.image_size), f"Image {image_path} is not {self.image_size}x{self.image_size}"
                    image = (np.array(image) / 255.0).astype(np.float32)
                    image = torch.from_numpy(image).permute(2, 0, 1)
                    images.append(image)
        else:
            for image_path in image_paths:
                abs_image_path = os.path.join(self.dataset_path, image_path)
                image = Image.open(abs_image_path)
                assert image.size == (self.image_size, self.image_size), f"Image {image_path} is not {self.image_size}x{self.image_size}"
                image = np.array(image) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                images.append(image)
        images = torch.stack(images, dim=0)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": images,
            "c2w": c2ws,
            "fxfycxcy": intrinsics,
            "index": indices,
            "scene_name": scene_name
        }


if __name__ == "__main__":
    # dry run the dataset
    dataset = Dataset(
        image_size=256,
        dataset_path="/home/junchen/projects/aip-fsanja/junchen/LVSM/re10k_preprocessed/test",
        num_input_views=2,
        num_target_views=2,
        num_ss_views=2,
        num_ood_target_views=2,
        min_dist=25,
        max_dist=100,
        inference=False
    )
    batch = dataset[0]
    print(batch["image"].shape)
    print(batch["c2w"].shape)
    print(batch["fxfycxcy"].shape)
    print(batch["index"].shape)
    print(batch["scene_name"])