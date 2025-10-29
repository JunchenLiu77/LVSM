# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F



class Dataset(Dataset):
    def __init__(self, config, dataset_path, num_input_views, num_target_views, num_ss_views, num_ood_target_views, min_dist, max_dist, inference=False):
        super().__init__()
        self.config = config

        try:
            with open(dataset_path, 'r') as f:
                all_scene_paths = f.read().splitlines()
            all_scene_paths = [path for path in all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{dataset_path}'")
            raise e
        

        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.num_ss_views = num_ss_views
        self.num_ood_target_views = num_ood_target_views
        self.min_dist = min_dist
        self.max_dist = max_dist

        self.inference = inference
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            view_idx_list = dict()
            # assert self.config.inference["view_idx_file_path"] is not None and os.path.exists(self.config.inference["view_idx_file_path"]), "view_idx_file_path must be provided for inference"
            # with open(self.config.inference["view_idx_file_path"], 'r') as f:
            view_idx_fp = f"data/evaluation_index_re10k_{num_ss_views}ss_{num_ood_target_views}ood_target_dist{min_dist}to{max_dist}.json"
            assert os.path.exists(view_idx_fp), f"View index file {view_idx_fp} does not exist, please run scripts/gen_index.py to generate it first."
            with open(view_idx_fp, 'r') as f:
                view_idx_list = json.load(f)
                # filter out None values, i.e. scenes that don't have specified input and targetviews
                view_idx_list_filtered = [k for k, v in view_idx_list.items() if v is not None]
                filtered_scene_paths = []
                for scene in all_scene_paths:
                    file_name = scene.split("/")[-1]
                    scene_name = file_name.split(".")[0]
                    if scene_name in view_idx_list_filtered:
                        filtered_scene_paths.append(scene)

                all_scene_paths = filtered_scene_paths
            
            # prevent memory leaking by converting dict to numpy array
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
            input_idx_list_np, target_idx_list_np, ss_idx_list_np, ood_target_idx_list_np = [], [], [], []
            for scene_path in all_scene_paths:
                data_json = json.load(open(scene_path, 'r'))
                scene_name = data_json["scene_name"]
                assert scene_name in view_idx_list, f"Scene {scene_name} is not in the view idx list."
                input_idx_list_np.append(view_idx_list[scene_name]["input"])
                target_idx_list_np.append(view_idx_list[scene_name]["target"])
                ss_idx_list_np.append(view_idx_list[scene_name]["ss"])
                ood_target_idx_list_np.append(view_idx_list[scene_name]["ood_target"])
            self.input_idx_list_np = np.array(input_idx_list_np).astype(np.int32)
            self.target_idx_list_np = np.array(target_idx_list_np).astype(np.int32)
            self.ss_idx_list_np = np.array(ss_idx_list_np).astype(np.int32)
            self.ood_target_idx_list_np = np.array(ood_target_idx_list_np).astype(np.int32)
            print(f"Found {len(input_idx_list_np)} scenes in index file, {len(all_scene_paths)} scenes exist in the dataset.")
        
        # prevent memory leaking by converting string list to numpy array
        self.all_scene_paths = np.array(all_scene_paths).astype(np.bytes_)


    def __len__(self):
        return len(self.all_scene_paths)


    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

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
        
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
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
        data_json = json.load(open(scene_path, 'r'))
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
        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        images, intrinsics, c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
    
        # except:
        #     traceback.print_exc()
        #     print(f"error loading")
        #     print(image_indices)
        #     print(image_paths_chosen)
        #     return self.__getitem__(random.randint(0, len(self) - 1))


        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        c2ws = self.preprocess_poses(c2ws, scene_scale_factor)

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

