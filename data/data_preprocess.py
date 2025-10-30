# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import json
import time
import multiprocessing as mp
from tqdm import tqdm

def preprocess_frames(frames_chosen, image_paths_chosen, image_size=256, patch_size=8, square_crop=True):
    resize_h = image_size
    images = []
    intrinsics = []
    for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
        image = Image.open(cur_image_path)
        original_image_w, original_image_h = image.size
        
        resize_w = int(resize_h / original_image_h * original_image_w)
        resize_w = int(round(resize_w / patch_size) * patch_size)

        image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
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


def worker_init():
    """
    Initialize worker process by limiting threads to prevent resource exhaustion.
    Each worker process should use only 1 thread to avoid creating too many threads overall.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def preprocess_single_scene(args):
    """
    Preprocess a single scene.
    """
    scene_path, output_dir, image_size, patch_size, square_crop, scene_scale_factor = args
    data_json = json.load(open(scene_path, 'r'))
    scene_name = data_json["scene_name"]
    frames = data_json["frames"]
    image_indices = list(range(len(frames)))
    image_paths = [frames[ic]["image_path"] for ic in image_indices]
    scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    input_images, input_intrinsics, input_c2ws = preprocess_frames(frames, image_paths, image_size, patch_size, square_crop)
    # lvsm use per-batch normalzation so we dont do pose normalization here.
    # input_c2ws = preprocess_poses(input_c2ws, scene_scale_factor)

    # all the information needed for the scene: image, c2w, fxfycxcy, scene_name
    # scene_name: str
    # images: [N, 3, H, W]
    # intrinsics: [N, 4]
    # c2ws: [N, 4, 4]
    
    # save images as png files and other information as json files
    os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
    scene_info = {
        "scene_name": scene_name,
        "frames": []
    }

    for i in range(len(input_images)):
        image = input_images[i].permute(1, 2, 0).cpu().numpy()
        image = (image * 255.0).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(output_dir, scene_name, f"{i:05d}.png"))
    
        scene_info["frames"].append({
            "image_path": os.path.join(scene_name, f"{i:05d}.png"),
            "intrinsics": input_intrinsics[i].tolist(),
            "c2ws": input_c2ws[i].tolist()
        })

    with open(os.path.join(scene_dir, "scene_info.json"), "w") as f:
        json.dump(scene_info, f, indent=4)
    
    return True


def preprocess_dataset(
    dataset_path, 
    output_dir, 
    image_size=256, 
    patch_size=8, 
    square_crop=True, 
    scene_scale_factor=1.35,
    num_processes=8,
):
    """
    Preprocess a dataset.
    """
    with open(dataset_path, 'r') as f:
        all_scene_paths = f.read().splitlines()
    all_scene_paths = [path for path in all_scene_paths if path.strip()]
    print(f"Found {len(all_scene_paths)} scenes in the dataset, checking the processed scenes...")

    # check the processed scenes by the number of images and json file
    processed_scenes = []
    for scene_path in all_scene_paths:
        raw_json = json.load(open(scene_path, 'r'))
        scene_name = raw_json["scene_name"]
        scene_dir = os.path.join(output_dir, scene_name)
        if os.path.exists(scene_dir):
            has_json = os.path.exists(os.path.join(scene_dir, "scene_info.json"))
            has_all_images = len(raw_json["frames"]) == len(os.listdir(scene_dir)) - 1
            if has_json and has_all_images:
                processed_scenes.append(scene_path)
    all_scene_paths = [path for path in all_scene_paths if path not in processed_scenes]
    print(f"Found {len(processed_scenes)} processed scenes, {len(all_scene_paths)} scenes to process")
    
    os.makedirs(output_dir, exist_ok=True)

    # process the scenes in parallel
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    args = [(scene_path, output_dir, image_size, patch_size, square_crop, scene_scale_factor) for scene_path in all_scene_paths]

    start_time = time.time()
    with mp.Pool(num_processes, initializer=worker_init) as pool:
        results = list(tqdm(
            pool.imap(preprocess_single_scene, args, chunksize=1),
            total=len(all_scene_paths),
            desc=f"Processing scenes with {num_processes} processes"
        ))
    elapsed_time = time.time() - start_time

    successful = sum(1 for success in results if success)
    failed = [(success, path) for success, path in zip(results, all_scene_paths) if not success]
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {successful}/{len(all_scene_paths)} scenes")
    
    if failed:
        print(f"Failed to process {len(failed)} scenes:")
        for _, path in failed:
            print(f"  - {path}")


if __name__ == "__main__":
    image_size = 256
    patch_size = 8
    square_crop = True
    scene_scale_factor = 1.35
    output_dir = "/home/junchen/projects/aip-fsanja/junchen/LVSM/re10k_preprocessed"
    train_dataset_path = "/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/train/full_list.txt"
    test_dataset_path = "/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt"
    preprocess_dataset(test_dataset_path, output_dir + "/test", image_size, patch_size, square_crop, scene_scale_factor)
    preprocess_dataset(train_dataset_path, output_dir + "/train", image_size, patch_size, square_crop, scene_scale_factor)