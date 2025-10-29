"""
Generate self-supervision, input and target index files based on 'evaluation_index_re10k.json'
"""

import json
import os
import sys
from tqdm import tqdm
import random


def gen_indices(dataset_path, index_file_path, num_ss_views, num_ood_target_views, min_frame_dist, max_frame_dist):
    assert num_ss_views >= 2, "We need at least 1 self-supervision view on each side"
    assert num_ood_target_views >= 2, "We need at least 1 ood target view on each side"
    assert os.path.exists(dataset_path), "Dataset path does not exist"
    assert os.path.exists(index_file_path), "Index file path does not exist"

    with open(dataset_path, 'r') as f:
        all_scene_paths = f.read().splitlines()
    all_scene_paths = [path for path in all_scene_paths if path.strip()]

    with open(index_file_path, 'r') as f:
        view_idx_map = json.load(f)
        view_idx_list_filtered = [k for k, v in view_idx_map.items() if v is not None]
        filtered_scene_paths = []
        for scene in all_scene_paths:
            file_name = scene.split("/")[-1]
            scene_name = file_name.split(".")[0]
            if scene_name in view_idx_list_filtered:
                filtered_scene_paths.append(scene)

        all_scene_paths = filtered_scene_paths

    output_indices = dict()
    for scene_path in tqdm(all_scene_paths):
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]
        
        assert scene_name in view_idx_list_filtered, f"Scene {scene_name} is not in the view idx list."
        indices = view_idx_map[scene_name]
        context, target = indices["context"], indices["target"]
    
        # sample one ss view on the left and on the right
        dist_left = random.randint(min_frame_dist, max_frame_dist)
        dist_right = random.randint(min_frame_dist, max_frame_dist)

        ss_left = max(context[0] - dist_left, 0)
        ss_right = min(context[1] + dist_right, len(frames) - 1)

        assert num_ss_views == 2, "logic below are for num_ss_views == 2"
        ss_indices = [ss_left, ss_right]

        # sample ood_target views on both sides
        num_ood_target_views_left = num_ood_target_views // 2
        num_ood_target_views_right = num_ood_target_views - num_ood_target_views_left
        ood_target_left_range = range(ss_left, context[0] + 1)
        ood_target_right_range = range(context[1], ss_right + 1)
        if context[0] - ss_left + 1 >= num_ood_target_views_left:
            ood_target_left = random.sample(ood_target_left_range, num_ood_target_views_left)
        else:
            # repeat the range multiple times to reach the desired number of ood_target views
            ood_target_left = ood_target_left_range * (num_ood_target_views_left // len(ood_target_left_range)) + random.sample(ood_target_left_range, num_ood_target_views_left % len(ood_target_left_range))
        if ss_right - context[1] + 1 >= num_ood_target_views_right:
            ood_target_right = random.sample(ood_target_right_range, num_ood_target_views_right)
        else:
            # repeat the range multiple times to reach the desired number of ood_target views
            ood_target_right = ood_target_right_range * (num_ood_target_views_right // len(ood_target_right_range)) + random.sample(ood_target_right_range, num_ood_target_views_right % len(ood_target_right_range))
        
        ood_target = sorted(ood_target_left + ood_target_right)
        
        output_indices[scene_name] = dict()
        output_indices[scene_name]["input"] = context
        output_indices[scene_name]["target"] = target
        output_indices[scene_name]["ss"] = ss_indices
        output_indices[scene_name]["ood_target"] = ood_target
    
    output_fp = f"{index_file_path.split('.')[0]}_{num_ss_views}ss_{num_ood_target_views}ood_target_dist{min_frame_dist}to{max_frame_dist}.json"
    with open(output_fp, 'w') as f:
        json.dump(output_indices, f, indent=4)
    
    print(f"Generated {len(output_indices)} indices and saved to {output_fp}")


if __name__ == "__main__":
    # Sample ss views and ood_target views for RE10K dataset. 
    # The encoder input views and target views are kept the same as in the index file.

    dataset_path = "/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt"
    index_file_path = "data/evaluation_index_re10k.json"
    num_ss_views = 2
    num_ood_target_views = 2
    min_dist = 25
    max_dist = 100
    gen_indices(dataset_path, index_file_path, num_ss_views, num_ood_target_views, min_dist, max_dist)