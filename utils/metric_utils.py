import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



@torch.no_grad()
def export_results(
    input,
    target,
    ss,
    ood_target,
    rendered_input,
    rendered_target,
    rendered_ss,
    rendered_ood_target,
    out_dir: str, 
    compute_metrics: bool = False,
    n_iters: int = None,
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        input: Input data batch
        target: Target data batch
        ss: Self-supervision data batch
        ood_target: OOD target data batch
        rendered_input: Rendered input data batch
        rendered_target: Rendered target data batch
        rendered_ss: Rendered self-supervision data batch
        rendered_ood_target: Rendered OOD target data batch
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
        n_iters: Number of iterations
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for batch_idx in range(input.image.size(0)):
        uid = input.index[batch_idx, 0, -1].item()
        scene_name = input.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        input_indices = input.index[batch_idx, :, 0].cpu().numpy()
        target_indices = target.index[batch_idx, :, 0].cpu().numpy()
        ss_indices = ss.index[batch_idx, :, 0].cpu().numpy()
        ood_target_indices = ood_target.index[batch_idx, :, 0].cpu().numpy()
        
        # Save images
        _save_images(
            input, 
            target, 
            ss,
            ood_target,
            rendered_input, 
            rendered_target, 
            rendered_ss,
            rendered_ood_target,
            batch_idx, 
            sample_dir, 
            n_iters
        )
        
        # Compute and save metrics if requested
        if compute_metrics:
            _save_metrics(
                input.image[batch_idx],
                target.image[batch_idx],
                ss.image[batch_idx],
                ood_target.image[batch_idx],
                rendered_input[batch_idx],
                rendered_target[batch_idx],
                rendered_ss[batch_idx],
                rendered_ood_target[batch_idx],
                input_indices,
                target_indices,
                ss_indices,
                ood_target_indices,
                sample_dir,
                scene_name,
                n_iters
            )
        
        # Save video if available
        # if hasattr(result, "video_rendering"):
        #     _save_video(result.video_rendering[batch_idx], sample_dir)


def visualize_intermediate_results(out_dir, input, target, ss, ood_target, rendered_input, rendered_target, rendered_ss, rendered_ood_target):
    os.makedirs(out_dir, exist_ok=True)

    def vis(gt, rendered, name):
        b, v, _, h, w = rendered.size()
        rendered = rendered.reshape(b * v, -1, h, w)
        gt = gt.reshape(b * v, -1, h, w)
        visualized_image = torch.cat((gt, rendered), dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        Image.fromarray(visualized_image).save(os.path.join(out_dir, f"rendered_{name}.jpg"))
        # with open(os.path.join(out_dir, f"uids_{name}.txt"), "w") as f:
        #     uids = [gt.index[b, 0, -1].item() for b in range(gt.index.size(0))]
        #     uids = "_".join([f"{uid:08}" for uid in uids])
        #     f.write(uids)

    vis(input.image, rendered_input, "input")
    vis(target.image, rendered_target, "target")
    vis(ss.image, rendered_ss, "ss")
    vis(ood_target.image, rendered_ood_target, "ood_target")


def _save_images(input, target, ss, ood_target, rendered_input, rendered_target, rendered_ss, rendered_ood_target, batch_idx, out_dir, n_iters=None):
    """Save visualization images."""
    # Save input image
    # input_img = input.image[batch_idx]
    # input_img = rearrange(input_img, "v c h w -> h (v w) c")
    # input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    # Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))
    prefix = f"{n_iters}iters_" if (n_iters is not None) else ""
    
    # Save GT input vs rendered input side-by-side
    def save_comparison(gt, rendered, name):
        comparison = torch.cat(
            (gt, rendered), 
            dim=2
        ).detach().cpu()
        comparison = rearrange(comparison, "v c h w -> h (v w) c")
        comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        Image.fromarray(comparison).save(os.path.join(out_dir, f"{prefix}{name}.png"))
    
    save_comparison(input.image[batch_idx], rendered_input[batch_idx], "input")
    save_comparison(target.image[batch_idx], rendered_target[batch_idx], "target")
    save_comparison(ss.image[batch_idx], rendered_ss[batch_idx], "ss")
    save_comparison(ood_target.image[batch_idx], rendered_ood_target[batch_idx], "ood_target")


def _save_metrics(input, target, ss, ood_target, rendered_input, rendered_target, rendered_ss, rendered_ood_target, input_indices, target_indices, ss_indices, ood_target_indices, out_dir, scene_name, n_iters=None):
    
    def compute_metrics(gt, rendered):
        gt = gt.to(torch.float32)
        rendered = rendered.to(torch.float32)
        psnr = compute_psnr(gt, rendered)
        lpips = compute_lpips(gt, rendered)
        ssim = compute_ssim(gt, rendered)
        return psnr, lpips, ssim

    input_psnr, input_lpips, input_ssim = compute_metrics(input, rendered_input)
    target_psnr, target_lpips, target_ssim = compute_metrics(target, rendered_target)
    ss_psnr, ss_lpips, ss_ssim = compute_metrics(ss, rendered_ss)
    ood_target_psnr, ood_target_lpips, ood_target_ssim = compute_metrics(ood_target, rendered_ood_target)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "input_psnr": float(input_psnr.mean()),
            "input_lpips": float(input_lpips.mean()),
            "input_ssim": float(input_ssim.mean()),
            "target_psnr": float(target_psnr.mean()),
            "target_lpips": float(target_lpips.mean()),
            "target_ssim": float(target_ssim.mean()),
            "ss_psnr": float(ss_psnr.mean()),
            "ss_lpips": float(ss_lpips.mean()),
            "ss_ssim": float(ss_ssim.mean()),
            "ood_target_psnr": float(ood_target_psnr.mean()),
            "ood_target_lpips": float(ood_target_lpips.mean()),
            "ood_target_ssim": float(ood_target_ssim.mean()),
        },
        "input_view": [],
        "target_view": [],
        "ss_view": [],
        "ood_target_view": []
    }
    for i, view_idx in enumerate(input_indices):
        metrics["input_view"].append({
            "view": int(view_idx), 
            "input_psnr": float(input_psnr[i]), 
            "input_lpips": float(input_lpips[i]), 
            "input_ssim": float(input_ssim[i])
        })
    
    for i, view_idx in enumerate(target_indices):
        metrics["target_view"].append({
            "view": int(view_idx), 
            "target_psnr": float(target_psnr[i]), 
            "target_lpips": float(target_lpips[i]), 
            "target_ssim": float(target_ssim[i])
        })
    
    for i, view_idx in enumerate(ss_indices):
        metrics["ss_view"].append({
            "view": int(view_idx), 
            "ss_psnr": float(ss_psnr[i]), 
            "ss_lpips": float(ss_lpips[i]), 
            "ss_ssim": float(ss_ssim[i])
        })
    
    for i, view_idx in enumerate(ood_target_indices):
        metrics["ood_target_view"].append({
            "view": int(view_idx), 
            "ood_target_psnr": float(ood_target_psnr[i]), 
            "ood_target_lpips": float(ood_target_lpips[i]), 
            "ood_target_ssim": float(ood_target_ssim[i])
        })
    
    # Save metrics to a single JSON file
    prefix = f"{n_iters}iters_" if (n_iters is not None) else ""
    with open(os.path.join(out_dir, f"{prefix}metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder, n_iters=None):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    prefix = f"{n_iters}iters_" if (n_iters is not None) else ""
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, f"{prefix}metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, f"{prefix}summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [sum(values) / len(values) for values in metrics.values()]
        averages_str = [f"{avg:.4f}" for avg in averages]
        f.write(f"average,{','.join(averages_str)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"{prefix}Average: {','.join(averages_str)}")

    # input_psnr, input_lpips, input_ssim, target_psnr, target_lpips, target_ssim, ss_psnr, ss_lpips, ss_ssim, ood_target_psnr, ood_target_lpips, ood_target_ssim
    return averages[0], averages[1], averages[2], averages[3], averages[4], averages[5], averages[6], averages[7], averages[8], averages[9], averages[10], averages[11]