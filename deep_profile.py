# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()



# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)


if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {config.inference_out_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    import time
    import numpy as np
    import torch
    import os
    from PIL import Image
    from einops import rearrange
    import json
    
    step_times = []
    load_times = []
    infer_times = []
    post_times = []
    
    # Detailed component times
    psnr_times = []
    ssim_times = []
    lpips_times = []
    image_save_times = []
    json_save_times = []
    
    # Deep profiling for unmeasured time
    directory_times = []
    data_prep_times = []
    loop_overhead_times = []
    export_call_times = []
    barrier_times = []
    
    print(f"[RANK {ddp_info.local_rank}] Deep profiling batch_size={config.training.batch_size_per_gpu}")
    
    for step, batch in enumerate(dataloader, start=1):
        step_start = time.time()
        
        # Data loading
        load_start = time.time()
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        torch.cuda.synchronize()
        load_end = time.time()
        
        # Model inference
        infer_start = time.time()
        result = model(batch)
        torch.cuda.synchronize()
        infer_end = time.time()
        
        # Post-processing with deep breakdown
        post_start = time.time()
        
        # Video rendering
        if config.inference.get("render_video", False):
            result = model.module.render_video(result, **config.inference.render_video_config)
        
        # Deep export_results breakdown
        if config.inference.get("compute_metrics", False):
            export_start = time.time()
            
            # Directory creation timing
            dir_start = time.time()
            os.makedirs(config.inference_out_dir, exist_ok=True)
            dir_end = time.time()
            directory_times.append(dir_end - dir_start)
            
            # Data preparation timing
            prep_start = time.time()
            input_data, target_data = result.input, result.target
            prep_end = time.time()
            data_prep_times.append(prep_end - prep_start)
            
            # Loop through batch
            loop_start = time.time()
            for batch_idx in range(input_data.image.size(0)):
                batch_loop_start = time.time()
                
                uid = input_data.index[batch_idx, 0, -1].item()
                sample_dir = os.path.join(config.inference_out_dir, f"{uid:06d}")
                
                # Sample directory creation
                sample_dir_start = time.time()
                os.makedirs(sample_dir, exist_ok=True)
                sample_dir_end = time.time()
                
                # Image saving with detailed timing
                img_start = time.time()
                input_img = result.input.image[batch_idx]
                input_img = rearrange(input_img, "v c h w -> h (v w) c")
                input_img_np = input_img.cpu().numpy()  # GPU->CPU transfer
                input_img_np = (input_img_np * 255.0).clip(0.0, 255.0).astype(np.uint8)
                Image.fromarray(input_img_np).save(os.path.join(sample_dir, "input.png"))

                comparison = torch.cat((result.target.image[batch_idx], result.render[batch_idx]), dim=2).detach().cpu()
                comparison_np = rearrange(comparison, "v c h w -> h (v w) c").numpy()
                comparison_np = (comparison_np * 255.0).clip(0.0, 255.0).astype(np.uint8)
                Image.fromarray(comparison_np).save(os.path.join(sample_dir, "gt_vs_pred.png"))
                img_end = time.time()
                image_save_times.append(img_end - img_start)
                
                # Metrics data preparation
                metrics_prep_start = time.time()
                target = target_data.image[batch_idx].to(torch.float32)
                prediction = result.render[batch_idx].to(torch.float32)
                metrics_prep_end = time.time()
                
                # PSNR timing (check device)
                psnr_start = time.time()
                print(f"[RANK {ddp_info.local_rank}] PSNR: target device={target.device}, pred device={prediction.device}")
                from utils.metric_utils import compute_psnr
                psnr_values = compute_psnr(target, prediction)
                torch.cuda.synchronize()
                psnr_end = time.time()
                psnr_times.append(psnr_end - psnr_start)
                
                # SSIM timing (check device)
                ssim_start = time.time()
                print(f"[RANK {ddp_info.local_rank}] SSIM: target device={target.device}, pred device={prediction.device}")
                from utils.metric_utils import compute_ssim
                ssim_values = compute_ssim(target, prediction)
                torch.cuda.synchronize()
                ssim_end = time.time()
                ssim_times.append(ssim_end - ssim_start)
                
                # LPIPS timing (check device)
                lpips_start = time.time()
                print(f"[RANK {ddp_info.local_rank}] LPIPS: target device={target.device}, pred device={prediction.device}")
                from utils.metric_utils import compute_lpips
                lpips_values = compute_lpips(target, prediction)
                torch.cuda.synchronize()
                lpips_end = time.time()
                lpips_times.append(lpips_end - lpips_start)
                
                # JSON creation and saving
                json_start = time.time()
                target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
                scene_name = input_data.scene_name[batch_idx]
                
                metrics = {
                    "summary": {
                        "scene_name": scene_name,
                        "psnr": float(psnr_values.mean()),
                        "lpips": float(lpips_values.mean()),
                        "ssim": float(ssim_values.mean())
                    },
                    "per_view": []
                }
                
                for i, view_idx in enumerate(target_indices):
                    metrics["per_view"].append({
                        "view": int(view_idx),
                        "psnr": float(psnr_values[i]),
                        "lpips": float(lpips_values[i]),
                        "ssim": float(ssim_values[i])
                    })
                
                with open(os.path.join(sample_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                json_end = time.time()
                json_save_times.append(json_end - json_start)
                
                batch_loop_end = time.time()
                loop_overhead_times.append((batch_loop_end - batch_loop_start) - 
                    (img_end - img_start) - (psnr_end - psnr_start) - 
                    (ssim_end - ssim_start) - (lpips_end - lpips_start) - 
                    (json_end - json_start) - (sample_dir_end - sample_dir_start) - 
                    (metrics_prep_end - metrics_prep_start))
                
            loop_end = time.time()
            export_end = time.time()
            export_call_times.append(export_end - export_start)
            
        else:
            from utils.metric_utils import export_results
            export_results(result, config.inference_out_dir, compute_metrics=False)
        
        torch.cuda.synchronize()
        post_end = time.time()
        
        load_times.append(load_end - load_start)
        infer_times.append(infer_end - infer_start)
        post_times.append(post_end - post_start)
        step_times.append(post_end - step_start)
        
        if step % 3 == 0 and ddp_info.is_main_process:
            print(f"[STEP {step}] total={np.mean(step_times[-3:]):.3f}s post={np.mean(post_times[-3:]):.3f}s")
    
    # Barrier timing
    barrier_start = time.time()
    torch.cuda.empty_cache()
    barrier_end = time.time()
    barrier_times.append(barrier_end - barrier_start)
    
    # Deep analysis
    if ddp_info.is_main_process and len(step_times) > 0:
        print("\n" + "="*120)
        print(f"DEEP PROFILING RESULTS (batch_size={config.training.batch_size_per_gpu}, {len(step_times)} steps)")
        print("="*120)
        
        avg_load = np.mean(load_times)
        avg_infer = np.mean(infer_times)
        avg_post = np.mean(post_times)
        avg_total = np.mean(step_times)
        
        print(f"Overall breakdown:")
        print(f"  Data loading:    {avg_load:.4f}s ({avg_load/avg_total*100:.1f}%)")
        print(f"  Model inference: {avg_infer:.4f}s ({avg_infer/avg_total*100:.1f}%)")
        print(f"  Post-processing: {avg_post:.4f}s ({avg_post/avg_total*100:.1f}%)")
        print(f"  Total per step:  {avg_total:.4f}s")
        
        if config.inference.get("compute_metrics", False) and psnr_times:
            print(f"\nMeasured post-processing components:")
            avg_psnr = np.mean(psnr_times)
            avg_ssim = np.mean(ssim_times)
            avg_lpips = np.mean(lpips_times)
            avg_img = np.mean(image_save_times)
            avg_json = np.mean(json_save_times)
            avg_dir = np.mean(directory_times) if directory_times else 0
            avg_prep = np.mean(data_prep_times) if data_prep_times else 0
            avg_loop = np.mean(loop_overhead_times) if loop_overhead_times else 0
            
            print(f"  PSNR computation: {avg_psnr:.4f}s ({avg_psnr/avg_post*100:.1f}%)")
            print(f"  SSIM computation: {avg_ssim:.4f}s ({avg_ssim/avg_post*100:.1f}%)")
            print(f"  LPIPS computation:{avg_lpips:.4f}s ({avg_lpips/avg_post*100:.1f}%)")
            print(f"  Image saving:     {avg_img:.4f}s ({avg_img/avg_post*100:.1f}%)")
            print(f"  JSON saving:      {avg_json:.4f}s ({avg_json/avg_post*100:.1f}%)")
            print(f"  Directory ops:    {avg_dir:.4f}s ({avg_dir/avg_post*100:.1f}%)")
            print(f"  Data preparation: {avg_prep:.4f}s ({avg_prep/avg_post*100:.1f}%)")
            print(f"  Loop overhead:    {avg_loop:.4f}s ({avg_loop/avg_post*100:.1f}%)")
            
            total_measured = avg_psnr + avg_ssim + avg_lpips + avg_img + avg_json + avg_dir + avg_prep + avg_loop
            unmeasured = avg_post - total_measured
            
            print(f"  Unmeasured/other: {unmeasured:.4f}s ({unmeasured/avg_post*100:.1f}%)")
            
            print(f"\nBOTTLENECK RANKING:")
            components = [
                ("LPIPS", avg_lpips),
                ("Image saving", avg_img),
                ("SSIM", avg_ssim),
                ("Loop overhead", avg_loop),
                ("Data prep", avg_prep),
                ("Directory ops", avg_dir),
                ("JSON saving", avg_json),
                ("PSNR", avg_psnr),
                ("Unmeasured", unmeasured)
            ]
            components.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, time_val) in enumerate(components):
                icon = "🔴" if i == 0 else "🟡" if i <= 2 else "🟢"
                print(f"  {i+1}. {icon} {name}: {time_val:.4f}s ({time_val/avg_post*100:.1f}%)")
        
        print("="*120)
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)