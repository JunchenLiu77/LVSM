# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import time
import wandb
import torch
import random
import json
import csv
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results, summarize_evaluation, export_results
from utils.training_utils import find_checkpoints
from PIL import Image

# Mute noisy warnings/logs from torch.compile/inductor
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
try:
    import torch._logging as torch_logging
    # Reduce log verbosity to only errors
    torch_logging.set_logs(dynamo='error', inductor='error', aot='error', fx='error')
except Exception:
    pass

from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0


# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Check if there are some checkpoints in the checkpoint_dir, if so, we will update the seed
seed = config.training.seed
checkpoint_dir = config.training.checkpoint_dir
all_ckpt_paths = find_checkpoints(checkpoint_dir)
if len(all_ckpt_paths) > 0:
    ckpt_path = all_ckpt_paths[-1]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    train_step = checkpoint["fwdbwd_pass_step"]
    seed = seed + train_step
    print(f"use seed {seed}")

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=seed)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    # Resume W&B if checkpoints exist to continue previous curves
    init_wandb_and_backup(config, resume=(len(all_ckpt_paths) > 0))
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

is_ttt = "ttt" in config.model.class_name
# g3r is not supported for gradient accumulation
if is_ttt and config.model.ttt.supervise_mode == "g3r":
    assert config.training.grad_accum_steps == 1, "Gradient accumulation is not supported for G3R supervision"

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]

# training set
train_set = Dataset(
    image_size=config.model.image_tokenizer.image_size,
    dataset_path="re10k_preprocessed/train.zip",
    num_input_views=config.training.num_input_views, 
    num_target_views=config.training.num_target_views, 
    num_ss_views=config.training.num_ss_views,
    num_ood_target_views=config.training.num_ood_target_views,
    min_dist=25,
    max_dist=100,
    inference=False
)
train_sampler = DistributedSampler(train_set, shuffle=True, seed=seed, drop_last=True)
train_loader = DataLoader(
    train_set,
    batch_size=config.training.batch_size_per_gpu,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=train_sampler,
)
train_loader_iter = iter(train_loader)

if config.training.test_every > 0:
    # test set, use sampler to keep align with LVSM official testset sampling
    test_set = Dataset(
        image_size=config.model.image_tokenizer.image_size,
        dataset_path="re10k_preprocessed/test.zip",
        num_input_views=2,
        num_target_views=3,
        num_ss_views=config.training.num_ss_views,
        num_ood_target_views=config.training.num_ood_target_views,
        min_dist=25,
        max_dist=100,
        inference=True
    )
    test_sampler = DistributedSampler(test_set)
    test_loader = DataLoader(
        test_set,
        batch_size=config.training.test_batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=config.training.prefetch_factor,
        sampler=test_sampler,
    )
    test_sampler.set_epoch(0)

    if is_ttt:
        iters = config.training.test_layers
        assert len(iters) > 0, "At least one test setting should be specified"


total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = config.training.batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(train_set))


module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)


optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
    is_ttt=is_ttt,
    learning_rate_ttt=config.training.lr_ttt if is_ttt else None,
    freeze_encoder=config.training.get("freeze_encoder", False),
    freeze_decoder=config.training.get("freeze_decoder", False),
    freeze_tokenizer=config.training.get("freeze_tokenizer", False),
    freeze_latent=config.training.get("freeze_latent", False),
    freeze_ttt=config.training.get("freeze_ttt", False),
)
optim_param_list = list(optimized_param_dict.values())


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

# change logic here:
# - if there are some checkpoints in the checkpoint_dir, we always resume from the latest checkpoint
# we don't reset training state and use the new seed for the next training
# - if there are no checkpoints in the checkpoint_dir, we check if the resume_ckpt is provided
# if so, we resume from the resume_ckpt and reset training state
# if not, we start from scratch

cur_train_step = 0
cur_param_update_step = 0
if all_ckpt_paths or (config.training.get("resume_ckpt", "") != ""):
    optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
        config.training.checkpoint_dir,
        config.training.resume_ckpt,
        model,
        optimizer,
        lr_scheduler,
    )
cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(train_set))
train_sampler.set_epoch(cur_epoch)

# Apply torch.compile if enabled

# Configure torch.compile with DDP compatibility
if config.training.get("use_torch_compile", False):
    import torch._dynamo
    # Disable DDP optimizer to prevent higher-order op conflicts
    torch._dynamo.config.optimize_ddp = False

    # Additional torch compile configurations for better compatibility
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True  # Fallback to eager mode if compilation fails
if config.training.get("use_torch_compile", False):
    # Disable compilation for ttt_update due to complex control flow
    import torch._dynamo
    try:
        target = model.module if hasattr(model, 'module') else model
        if hasattr(target, 'ttt_update'):
            torch._dynamo.disable(target.ttt_update)
            print_rank0("Disabled compilation for ttt_update function (complex control flow)")
    except Exception as e:
        print_rank0(f"Warning: could not disable compilation for ttt_update: {e}")
    model = torch.compile(model)
    print_rank0("Model compilation completed.")

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()

# Record wall-clock start time for elapsed-time logging
start_wall_time = time.time()

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(train_set))

    # test on multiple nodes - run before optimizer update to test at initial state
    if config.training.test_every > 0 and (cur_train_step == 0 or cur_train_step % config.training.test_every == 0):
        print_rank0(f"Running inference at step {cur_train_step} (before optimizer update)")
        out_dir = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}_inference")
        os.makedirs(out_dir, exist_ok=True)
        
        # instantiate a new iterator every time we test
        test_loader_iter = iter(test_loader)
        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            # accumulate metrics across all test batches this inference step
            metrics = {}
            for (batch_idx, batch) in enumerate(test_loader_iter):
                if config.inference.get("first_n_batches", None) is not None and batch_idx >= config.inference.get("first_n_batches", None):
                    break
                print(f"[Rank {ddp_info.local_rank}] Running inference on the {batch_idx}th batch")
                batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
                if is_ttt:
                    n_input, n_ss, n_ood_target = config.training.num_input_views, config.training.num_ss_views, config.training.num_ood_target_views
                    for n_views in [n_ss, n_ss + n_input, n_ss + n_input + n_ood_target]:
                        input_views_ss = ood_target_views_ss = False
                        if n_views >= n_ss + n_input:
                            # use input views to calculate ss loss
                            input_views_ss = True
                        if n_views >= n_ss + n_input + n_ood_target:
                            # use ood_target views to calculate ss loss
                            ood_target_views_ss = True
                        for n_iters in iters:
                            real_n_iters = n_iters
                            if config.model.ttt.progressive:
                                real_n_iters = int(1 + (n_iters - 1) * min(1.0, max(0, (cur_train_step - config.model.ttt.warmup_steps) / config.model.ttt.warmup_steps)))
                            if config.model.ttt.supervise_mode != "g3r":
                                raise NotImplementedError("TTT without G3R supervision is not supported yet")
                            else:
                                input = None
                                target = None
                                ss = None
                                ood_target = None
                                s = None
                                ss_pose_tokens = None
                                target_pose_tokens = None
                                ood_target_pose_tokens = None
                                ttt_metrics = {"layers": []}
                                ttt_metrics["n_iters"] = real_n_iters

                                for idx in range(real_n_iters):
                                    is_last = (idx == real_n_iters - 1)
                                    layer_idx = 0
                                    iter_idx = idx % config.model.ttt.n_iters_per_layer
                                    t = idx / real_n_iters

                                    # in g3r, input loss metrics and target loss metrics are calculated on the updated state s.
                                    input, target, ss, ood_target, input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss, s, ss_pose_tokens, target_pose_tokens, ood_target_pose_tokens, layer_metrics = model(
                                        batch,
                                        num_input_views=config.training.num_input_views,
                                        num_target_views=3,
                                        num_ss_views=config.training.num_ss_views,
                                        num_ood_target_views=config.training.num_ood_target_views,
                                        is_g3r=True,
                                        has_target_image=True,
                                        training=False,
                                        layer_idx=layer_idx,
                                        iter_idx=iter_idx,
                                        t=t,
                                        input=input,
                                        target=target,
                                        ss=ss,
                                        ood_target=ood_target,
                                        s=s,
                                        ss_pose_tokens=ss_pose_tokens,
                                        target_pose_tokens=target_pose_tokens,
                                        ood_target_pose_tokens=ood_target_pose_tokens,
                                        is_last=is_last,
                                        input_views_ss=input_views_ss,
                                        ood_target_views_ss=ood_target_views_ss,
                                    )
                            # export results with the iterations upper bound, merge per-batch
                            out_dir_nviews = os.path.join(out_dir, f"nviews_{n_views}")
                            os.makedirs(out_dir_nviews, exist_ok=True)
                            per_scene_metrics = export_results(
                                input, target, ss, ood_target,
                                rendered_input, rendered_target, rendered_ss, rendered_ood_target,
                                out_dir_nviews,
                                compute_metrics=config.inference.get("compute_metrics"),
                                n_iters=real_n_iters,
                            )
                            # nest metrics by n_views -> real_n_iters
                            if n_views not in metrics:
                                metrics[n_views] = {}
                            if real_n_iters not in metrics[n_views]:
                                metrics[n_views][real_n_iters] = {}
                            metrics[n_views][real_n_iters].update(per_scene_metrics)
                            # delete per_scene eval metrics and free GPU memory for this test batch
                            del input, target, ss, ood_target
                            del rendered_input, rendered_target, rendered_ss, rendered_ood_target
                            del loss
                            del s, ss_pose_tokens, target_pose_tokens, ood_target_pose_tokens
                            del input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, layer_metrics, ttt_metrics, per_scene_metrics
                            torch.cuda.empty_cache()
                else:
                    input, target, ss, ood_target, input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss = model(
                        batch,
                        num_input_views=config.training.num_input_views,
                        num_target_views=3,
                        num_ss_views=config.training.num_ss_views,
                        num_ood_target_views=config.training.num_ood_target_views,
                        has_target_image=True,
                        training=False,
                    )
                    per_scene_metrics = export_results(
                        input, target, ss, ood_target,
                        rendered_input, rendered_target, rendered_ss, rendered_ood_target,
                        out_dir,
                        compute_metrics=config.inference.get("compute_metrics"),
                        n_iters=0,
                    )
                    # ugly but necessary, since the metrics is a nested dict
                    if 0 not in metrics:
                        metrics[0] = {}
                    metrics[0].update(per_scene_metrics)
                    
                    # free GPU memory for this test batch
                    del input, target, ss, ood_target
                    del rendered_input, rendered_target, rendered_ss, rendered_ood_target
                    del loss
                    del input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, per_scene_metrics
                    del batch
                    torch.cuda.empty_cache()
            dist.barrier()
            if config.inference.get("compute_metrics", False):
                if is_ttt:
                    # gather metrics from all ranks for nested structure {n_views: {n_iters: {uid: metrics}}}
                    def gather_metrics(local_metrics):
                        """
                        Gather nested metrics dicts from all ranks and merge by n_views -> n_iters -> uid.
                        Structure: {n_views: {n_iters: {uid: metrics_dict}}}
                        """
                        gathered = [None for _ in range(ddp_info.world_size)]
                        dist.all_gather_object(gathered, local_metrics)
                        merged = {}
                        for part in gathered:
                            if not part:
                                continue
                            for k_nviews, by_iters in part.items():
                                if k_nviews not in merged:
                                    merged[k_nviews] = {}
                                for k_iters, by_uid in by_iters.items():
                                    if k_iters not in merged[k_nviews]:
                                        merged[k_nviews][k_iters] = {}
                                    merged[k_nviews][k_iters].update(by_uid)
                        return merged

                    metrics = gather_metrics(metrics)

                    # Save combined JSON and CSV averages, then log to wandb for each n_views and iter setting
                    n_input, n_ss, n_ood_target = config.training.num_input_views, config.training.num_ss_views, config.training.num_ood_target_views
                    n_views_list = [n_ss, n_ss + n_input, n_ss + n_input + n_ood_target]
                    for n_views in n_views_list:
                        for n_iters in iters:
                            real_n_iters = n_iters
                            if config.model.ttt.progressive:
                                real_n_iters = min(n_iters, int(1 + (n_iters - 1) * min(1.0, max(0, (cur_train_step - config.model.ttt.warmup_steps) / config.model.ttt.warmup_steps))))

                            test_name = f"test_n{n_views}_{real_n_iters}iters"
                            out_dir_nviews = os.path.join(out_dir, f"nviews_{n_views}")
                            if ddp_info.is_main_process:
                                # Persist merged per-scene metrics JSON (sorted by uid)
                                all_scenes = metrics.get(n_views, {}).get(real_n_iters, {})
                                json_path = os.path.join(out_dir_nviews, f"{real_n_iters}iters_metrics.json")
                                try:
                                    os.makedirs(out_dir_nviews, exist_ok=True)
                                    sorted_items = sorted(all_scenes.items(), key=lambda kv: int(kv[0]) if not isinstance(kv[0], int) else kv[0])
                                    ordered = {f"{int(uid):06d}": data for uid, data in sorted_items}
                                    with open(json_path, "w") as f:
                                        json.dump(ordered, f, indent=2)
                                except Exception:
                                    pass

                                # Compute averages
                                metric_keys = [
                                    "input_psnr", "input_lpips", "input_ssim",
                                    "target_psnr", "target_lpips", "target_ssim",
                                    "ss_psnr", "ss_lpips", "ss_ssim",
                                    "ood_target_psnr", "ood_target_lpips", "ood_target_ssim",
                                ]
                                summaries = [scene_metrics["summary"] for scene_metrics in all_scenes.values()]
                                averages = {k: 0.0 for k in metric_keys}
                                if len(summaries) > 0:
                                    for k in metric_keys:
                                        averages[k] = sum(s[k] for s in summaries) / len(summaries)
                                else:
                                    # keep zeros if no scenes gathered
                                    pass

                                # Write CSV with per-scene summaries and averages (sorted by uid, .4f precision)
                                csv_path = os.path.join(out_dir_nviews, f"{real_n_iters}iters_summary.csv")
                                try:
                                    with open(csv_path, "w", newline="") as f:
                                        writer = csv.writer(f)
                                        # header
                                        writer.writerow(["Index"] + metric_keys)
                                        # rows per scene
                                        for uid, scene in sorted(all_scenes.items(), key=lambda kv: int(kv[0]) if not isinstance(kv[0], int) else kv[0]):
                                            summary = scene.get("summary", {})
                                            row = [f"{int(uid):06d}"] + [f"{summary.get(k, 0.0):.4f}" for k in metric_keys]
                                            writer.writerow(row)
                                        # blank line and averages
                                        writer.writerow([])
                                        avg_row = ["average"] + [f"{averages[k]:.4f}" for k in metric_keys]
                                        writer.writerow(avg_row)
                                except Exception:
                                    pass

                                # Map averages to names expected in wandb logging below
                                input_psnr = averages["input_psnr"]
                                input_lpips = averages["input_lpips"]
                                input_ssim = averages["input_ssim"]
                                target_psnr = averages["target_psnr"]
                                target_lpips = averages["target_lpips"]
                                target_ssim = averages["target_ssim"]
                                ss_psnr = averages["ss_psnr"]
                                ss_lpips = averages["ss_lpips"]
                                ss_ssim = averages["ss_ssim"]
                                ood_target_psnr = averages["ood_target_psnr"]
                                ood_target_lpips = averages["ood_target_lpips"]
                                ood_target_ssim = averages["ood_target_ssim"]
                                wandb.log({
                                    f"{test_name}/input_psnr": input_psnr,
                                    f"{test_name}/input_lpips": input_lpips,
                                    f"{test_name}/input_ssim": input_ssim,
                                    f"{test_name}/target_psnr": target_psnr,
                                    f"{test_name}/target_lpips": target_lpips,
                                    f"{test_name}/target_ssim": target_ssim,
                                    f"{test_name}/ss_psnr": ss_psnr,
                                    f"{test_name}/ss_lpips": ss_lpips,
                                    f"{test_name}/ss_ssim": ss_ssim,
                                    f"{test_name}/ood_target_psnr": ood_target_psnr,
                                    f"{test_name}/ood_target_lpips": ood_target_lpips,
                                    f"{test_name}/ood_target_ssim": ood_target_ssim,
                                }, step=cur_train_step)

                            # log scene images to wandb, only uid 162 and 183 scenes are logged
                            if ddp_info.is_main_process:
                                for uid in [162, 183]:
                                    sample_dir = os.path.join(out_dir_nviews, f"{uid:06d}")
                                    if not os.path.exists(sample_dir):
                                        continue
                                    prefix = f"{real_n_iters}iters_"
                                    input_img = Image.open(os.path.join(sample_dir, f"{prefix}input.png"))
                                    target_img = Image.open(os.path.join(sample_dir, f"{prefix}target.png"))
                                    ss_img = Image.open(os.path.join(sample_dir, f"{prefix}ss.png"))
                                    ood_target_img = Image.open(os.path.join(sample_dir, f"{prefix}ood_target.png"))
                                    wandb.log({
                                        f"{test_name}/uid_{uid}/input": wandb.Image(input_img),
                                        f"{test_name}/uid_{uid}/target": wandb.Image(target_img),
                                        f"{test_name}/uid_{uid}/ss": wandb.Image(ss_img),
                                        f"{test_name}/uid_{uid}/ood_target": wandb.Image(ood_target_img),
                                    }, step=cur_train_step)

                else:
                    # Non-TTT: gather metrics across ranks, save JSON/CSV, and log to wandb
                    def gather_metrics(local_metrics):
                        """
                        Gather nested metrics dicts from all ranks and merge by n_iters -> uid.
                        Structure: {n_iters: {uid: metrics_dict}}
                        """
                        gathered = [None for _ in range(ddp_info.world_size)]
                        dist.all_gather_object(gathered, local_metrics)
                        merged = {}
                        for part in gathered:
                            if not part:
                                continue
                            for k_iters, by_uid in part.items():
                                if k_iters not in merged:
                                    merged[k_iters] = {}
                                merged[k_iters].update(by_uid)
                        return merged

                    real_n_iters = 0
                    metrics = gather_metrics(metrics)

                    test_name = f"test_{real_n_iters}iters"
                    if ddp_info.is_main_process:
                        # Persist merged per-scene metrics JSON (sorted by uid)
                        all_scenes = metrics.get(real_n_iters, {})
                        json_path = os.path.join(out_dir, f"{real_n_iters}iters_metrics.json")
                        try:
                            sorted_items = sorted(all_scenes.items(), key=lambda kv: int(kv[0]) if not isinstance(kv[0], int) else kv[0])
                            ordered = {f"{int(uid):06d}": data for uid, data in sorted_items}
                            with open(json_path, "w") as f:
                                json.dump(ordered, f, indent=2)
                        except Exception:
                            pass

                        # Compute averages
                        metric_keys = [
                            "input_psnr", "input_lpips", "input_ssim",
                            "target_psnr", "target_lpips", "target_ssim",
                            "ss_psnr", "ss_lpips", "ss_ssim",
                            "ood_target_psnr", "ood_target_lpips", "ood_target_ssim",
                        ]
                        summaries = [scene_metrics["summary"] for scene_metrics in all_scenes.values()]
                        averages = {k: 0.0 for k in metric_keys}
                        if len(summaries) > 0:
                            for k in metric_keys:
                                averages[k] = sum(s[k] for s in summaries) / len(summaries)
                        else:
                            pass

                        # Write CSV with per-scene summaries and averages (sorted by uid, .4f precision)
                        csv_path = os.path.join(out_dir, f"{real_n_iters}iters_summary.csv")
                        try:
                            with open(csv_path, "w", newline="") as f:
                                writer = csv.writer(f)
                                # header
                                writer.writerow(["Index"] + metric_keys)
                                # rows per scene
                                for uid, scene in sorted(all_scenes.items(), key=lambda kv: int(kv[0]) if not isinstance(kv[0], int) else kv[0]):
                                    summary = scene.get("summary", {})
                                    row = [f"{int(uid):06d}"] + [f"{summary.get(k, 0.0):.4f}" for k in metric_keys]
                                    writer.writerow(row)
                                # blank line and averages
                                writer.writerow([])
                                avg_row = ["average"] + [f"{averages[k]:.4f}" for k in metric_keys]
                                writer.writerow(avg_row)
                        except Exception:
                            pass

                        # Map averages to names expected in wandb logging below
                        input_psnr = averages["input_psnr"]
                        input_lpips = averages["input_lpips"]
                        input_ssim = averages["input_ssim"]
                        target_psnr = averages["target_psnr"]
                        target_lpips = averages["target_lpips"]
                        target_ssim = averages["target_ssim"]
                        ss_psnr = averages["ss_psnr"]
                        ss_lpips = averages["ss_lpips"]
                        ss_ssim = averages["ss_ssim"]
                        ood_target_psnr = averages["ood_target_psnr"]
                        ood_target_lpips = averages["ood_target_lpips"]
                        ood_target_ssim = averages["ood_target_ssim"]
                        wandb.log({
                            f"{test_name}/input_psnr": input_psnr,
                            f"{test_name}/input_lpips": input_lpips,
                            f"{test_name}/input_ssim": input_ssim,
                            f"{test_name}/target_psnr": target_psnr,
                            f"{test_name}/target_lpips": target_lpips,
                            f"{test_name}/target_ssim": target_ssim,
                            f"{test_name}/ss_psnr": ss_psnr,
                            f"{test_name}/ss_lpips": ss_lpips,
                            f"{test_name}/ss_ssim": ss_ssim,
                            f"{test_name}/ood_target_psnr": ood_target_psnr,
                            f"{test_name}/ood_target_lpips": ood_target_lpips,
                            f"{test_name}/ood_target_ssim": ood_target_ssim,
                        }, step=cur_train_step)

                    # log scene images to wandb, only uid 162 and 183 scenes are logged
                    if ddp_info.is_main_process:
                        for uid in [162, 183]:
                            sample_dir = os.path.join(out_dir, f"{uid:06d}")
                            if not os.path.exists(sample_dir):
                                continue
                            prefix = f"{real_n_iters}iters_"
                            input_img = Image.open(os.path.join(sample_dir, f"{prefix}input.png"))
                            target_img = Image.open(os.path.join(sample_dir, f"{prefix}target.png"))
                            ss_img = Image.open(os.path.join(sample_dir, f"{prefix}ss.png"))
                            ood_target_img = Image.open(os.path.join(sample_dir, f"{prefix}ood_target.png"))
                            wandb.log({
                                f"{test_name}/uid_{uid}/input": wandb.Image(input_img),
                                f"{test_name}/uid_{uid}/target": wandb.Image(target_img),
                                f"{test_name}/uid_{uid}/ss": wandb.Image(ss_img),
                                f"{test_name}/uid_{uid}/ood_target": wandb.Image(ood_target_img),
                            }, step=cur_train_step)

    try:
        data = next(train_loader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting train_loader epoch to {cur_epoch}; might take a while...")
        train_sampler.set_epoch(cur_epoch)
        train_loader_iter = iter(train_loader)
        data = next(train_loader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}

    n_iters = 1
    if is_ttt and config.model.ttt.supervise_mode == "g3r":
        # When we follow the G3R supervision manner, we backpropagate the supervision loss n times per data sample.
        n_iters = config.model.ttt.n_layer * config.model.ttt.n_iters_per_layer
        if config.model.ttt.progressive:
            # run the first 10K iterations with 1 iters, then linearly grow to the max number of iters in the next 10K iterations
            n_iters = min(n_iters, int(1 + (n_iters - 1) * min(1.0, max(0, (cur_train_step - config.model.ttt.warmup_steps) / config.model.ttt.warmup_steps))))
        input = None
        target = None
        ss = None
        ood_target = None
        s = None
        ss_pose_tokens = None
        target_pose_tokens = None
        ood_target_pose_tokens = None
        ttt_metrics = {"layers": []}
        ttt_metrics["n_iters"] = n_iters
    
    for idx in range(n_iters):
        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            if is_ttt and config.model.ttt.supervise_mode == "g3r":
                is_last = (idx == n_iters - 1)
                layer_idx = 0 # always use one layer
                iter_idx = idx % config.model.ttt.n_iters_per_layer
                t = idx / n_iters
                
                # in g3r, input loss metrics and target loss metrics are calculated on the updated state s.
                input, target, ss, ood_target, input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss, s, ss_pose_tokens, target_pose_tokens, ood_target_pose_tokens, layer_metrics = model(
                    batch,
                    num_input_views=config.training.num_input_views,
                    num_target_views=config.training.num_target_views,
                    num_ss_views=config.training.num_ss_views,
                    num_ood_target_views=config.training.num_ood_target_views,
                    is_g3r=True,
                    has_target_image=True,
                    training=True,
                    layer_idx=layer_idx,
                    iter_idx=iter_idx,
                    t=t,
                    input=input,
                    target=target,
                    ss=ss,
                    ood_target=ood_target,
                    s=s,
                    ss_pose_tokens=ss_pose_tokens,
                    target_pose_tokens=target_pose_tokens,
                    ood_target_pose_tokens=ood_target_pose_tokens,
                    is_last=is_last,
                    input_views_ss=config.model.ttt.ss_4views, # whether to use input views as well in calculating ss loss
                )
                
                ttt_metrics['layers'].append(layer_metrics)
            elif is_ttt:
                raise NotImplementedError("TTT without G3R supervision is not supported yet")
            else:
                input, target, ss, ood_target, input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss = model(
                    batch,
                    num_input_views=config.training.num_input_views,
                    num_target_views=config.training.num_target_views,
                    num_ss_views=config.training.num_ss_views,
                    num_ood_target_views=config.training.num_ood_target_views,
                    has_target_image=True,
                    training=True,
                )

        update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
        
        # Only sync gradients on the final gradient accumulation step
        if update_grads:
            # Final step in gradient accumulation - sync gradients
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            # Intermediate step - don't sync yet
            with model.no_sync():
                scaler.scale(loss / grad_accum_steps).backward()

        total_grad_norm = None
        if update_grads:
            skip_optimizer_step = False
            # Skip optimizer step if loss is NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected, skip this iteration")
                skip_optimizer_step = True
                if config.training.supervision == "target":
                    target_loss_metrics.loss.data = torch.zeros_like(loss)
                elif config.training.supervision == "input":
                    input_loss_metrics.loss.data = torch.zeros_like(loss)

            # Check gradient norm and update optimizer if everything is fine
            if not skip_optimizer_step:
                # Unscales the gradients
                scaler.unscale_(optimizer) 
                # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
                with torch.no_grad():
                    for n, p in optimized_param_dict.items():
                        if p.requires_grad and (p.grad is not None):
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
            
                # Debug: show gradient norms for key modules to verify updates
                # try:
                #     if ddp_info.local_rank == 0:
                #         mdl = model.module if hasattr(model, "module") else model
                #         def module_grad_norm(mod):
                #             sq = 0.0
                #             found = False
                #             for p in mod.parameters():
                #                 if p.grad is not None:
                #                     g = p.grad.detach()
                #                     sq += float(torch.sum(g * g).item())
                #                     found = True
                #             return (sq ** 0.5) if found else 0.0
                #         # latent grad norm
                #         if hasattr(mdl, "n_light_field_latent") and isinstance(mdl.n_light_field_latent, torch.nn.Parameter):
                #             latent_gnorm = mdl.n_light_field_latent.grad.detach().norm().item() if mdl.n_light_field_latent.grad is not None else 0.0
                #         else:
                #             latent_gnorm = 0.0
                #         # image tokenizer grad norm
                #         img_tok_gnorm = module_grad_norm(mdl.image_tokenizer) if hasattr(mdl, "image_tokenizer") else 0.0
                #         # encoder grad norm
                #         enc_gnorm = module_grad_norm(mdl.transformer_encoder) if hasattr(mdl, "transformer_encoder") else 0.0
                #         # decoder grad norm
                #         dec_gnorm = module_grad_norm(mdl.transformer_decoder) if hasattr(mdl, "transformer_decoder") else 0.0
                #         print(f"[step {cur_train_step} iter {idx}] grad_norms: latent={latent_gnorm:.6e}, img_tokenizer={img_tok_gnorm:.6e}, encoder={enc_gnorm:.6e}, decoder={dec_gnorm:.6e}")
                # except Exception:
                #     pass

                # visualize the grad norm of each layer of our transformer (FOR DEBUG)
                if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                    grad_norms = {}  # Dictionary to store norms per layer
                    for name, param in model.named_parameters():
                        if param.grad is not None:  # Some parameters might not have gradients
                            grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                    for layer_name, grad_norm in grad_norms.items():
                        wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

                total_grad_norm = 0.0
                if config.training.grad_clip_norm > 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                    if total_grad_norm > config.training.grad_clip_norm * 2.0:
                        print(f"WARNING: step {cur_train_step} {idx}th iter grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                    allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                    if total_grad_norm > allowed_gradnorm:
                        skip_optimizer_step = True
                        print(f"WARNING: step {cur_train_step} {idx}th iter grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                    # show grad norm in wandb if it's too large
                    display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                    if display_grad_norm and ddp_info.is_main_process:
                        wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

                # since skip flag may be updated because of grad norm, we check it again
                if not skip_optimizer_step:
                    scaler.step(optimizer)
                    cur_param_update_step += 1

            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if is_ttt and config.model.ttt.supervise_mode == "g3r":
                s = s.detach().requires_grad_(True)
                ss_pose_tokens = ss_pose_tokens.detach()
                target_pose_tokens = target_pose_tokens.detach()
                ood_target_pose_tokens = ood_target_pose_tokens.detach()

    # for g3r, the lr scheduler will be updated after all the inner iterations are done
    lr_scheduler.step()
    cur_train_step += 1
    export_inter_results = (((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0) and (config.training.vis_every > 0))

    # log and save checkpoint
    if ddp_info.is_main_process:
        input_loss_dict = {k: float(f"{v.item():.6f}") for k, v in input_loss_metrics.items()}
        target_loss_dict = {k: float(f"{v.item():.6f}") for k, v in target_loss_metrics.items()}
        ss_loss_dict = {k: float(f"{v.item():.6f}") for k, v in ss_loss_metrics.items()}
        ood_target_loss_dict = {k: float(f"{v.item():.6f}") for k, v in ood_target_loss_metrics.items()}
        
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            if is_ttt:
                print_str += f" | ttt_iters: {ttt_metrics['n_iters']}"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | Elapsed: {(time.time() - start_wall_time):.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
            # Add loss values
            print_str += "\ninput: "
            for k, v in input_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print_str += "\ntarget: "
            for k, v in target_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print_str += "\nss: "
            for k, v in ss_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print_str += "\nood_target: "
            for k, v in ood_target_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            
            if is_ttt and config.model.ttt.distill_factor > 0.0:
                print_str += f"\ndistillation: {ttt_metrics['last_distillation_loss']:.6f}"
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "elapsed_time": time.time() - start_wall_time,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/input/" + k: v for k, v in input_loss_dict.items()})
            log_dict.update({"train/target/" + k: v for k, v in target_loss_dict.items()})
            log_dict.update({"train/ss/" + k: v for k, v in ss_loss_dict.items()})
            log_dict.update({"train/ood_target/" + k: v for k, v in ood_target_loss_dict.items()})
            if is_ttt and config.model.ttt.distill_factor > 0.0:
                log_dict["train/distillation"] = ttt_metrics["last_distillation_loss"]
            
            # Add TTT metrics to logging
            if is_ttt:
                assert ttt_metrics is not None, "TTT metrics are not found"
                log_dict["ttt/n_iters"] = ttt_metrics["n_iters"]
                # Log per-layer metrics
                if 'layers' in ttt_metrics and len(ttt_metrics['layers']) > 0:
                    for i, layer_metrics in enumerate(ttt_metrics['layers']):
                        for key, value in layer_metrics.items():
                            log_dict[f'ttt/layer_{i}/{key}'] = value
            
                # Add TTT gradient metrics
                if hasattr(model.module if hasattr(model, 'module') else model, 'ttt_blocks'):
                    ttt_blocks = (model.module if hasattr(model, 'module') else model).ttt_blocks
                    total_ttt_grad_norm = 0.0
                    if ttt_blocks is not None:
                        for i, block in enumerate(ttt_blocks):
                            block_grad_norm = 0.0
                            max_grad = 0.0
                            for name, param in block.named_parameters():
                                if param.grad is not None:
                                    grad_norm = torch.norm(param.grad).item()
                                    block_grad_norm += grad_norm * grad_norm
                                    max_grad = max(max_grad, torch.max(torch.abs(param.grad)).item())
                            
                            if block_grad_norm > 0:
                                block_grad_norm = block_grad_norm ** 0.5
                                log_dict[f'ttt/grad/block_{i}_norm'] = block_grad_norm
                                log_dict[f'ttt/grad/block_{i}_max'] = max_grad
                                total_ttt_grad_norm += block_grad_norm * block_grad_norm
                            else:
                                log_dict[f'ttt/grad/block_{i}_norm'] = 0.0
                                log_dict[f'ttt/grad/block_{i}_max'] = 0.0
                    
                    if total_ttt_grad_norm > 0:
                        log_dict['ttt/grad/total_norm'] = (total_ttt_grad_norm ** 0.5)
                    else:
                        log_dict['ttt/grad/total_norm'] = 0.0

            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps) or (cur_train_step == total_train_steps // 2):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")

            # if current ckpt folder has more than 4 checkpoints, delete the oldest one, but always keep the half-process checkpoint.
            if len(os.listdir(config.training.checkpoint_dir)) > config.training.get('max_checkpoints', 4):
                ckpts = sorted(
                    [f for f in os.listdir(config.training.checkpoint_dir) if f.endswith(".pt")],
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
                for ckpt in ckpts[:-config.training.get('max_checkpoints', 4)]:
                    # Skip deleting the half-process checkpoint, and skip deleting all "10k" checkpoints (e.g., 10000, 20000, etc.)
                    if ckpt == f"ckpt_{total_train_steps // 2:016}.pt" or int(ckpt.split("_")[1].split(".")[0]) % 10000 == 0:
                        continue
                    os.remove(os.path.join(config.training.checkpoint_dir, ckpt))

        # export intermediate visualization results
        # if export_inter_results:
        #     vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
        #     os.makedirs(vis_path, exist_ok=True)
        #     visualize_intermediate_results(vis_path, input, target, ss, ood_target, rendered_input, rendered_target, rendered_ss, rendered_ood_target)
        #     model.train()
    
    # delete all tensors
    del batch, input, target, ss, ood_target, input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss
    if is_ttt:
        del ttt_metrics
    if is_ttt and config.model.ttt.supervise_mode == "g3r":
        del s, ss_pose_tokens, target_pose_tokens, ood_target_pose_tokens
    if ddp_info.is_main_process:
        del input_loss_dict, target_loss_dict, ss_loss_dict, ood_target_loss_dict
    torch.cuda.empty_cache()
    
    dist.barrier() 

dist.barrier()
dist.destroy_process_group()
