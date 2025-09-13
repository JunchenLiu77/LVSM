# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modified version with detailed profiling

import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
from collections import defaultdict
import numpy as np

# Profiling utilities
class TimingProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
        
    def start_timer(self, name):
        self.current_timers[name] = time.perf_counter()
        
    def end_timer(self, name):
        if name in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[name]
            self.timings[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return 0
    
    def get_stats(self, name):
        if name in self.timings and len(self.timings[name]) > 0:
            values = np.array(self.timings[name])
            # Exclude first few for warmup
            if len(values) > 5:
                values = values[5:]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return None
    
    def print_summary(self):
        print("\n" + "="*60)
        print("PROFILING SUMMARY (excluding first 5 iterations)")
        print("="*60)
        
        # Sort by mean time
        sorted_names = sorted(self.timings.keys(), 
                            key=lambda x: np.mean(self.timings[x][5:] if len(self.timings[x]) > 5 else self.timings[x]),
                            reverse=True)
        
        total_time = 0
        for name in sorted_names:
            stats = self.get_stats(name)
            if stats:
                print(f"\n{name}:")
                print(f"  Mean: {stats['mean']*1000:.2f} ms")
                print(f"  Std:  {stats['std']*1000:.2f} ms")
                print(f"  Min:  {stats['min']*1000:.2f} ms")
                print(f"  Max:  {stats['max']*1000:.2f} ms")
                print(f"  Samples: {stats['count']}")
                if 'total_iteration' not in name:
                    total_time += stats['mean']
        
        print(f"\nSum of components: {total_time*1000:.2f} ms")
        print("="*60 + "\n")

profiler = TimingProfiler()

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
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

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))


module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])


optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)


if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(ckpt_load_path, model, optimizer, lr_scheduler, reset_training_state)
start_train_step = cur_train_step

assert config.training.use_amp, "This profiled version requires AMP to be enabled"
scaler = torch.cuda.amp.GradScaler(enabled=(config.training.amp_dtype == "fp16"))
if ddp_info.is_main_process:
    print_rank0(f"Using AMP with dtype: {config.training.amp_dtype}")
    print_rank0(f"Grad scaler enabled: {scaler.is_enabled()}")

tic = time.time()
cur_epoch = 0
tic_iter = time.time()

# Main training loop with profiling
for cur_train_step in range(start_train_step, total_train_steps):
    profiler.start_timer('total_iteration')
    
    # Data loading
    profiler.start_timer('data_loading')
    try:
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    data_load_time = profiler.end_timer('data_loading')

    # Data transfer to GPU
    profiler.start_timer('data_to_gpu')
    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}
    profiler.end_timer('data_to_gpu')

    # Forward pass
    profiler.start_timer('forward_pass')
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(batch)
    torch.cuda.synchronize()  # Ensure forward is complete
    forward_time = profiler.end_timer('forward_pass')

    # Backward pass
    profiler.start_timer('backward_pass')
    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    torch.cuda.synchronize()  # Ensure backward is complete
    backward_time = profiler.end_timer('backward_pass')
    
    cur_train_step += 1

    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

    # Optimizer step
    if update_grads:
        profiler.start_timer('optimizer_step')
        
        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if not skip_optimizer_step:
            profiler.start_timer('grad_unscale')
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            profiler.end_timer('grad_unscale')
            
            profiler.start_timer('grad_nan_handling')
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
            profiler.end_timer('grad_nan_handling')
        
            # visualize the grad norm of each layer of our transformer (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}  # Dictionary to store norms per layer
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Some parameters might not have gradients
                        grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                for layer_name, grad_norm in grad_norms.items():
                    wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

            profiler.start_timer('grad_clipping')
            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                if total_grad_norm > config.training.grad_clip_norm * 2.0:
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                # show grad norm in wandb if it's too large
                display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                if display_grad_norm and ddp_info.is_main_process:
                    wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)
            profiler.end_timer('grad_clipping')

            # since skip flag may be updated because of grad norm, we check it again
            if not skip_optimizer_step:
                profiler.start_timer('optimizer_update')
                scaler.step(optimizer)
                cur_param_update_step += 1
                profiler.end_timer('optimizer_update')

        profiler.start_timer('scaler_lr_update')
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        profiler.end_timer('scaler_lr_update')
        
        optimizer_time = profiler.end_timer('optimizer_step')
    else:
        optimizer_time = 0

    # Logging
    profiler.start_timer('logging')
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            total_iter_time = profiler.end_timer('total_iteration')
            
            # Print detailed timing breakdown
            print_str = f"\n[Epoch {int(cur_epoch):>3d}] | Step: {int(cur_train_step):>6d} (Param: {int(cur_param_update_step):>6d})"
            print_str += f" | Total: {total_iter_time*1000:.1f}ms | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            
            # Timing breakdown
            print_str += f"TIMING: Data: {data_load_time*1000:.1f}ms | "
            print_str += f"Fwd: {forward_time*1000:.1f}ms | "
            print_str += f"Bwd: {backward_time*1000:.1f}ms | "
            if update_grads:
                print_str += f"Opt: {optimizer_time*1000:.1f}ms\n"
            else:
                print_str += "Opt: skip\n"
            
            # Add loss values
            print_str += "LOSS: "
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)
            
            # Every 50 iterations, print profiling summary
            if cur_train_step % 50 == 0 and cur_train_step > start_train_step + 10:
                profiler.print_summary()
        else:
            profiler.end_timer('total_iteration')

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
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
                # Add detailed timing to wandb
                "timing/data_loading": data_load_time * 1000,
                "timing/forward_pass": forward_time * 1000,
                "timing/backward_pass": backward_time * 1000,
            }
            if update_grads:
                log_dict["timing/optimizer_step"] = optimizer_time * 1000
            
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            profiler.start_timer('checkpoint_save')
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
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"model_{cur_train_step:06d}.pt")
            torch.save(checkpoint, ckpt_path)
            if ddp_info.is_main_process:
                # dump out config
                config.save_to_yaml(os.path.join(config.training.checkpoint_dir, "config.yaml"))
                # keep the lastest 4 checkpoints
                ckpts = sorted(
                    [f for f in os.listdir(config.training.checkpoint_dir) if f.endswith(".pt")],
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
                if len(ckpts) > config.training.get('max_checkpoints', 4):
                    for ckpt in ckpts[:-config.training.get('max_checkpoints', 4)]:
                        os.remove(os.path.join(config.training.checkpoint_dir, ckpt))
                # Also save a copy as latest.pt
                latest_path = os.path.join(config.training.checkpoint_dir, "latest.pt")
                torch.save(checkpoint, latest_path)
            profiler.end_timer('checkpoint_save')
    else:
        profiler.end_timer('total_iteration')
    
    profiler.end_timer('logging')

#     # visualize intermediate results
#     if export_inter_results and ddp_info.is_main_process:
#         profiler.start_timer('visualization')
#         visualize_intermediate_results(batch, ret_dict, export_path=config.training.output_dir, step=cur_train_step)
#         profiler.end_timer('visualization')

    tic = time.time()
    
    # Update epoch
    cur_epoch = cur_train_step * total_batch_size // len(dataset)

# Final profiling summary
if ddp_info.is_main_process:
    print("\n" + "="*80)
    print("FINAL PROFILING RESULTS")
    print("="*80)
    profiler.print_summary()
    
    # Calculate bottleneck
    print("\nBOTTLENECK ANALYSIS:")
    print("-"*40)
    components = ['data_loading', 'forward_pass', 'backward_pass', 'optimizer_step']
    component_times = {}
    total_component_time = 0
    
    for comp in components:
        stats = profiler.get_stats(comp)
        if stats:
            component_times[comp] = stats['mean'] * 1000  # Convert to ms
            total_component_time += component_times[comp]
    
    if total_component_time > 0:
        for comp, time_ms in sorted(component_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_ms / total_component_time) * 100
            print(f"{comp:20s}: {time_ms:6.1f} ms ({percentage:5.1f}%)")
    
    print("-"*40)
    print(f"Total accounted time: {total_component_time:.1f} ms")
    
    # Check for any unaccounted time
    total_stats = profiler.get_stats('total_iteration')
    if total_stats:
        total_time_ms = total_stats['mean'] * 1000
        unaccounted = total_time_ms - total_component_time
        if unaccounted > 0:
            print(f"Unaccounted overhead: {unaccounted:.1f} ms ({(unaccounted/total_time_ms)*100:.1f}%)")
    
    print("="*80)

print_rank0("Training completed!")
if ddp_info.is_main_process:
    wandb.finish()
