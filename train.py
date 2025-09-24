# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

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
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)


optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
    is_ttt="ttt" in module,
    freeze_encoder=config.training.get("freeze_encoder", False),
    freeze_decoder=config.training.get("freeze_decoder", False),
    freeze_tokenizer=config.training.get("freeze_tokenizer", False),
    freeze_latent=config.training.get("freeze_latent", False),
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

cur_train_step = 0
cur_param_update_step = 0
if config.training.get("resume_ckpt", "") != "":
    optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
        ckpt_load_path,
        model,
        optimizer,
        lr_scheduler,
        reset_training_state,
    )

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        # if start_train_step == cur_train_step:
        #     print(f"Current Rank {ddp_info.local_rank} Restarting training from step {cur_train_step}. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        #     datasampler.set_epoch(cur_epoch)
        #     dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}


    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(batch)

    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if config.training.supervision == "target":
        loss = ret_dict.target_loss_metrics.loss
    elif config.training.supervision == "input":
        loss = ret_dict.input_loss_metrics.loss
    else:
        raise ValueError(f"Invalid supervision type: {config.training.supervision}")
    
    if update_grads:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(loss / grad_accum_steps).backward()
    else:
        scaler.scale(loss / grad_accum_steps).backward()
    cur_train_step += 1

    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

    if update_grads:
        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            if config.training.supervision == "target":
                ret_dict.target_loss_metrics.loss.data = torch.zeros_like(loss)
            elif config.training.supervision == "input":
                ret_dict.input_loss_metrics.loss.data = torch.zeros_like(loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if not skip_optimizer_step:
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        
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
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                # show grad norm in wandb if it's too large
                display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                if display_grad_norm and ddp_info.is_main_process:
                    wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

            # since skip flag may be updated because of grad norm, we check it again
            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # log and save checkpoint
    if ddp_info.is_main_process:
        target_loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.target_loss_metrics.items()}
        input_loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.input_loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
            # Add loss values
            print_str += "\ntarget: "
            for k, v in target_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print_str += "\ninput: "
            for k, v in input_loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
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
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/target/" + k: v for k, v in target_loss_dict.items()})
            log_dict.update({"train/input/" + k: v for k, v in input_loss_dict.items()})

            # Add TTT metrics to logging
            if hasattr(ret_dict, 'ttt_metrics') and ret_dict.ttt_metrics is not None:
                ttt_metrics = ret_dict.ttt_metrics
                log_dict['ttt/initial_state_norm'] = ttt_metrics['initial_state_norm']
                log_dict['ttt/final_state_norm'] = ttt_metrics['final_state_norm']
                
                # Log per-layer metrics
                if 'layers' in ttt_metrics and len(ttt_metrics['layers']) > 0:
                    for i, layer_metrics in enumerate(ttt_metrics['layers']):
                        for key, value in layer_metrics.items():
                            log_dict[f'ttt/layer_{i}/{key}'] = value
                    
                    # Average metrics across layers
                    layer_keys = ttt_metrics['layers'][0].keys()
                    for key in layer_keys:
                        values = [layer[key] for layer in ttt_metrics['layers']]
                        log_dict[f'ttt/avg_{key}'] = sum(values) / len(values)
            
            # Add TTT gradient metrics
            if hasattr(model.module if hasattr(model, 'module') else model, 'ttt_blocks'):
                ttt_blocks = (model.module if hasattr(model, 'module') else model).ttt_blocks
                total_ttt_grad_norm = 0.0
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
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
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
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)
            torch.cuda.empty_cache()
            model.train()

            
    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()
        

dist.barrier()
dist.destroy_process_group()
