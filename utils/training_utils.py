# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def fix_checkpoint_state_dict_compatibility(model, checkpoint_state_dict):
    """Fix compatibility between compiled and non-compiled model state dicts."""
    # Get model state dict keys to determine structure
    if hasattr(model, 'module'):  # DDP wrapped model
        model_keys = set(model.module.state_dict().keys())
    else:
        model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint_state_dict.keys())
    # Check if model is compiled (has _orig_mod prefix)
    model_compiled = any(k.startswith('_orig_mod.') for k in model_keys)
    checkpoint_compiled = any(k.startswith('_orig_mod.') for k in checkpoint_keys)
    # If both have same compilation status, return as-is
    if model_compiled == checkpoint_compiled:
        return checkpoint_state_dict
    # Case 1: Model is compiled, checkpoint is not compiled
    if model_compiled and not checkpoint_compiled:
        print('Converting non-compiled checkpoint for compiled model (adding _orig_mod prefixes)')
        return {f'_orig_mod.{k}': v for k, v in checkpoint_state_dict.items()}
    # Case 2: Model is not compiled, checkpoint is compiled
    if (not model_compiled) and checkpoint_compiled:
        print('Converting compiled checkpoint for non-compiled model (removing _orig_mod prefixes)')
        fixed = {}
        for k, v in checkpoint_state_dict.items():
            fixed[k[len('_orig_mod.'):]] = v if k.startswith('_orig_mod.') else v
        return fixed
    return checkpoint_state_dict


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def create_optimizer(
    model, 
    weight_decay, 
    learning_rate, 
    betas, 
    is_ttt=False, 
    learning_rate_ttt=None, 
    freeze_encoder=False, 
    freeze_decoder=False, 
    freeze_tokenizer=False, 
    freeze_latent=False, 
    freeze_ttt=False
):
    # if is_ttt, then 'freeze' parameters determine whether to freeze the encoder and decoder parameters.
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    if not is_ttt:
        # filter out those that do not require grad
        optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}
    else:
        # if ttt, freeze the encoder and decoder parameters. For light_field_latent, we need to keed the gradient of it but
        # dont update it through optimizer.
        for name, param in all_param_dict.items():
            if "ttt" in name or "light_field_latent" in name:
                continue
            if (not freeze_encoder and "encoder" in name) or \
                (not freeze_decoder and "decoder" in name) or \
                (not freeze_tokenizer and "tokenizer" in name) or \
                (not freeze_ttt and "ttt" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # only update ttt blocks
        optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad and ("light_field_latent" not in name or not freeze_latent)}

    decay_params, nodecay_params = [], []
    decay_params_ttt, nodecay_params_ttt = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            if "ttt" in name:
                nodecay_params_ttt.append(param)
            else:
                nodecay_params.append(param)
        else:
            if "ttt" in name:
                decay_params_ttt.append(param)
            else:
                decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
    ]
    if is_ttt:
        assert learning_rate_ttt is not None, "learning_rate_ttt must be provided for TTT"
        optim_groups.append({'params': decay_params_ttt, 'weight_decay': weight_decay, 'lr': learning_rate_ttt})
        optim_groups.append({'params': nodecay_params_ttt, 'weight_decay': 0.0, 'lr': learning_rate_ttt})
        
    # use fused AdamW optimizer by default. 
    optimizer = torch.optim.AdamW(optim_groups, betas=betas,fused=True)
    
    # Print Model Information
    if dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            elif len(parts) > 1 and parts[0] == 'module':
                return parts[1]
            return parts[0]  # Fallback to first part if no 'module.' prefix
        print(f'Optimizer: AdamW, learning rate: {learning_rate}, learning rate_ttt: {learning_rate_ttt}, weight decay: {weight_decay}, betas: {betas}')
        # Number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
        
        print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')
        
    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights with compilation compatibility fix
    fixed_state_dict = fix_checkpoint_state_dict_compatibility(model, checkpoint['model'])
    
    if isinstance(model, DDP):
        status = model.module.load_state_dict(fixed_state_dict, strict=False)
    else:
        status = model.load_state_dict(fixed_state_dict, strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step


