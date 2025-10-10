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
ddp_info = init_distributed(seed=777, deterministic=True)
print(f"DDP info: {ddp_info}")
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
model.module.load_ckpt(config.training.resume_ckpt)
is_ttt = "ttt" in module


if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {config.training.checkpoint_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

if config.inference.get("first_n_batches", None) is not None:
    print(f"Running first {config.inference.get('first_n_batches', None)} batches")

# TTT need calculate gradient
with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for (idx, batch) in enumerate(dataloader):
        print(f"Running  the {idx}th batch")
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        if config.inference.get("first_n_batches", None) is not None and idx >= config.inference.get("first_n_batches", None):
            break
        
        n_iters = 1
        if is_ttt and config.model.ttt.supervise_mode == "g3r":
            # When we follow the G3R supervision manner, we backpropagate the supervision loss n times per data sample.
            n_iters = config.model.ttt.n_layer * config.model.ttt.n_iters_per_layer + 1
            # doesn't have model parameter synchronization here, so we can use model.module directly
            input = None
            target = None
            s = None
            full_encoded_latents = None
            input_pose_tokens = None
            target_pose_tokens = None
            ttt_metrics = {"layers": []}
        
        for idx in range(n_iters):
            if is_ttt and config.model.ttt.supervise_mode == "g3r":
                is_last = (idx == n_iters - 1)
                layer_idx = idx // config.model.ttt.n_iters_per_layer
                iter_idx = idx % config.model.ttt.n_iters_per_layer
                # TODO: kinda hacky here
                if is_last:
                    layer_idx = config.model.ttt.n_layer - 1
                    iter_idx = config.model.ttt.n_iters_per_layer - 1
                
                input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, s, full_encoded_latents, input_pose_tokens, target_pose_tokens, layer_metrics = model(
                    batch,
                    has_target_image=True,
                    layer_idx=layer_idx,
                    iter_idx=iter_idx,
                    input=input,
                    target=target,
                    s=s,
                    full_encoded_latents=full_encoded_latents,
                    input_pose_tokens=input_pose_tokens,
                    target_pose_tokens=target_pose_tokens,
                    update=not is_last
                )
                
                s = s.detach().requires_grad_(True)
                full_encoded_latents = full_encoded_latents.detach() if full_encoded_latents is not None else None
                input_pose_tokens = input_pose_tokens.detach()
                target_pose_tokens = target_pose_tokens.detach()
                
                if not is_last:
                    ttt_metrics['layers'].append(layer_metrics)
                else:
                    last_layer_metrics = layer_metrics
                    ttt_metrics["last_input_loss"] = last_layer_metrics["input_loss"]
                    ttt_metrics["last_target_loss"] = last_layer_metrics["target_loss"]
                    if config.model.ttt.distill_factor > 0.0:
                        ttt_metrics["last_distillation_loss"] = last_layer_metrics["distillation_loss"]
            else:
                input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, ttt_metrics = model(batch)
        if config.inference.get("render_video", False):
            raise NotImplementedError("Need some closer look here.")
            result= model.module.render_video(result, **config.inference.render_video_config)
        export_results(input, target, rendered_input, rendered_target, config.training.checkpoint_dir, compute_metrics=config.inference.get("compute_metrics"))
        del input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, ttt_metrics
        torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.training.checkpoint_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.training.checkpoint_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)