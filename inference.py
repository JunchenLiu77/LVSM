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
dataset = Dataset(
    config, 
    dataset_path="/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt", 
    num_input_views=config.training.num_input_views, 
    num_target_views=config.training.num_target_views, 
    inference=True
)

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

if is_ttt:
    enc_views, ss_views = [], []
    iters = config.training.test_layers
    if config.training.test_1enc1ss:
        enc_views.append(1)
        ss_views.append(1)
    if config.training.test_2enc2ss:
        enc_views.append(2)
        ss_views.append(2)
    # hacky way to test 4 views encoder and 4 views ss
    # enc_views.append(8)
    # ss_views.append(8)
    # enc_views.append(2)
    # ss_views.append(30)
    assert len(enc_views) > 0 and len(ss_views) > 0 and len(iters) > 0, "At least one test setting should be specified"

if config.inference.get("first_n_batches", None) is not None:
    print(f"Running first {config.inference.get('first_n_batches', None)} batches")

dataloader_iter = iter(dataloader)
out_dir = config.training.checkpoint_dir
os.makedirs(out_dir, exist_ok=True)
with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for (idx, batch) in enumerate(dataloader_iter):
        print(f"[Rank {ddp_info.local_rank}] Running inference on the {idx}th batch")
        if config.inference.get("first_n_batches", None) is not None and idx >= config.inference.get("first_n_batches", None):
            break
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        if is_ttt:
            for n_iters in iters:
                real_n_iters = n_iters
                # if config.model.ttt.progressive:
                #     # bound the number of iterations by the warmup steps
                #     real_n_iters = int(1 + (n_iters - 1) * min(1.0, cur_train_step / config.model.ttt.warmup_steps))
                for i in range(len(enc_views)):
                    n_encoder_views = enc_views[i]
                    n_ss_views = ss_views[i]
                    if config.model.ttt.supervise_mode != "g3r":
                        input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, ttt_metrics = model(
                            batch,
                            num_input_views=config.training.num_input_views,
                            num_target_views=3,
                            is_g3r=False, # at inference time, whether using G3R supervision behaves the same.
                            n_encoder_views=n_encoder_views,
                            n_ss_views=n_ss_views,
                            n_iters=real_n_iters,
                            has_target_image=True,
                            target_has_input=False,
                        )
                    else:
                        input = None
                        target = None
                        s = None
                        full_encoded_latents = None
                        input_pose_tokens = None
                        target_pose_tokens = None
                        
                        for idx in range(real_n_iters):
                            is_last = (idx == real_n_iters - 1)
                            layer_idx = 0
                            iter_idx = idx % config.model.ttt.n_iters_per_layer

                            # in g3r, input loss metrics and target loss metrics are calculated on the updated state s.
                            input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, s, full_encoded_latents, input_pose_tokens, target_pose_tokens, layer_metrics = model(
                                batch,
                                num_input_views=config.training.num_input_views,
                                num_target_views=3,
                                is_g3r=True,
                                n_encoder_views=n_encoder_views,
                                n_ss_views=n_ss_views,
                                has_target_image=True,
                                target_has_input=False,
                                layer_idx=layer_idx,
                                iter_idx=iter_idx,
                                input=input,
                                target=target,
                                s=s,
                                full_encoded_latents=full_encoded_latents,
                                input_pose_tokens=input_pose_tokens,
                                target_pose_tokens=target_pose_tokens,
                                is_last=is_last,
                            )
                    # export results with the iterations upper bound
                    export_results(input, target, rendered_input, rendered_target, out_dir, compute_metrics=config.inference.get("compute_metrics"), n_encoder_views=n_encoder_views, n_ss_views=n_ss_views, n_iters=real_n_iters)
        else:
            input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, ttt_metrics = model(
                batch,
                num_input_views=config.training.num_input_views,
                num_target_views=3,
                has_target_image=True,
                target_has_input=False,
            )
            export_results(input, target, rendered_input, rendered_target, out_dir, compute_metrics=config.inference.get("compute_metrics"))
    dist.barrier()
    if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
        if is_ttt:
            for n_iters in iters:
                for i in range(len(enc_views)):
                    n_encoder_views = enc_views[i]
                    n_ss_views = ss_views[i]
                    input_psnr, input_lpips, input_ssim, target_psnr, target_lpips, target_ssim = summarize_evaluation(out_dir, n_encoder_views=n_encoder_views, n_ss_views=n_ss_views, n_iters=n_iters)
                    
                    # log results in wandb
                    # test_name = f"test_{n_encoder_views}enc{n_ss_views}ss_{n_iters}iters"
                    # wandb.log({
                    #     f"{test_name}/input_psnr": input_psnr,
                    #     f"{test_name}/input_lpips": input_lpips,
                    #     f"{test_name}/input_ssim": input_ssim,
                    #     f"{test_name}/target_psnr": target_psnr,
                    #     f"{test_name}/target_lpips": target_lpips,
                    #     f"{test_name}/target_ssim": target_ssim,
                    # }, step=cur_train_step)
        else:
            input_psnr, input_lpips, input_ssim, target_psnr, target_lpips, target_ssim = summarize_evaluation(out_dir)

        # if config.inference.get("generate_website", True):
        #     os.system(f"python generate_html.py {out_dir}")


dist.barrier()

# if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
#     summarize_evaluation(config.training.checkpoint_dir)
#     if config.inference.get("generate_website", True):
#         os.system(f"python generate_html.py {config.training.checkpoint_dir}")
# dist.barrier()
dist.destroy_process_group()
exit(0)