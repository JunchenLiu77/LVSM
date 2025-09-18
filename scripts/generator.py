#!/usr/bin/env python3
"""
LVSM Training Script Generator

This script generates customized SLURM job scripts for training LVSM models
with flexible configuration options.

Usage:
    python scripts/generator.py [options]
    
Examples:
    # Basic decoder-only training
    python scripts/generator.py --model decoder-only --steps 1000
    
    # Encoder-decoder with custom image size
    python scripts/generator.py --model encoder-decoder --image-size 512 --batch-size 4
    
    # Inference mode
    python scripts/generator.py --inference --model decoder-only --resume-ckpt latest.pt
"""

import argparse
import os
from datetime import datetime
from pathlib import Path


class Generator:
    def __init__(self):
        self.model_config = {
            'decoder-only': 'configs/LVSM_scene_decoder_only.yaml',
            'encoder-decoder': 'configs/LVSM_scene_encoder_decoder.yaml',
            "encoder-decoder-ttt": "configs/LVSM_scene_encoder_decoder_ttt.yaml"
        }
        
        # Default SLURM settings
        self.slurm_defaults = {
            'job_name': 'lvsm',
            'account': 'aip-fsanja',
            'time': '00-03:00:00',
            'nodes': 1,
            'mem': '48GB',
            'cpus_per_task': 8,
            'gpus_per_node': 'l40s:4',
            'partition': 'gpubase_l40s_b1'
        }
        
    def generate_script(self, args):
        """Generate the complete SLURM script"""
        
        # Generate output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{args.model.replace('-', '_')}_{timestamp}"
        output_dir = f"results/{exp_name}"
        script_path = f"{output_dir}/run.sh"
        
        # Determine config file
        config_file = self.model_config[args.model]
        
        # Build parameter overrides
        overrides = self._build_overrides(args, output_dir)
        
        # Generate the script content
        script_content = self._generate_slurm_script(
            args, config_file, overrides, output_dir
        )
        
        # Save the script
        os.makedirs(output_dir, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _build_overrides(self, args, output_dir):
        """Build configuration overrides from arguments"""
        overrides = []
        
        # Checkpoint directory
        overrides.append(f'training.checkpoint_dir="{output_dir}"')
        
        if args.dataset_path:
            overrides.append(f'training.dataset_path="{args.dataset_path}"')
        
        # Model configuration overrides
        if args.image_size:
            overrides.append(f'model.image_tokenizer.image_size={args.image_size}')
            overrides.append(f'model.target_pose_tokenizer.image_size={args.image_size}')
        
        if args.d_model:
            overrides.append(f'model.transformer.d={args.d_model}')
        
        # Handle n_layer for different architectures
        if args.model == 'decoder-only':
            if args.n_layers:
                overrides.append(f'model.transformer.n_layer={args.n_layers}')
        elif args.model == 'encoder-decoder':
            if args.encoder_layers:
                overrides.append(f'model.transformer.encoder_n_layer={args.encoder_layers}')
            if args.decoder_layers:
                overrides.append(f'model.transformer.decoder_n_layer={args.decoder_layers}')
            if args.n_latent:
                overrides.append(f'model.transformer.n_latent_vectors={args.n_latent}')
        elif args.model == 'encoder-decoder-ttt':
            if args.encoder_layers:
                overrides.append(f'model.transformer.encoder_n_layer={args.encoder_layers}')
            if args.decoder_layers:
                overrides.append(f'model.transformer.decoder_n_layer={args.decoder_layers}')
            if args.n_latent:
                overrides.append(f'model.transformer.n_latent_vectors={args.n_latent}')
            if args.ttt_layers:
                overrides.append(f'model.ttt.n_layer={args.ttt_layers}')
            if args.state_lr_mode:
                overrides.append(f'model.ttt.state_lr_mode={args.state_lr_mode}')
            if args.state_lr_init:
                overrides.append(f'model.ttt.state_lr_init={args.state_lr_init}')
            if args.state_lr:
                overrides.append(f'model.ttt.state_lr={args.state_lr}')
            if args.opt_model:
                overrides.append(f'model.ttt.opt_model={args.opt_model}')
            if args.mlp_dim:
                overrides.append(f'model.ttt.mlp_dim={args.mlp_dim}')
            if args.use_positional_encoding:
                overrides.append('model.ttt.use_positional_encoding=true')
            elif args.no_positional_encoding:
                overrides.append('model.ttt.use_positional_encoding=false')
        
        # Training configuration overrides
        if args.batch_size:
            overrides.append(f'training.batch_size_per_gpu={args.batch_size}')
        
        if args.steps:
            overrides.append(f'training.train_steps={args.steps}')
        
        if args.lr:
            overrides.append(f'training.lr={args.lr}')
            
        if args.warmup:
            overrides.append(f'training.warmup={args.warmup}')
        
        if args.resume_ckpt:
            overrides.append(f'training.resume_ckpt="{args.resume_ckpt}"')
            
        if args.reset_training_state:
            overrides.append(f'training.reset_training_state=true')
        elif args.no_reset_training_state:
            overrides.append(f'training.reset_training_state=false')
        
        if args.wandb_exp_name:
            overrides.append(f'training.wandb_exp_name="{args.wandb_exp_name}"')
        
        # Inference mode
        if args.inference:
            overrides.append('inference.if_inference=true')
        elif args.no_inference:
            overrides.append('inference.if_inference=false')
        
        # AMP settings
        if args.amp_dtype:
            overrides.append(f'training.use_amp=true')
            overrides.append(f'training.amp_dtype={args.amp_dtype}')
        
        if args.grad_checkpoint:
            overrides.append(f'training.grad_checkpoint=true')
        elif args.no_grad_checkpoint:
            overrides.append(f'training.grad_checkpoint=false')
        
        return overrides
    
    def _generate_slurm_script(self, args, config_file, overrides, output_dir):
        """Generate the complete SLURM script content"""
        
        # Update SLURM settings based on arguments
        slurm = self.slurm_defaults.copy()
        if args.time:
            slurm['time'] = args.time
        if args.nodes:
            slurm['nodes'] = args.nodes
        if args.gpus:
            slurm['gpus_per_node'] = f'l40s:{args.gpus}'
        if args.memory:
            slurm['mem'] = args.memory
        if args.partition:
            slurm['partition'] = args.partition
        
        # Determine which Python script to use
        if args.inference:
            torchrun_script = 'inference.py'
        else:   
            torchrun_script = 'train.py'
            if args.profile:
                torchrun_script = 'train_profiled.py'
        
        # Build the override string
        override_str = ' \\\n    '.join(overrides) if overrides else ''
        if override_str:
            override_str = ' \\\n    ' + override_str
        
        # Generate the script
        script = f'''#!/bin/bash
#SBATCH --job-name={slurm['job_name']}
#SBATCH --account={slurm['account']}
#SBATCH --output={output_dir}/%x_%j.out
#SBATCH --error={output_dir}/%x_%j.err
#SBATCH --time={slurm['time']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --mem={slurm['mem']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --gpus-per-node={slurm['gpus_per_node']}
#SBATCH --ntasks-per-node=1
#SBATCH --partition={slurm['partition']}

# Generated by generator.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Model: {args.model}
# Mode: {'Inference' if args.inference else 'Training'}

echo "=============================================="
echo "LVSM {'Inference' if args.inference else 'Training'} - {args.model.upper()} Model"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: {args.gpus or 4}x L40s"
echo "Start time: $(date)"
echo "Default Config: {config_file}"
echo "=============================================="

# Load modules
module load python/3.12
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8

# Environment variables
export OMP_NUM_THREADS=2
export IBV_FORK_SAFE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# Optimized NCCL settings (P2P and IB enabled for L40s or A100)
# export NCCL_IB_DISABLE=1  # Uncomment to disable InfiniBand
# export NCCL_P2P_DISABLE=1  # Uncomment to disable P2P

# Debug settings (optional)
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1  # For debugging

# Suppress libibverbs warnings
exec 3>&2
exec 2> >(grep -v 'libibverbs: Warning' >&3)

echo
echo "Starting {'inference' if args.inference else 'training'}..."
echo "Configuration overrides:"'''
        
        # Add override display
        if overrides:
            for override in overrides:
                script += f'\necho "  - {override}"'
        else:
            script += '\necho "  None (using defaults)"'
        
        script += f'''
echo

# Run the training/inference
srun --time {slurm['time']} uv run torchrun \\
    --nproc_per_node={args.gpus or 4} \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    {torchrun_script} \\
    --config {config_file}{override_str}

# Restore stderr
exec 2>&3
exec 3>&-

echo
echo "=============================================="
echo "{'Inference' if args.inference else 'Training'} completed at: $(date)"
echo "=============================================="

exit 0
'''
        return script


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM training scripts for LVSM models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument('--dataset-path', type=str, default="/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/train/full_list.txt",
                        help='Dataset path')
    
    # Model selection
    parser.add_argument('--model', choices=['decoder-only', 'encoder-decoder', 'encoder-decoder-ttt'], 
                        default='decoder-only',
                        help='Model architecture to use')
    
    # Model configuration
    parser.add_argument('--image-size', type=int, 
                        help='Image size for tokenizers (e.g., 256, 512)')
    parser.add_argument('--d-model', type=int,
                        help='Transformer dimension')
    parser.add_argument('--n-layers', type=int,
                        help='Number of layers (decoder-only)')
    parser.add_argument('--encoder-layers', type=int,
                        help='Number of encoder layers (encoder-decoder)')
    parser.add_argument('--decoder-layers', type=int,
                        help='Number of decoder layers (encoder-decoder)')
    parser.add_argument('--n-latent', type=int,
                        help='Number of latent vectors (encoder-decoder)')
    parser.add_argument('--ttt-layers', type=int,
                        help='Number of TTT layers (encoder-decoder-ttt)')
    parser.add_argument('--state-lr-mode', choices=['fixed', 'learnable'],
                        help='State learning rate mode (encoder-decoder-ttt)')
    parser.add_argument('--state-lr-init', type=float,
                        help='Initial value for learnable state_lr (pre-sigmoid), only used when state_lr_mode is "learnable"')
    parser.add_argument('--state-lr', type=float,
                        help='TTT state learning rate (encoder-decoder-ttt)')
    parser.add_argument('--opt-model', choices=['mlp', 'transformer', 'flatten_mlp'],
                        help='TTT optimization model (encoder-decoder-ttt)')
    parser.add_argument('--mlp-dim', type=int,
                        help='TTT MLP dimension (encoder-decoder-ttt)')
    parser.add_argument('--use-positional-encoding', action='store_true', default=None,
                        help='Enable positional encoding in TTT transformer (encoder-decoder-ttt)')
    parser.add_argument('--no-positional-encoding', action='store_true', default=None,
                        help='Disable positional encoding in TTT transformer (encoder-decoder-ttt)')
                        
    # Training configuration
    parser.add_argument('--batch-size', type=int,
                        help='Batch size per GPU')
    parser.add_argument('--steps', type=int,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--warmup', type=int,
                        help='Warmup steps')
    parser.add_argument('--resume-ckpt', type=str,
                        help='Checkpoint path to resume from')
    parser.add_argument('--reset-training-state', action='store_true',
                        help='Reset training state')
    parser.add_argument('--no-reset-training-state', action='store_true',
                        help='Do not reset training state')
    parser.add_argument('--wandb-exp-name', type=str,
                        help='WandB experiment name')
    parser.add_argument('--amp-dtype', choices=['bf16', 'fp16', 'fp32'],
                        help='AMP data type')
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--no-grad-checkpoint', action='store_true',
                        help='Disable gradient checkpointing')
    # Inference mode
    parser.add_argument('--inference', action='store_true',
                        help='Generate inference script instead of training')
    parser.add_argument('--no-inference', action='store_true',
                        help='Generate training script instead of inference')
    # SLURM configuration
    parser.add_argument('--time', type=str,
                        help='Wall time limit (e.g., 00-02:00:00)')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--gpus', type=int,
                        help='Number of GPUs per node')
    parser.add_argument('--memory', type=str,
                        help='Memory allocation (e.g., 48GB)')
    parser.add_argument('--partition', type=str,
                        help='SLURM partition')
    
    # Other options
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling mode')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print script without saving')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the job immediately after generation')
    
    args = parser.parse_args()
    
    # Generate the script
    generator = Generator()
    script_path = generator.generate_script(args)
    
    print(f"Generated script: {script_path}")
    
    if args.dry_run:
        print("\n--- Script Content ---")
        with open(script_path, 'r') as f:
            print(f.read())
        os.remove(script_path)  # Clean up in dry-run mode
    
    if args.submit and not args.dry_run:
        print(f"Submitting job...")
        os.system(f"sbatch {script_path}")
    else:
        print(f"To submit: sbatch {script_path}")


if __name__ == '__main__':
    main()
