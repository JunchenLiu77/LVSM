#!/bin/bash
#SBATCH --job-name=lvsm-train
#SBATCH --account=aip-fsanja
#SBATCH --output=results/train_%j.out
#SBATCH --error=results/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=l40s:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_l40s_b1

# Usage: sbatch scripts/train_bf16.sh [--profile] [--steps N]
# Examples:
#   sbatch scripts/train_bf16.sh                    # Regular training
#   sbatch scripts/train_bf16.sh --profile          # Training with profiling
#   sbatch scripts/train_bf16.sh --steps 100        # Train for 100 steps
#   sbatch scripts/train_bf16.sh --profile --steps 50  # Profile for 50 steps

# Parse arguments
PROFILE_MODE=false
TRAIN_STEPS=1000  # Default
TRAIN_SCRIPT="train.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: sbatch scripts/train_bf16.sh [--profile] [--steps N]"
            echo "Options:"
            echo "  --profile    Enable detailed profiling mode"
            echo "  --steps N    Number of training steps (default: 1000)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        --profile)
            PROFILE_MODE=true
            TRAIN_SCRIPT="train_profiled.py"
            shift
            ;;
        --steps)
            TRAIN_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== LVSM Training with BF16 AMP ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 4x L40s"
echo "Profile mode: $PROFILE_MODE"
echo "Training steps: $TRAIN_STEPS"
echo "Start time: $(date)"
echo "==========================================="

# Load modules
module load python/3.12
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8

# Environment variables
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export IBV_FORK_SAFE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# For profiling, use synchronous CUDA operations
if [ "$PROFILE_MODE" = true ]; then
    export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for more accurate profiling
    echo "Running in PROFILE mode with detailed timing breakdown"
else
    echo "Running in NORMAL training mode"
fi

echo "Creating output directories..."
mkdir -p results/logs
mkdir -p experiments/checkpoints/lvsm_bf16

# Suppress libibverbs warnings
exec 3>&2
exec 2> >(grep -v 'libibverbs: Warning' >&3)

echo
echo "Starting training..."
echo

uv run torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --config configs/LVSM_scene_decoder_only.yaml \
    training.dataset_path="/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/train/full_list.txt" \
    training.resume_ckpt="./ckpts/scene_decoder_only_256.pt" \
    training.reset_training_state=true \
    training.checkpoint_dir="./experiments/checkpoints/lvsm_bf16" \
    training.batch_size_per_gpu=8 \
    training.target_has_input=false \
    training.num_views=8 \
    training.square_crop=true \
    training.num_input_views=2 \
    training.num_target_views=6 \
    training.train_steps=$TRAIN_STEPS \
    training.checkpoint_every=500 \
    training.wandb_exp_name="LVSM_bf16_$(date +%Y%m%d_%H%M%S)" \
    training.wandb_log_every=10 \
    training.print_every=10 \
    training.vis_every=500 \
    training.grad_clip_norm=1.0 \
    training.lr=0.0004 \
    training.warmup=100 \
    training.use_amp=true \
    training.amp_dtype=bf16

# Restore stderr
exec 2>&3
exec 3>&-

echo
echo "==========================================="
echo "Training completed at: $(date)"

if [ "$PROFILE_MODE" = true ]; then
    echo
    echo "Profiling Summary:"
    echo "  Check the output above for:"
    echo "  1. Per-iteration timing breakdown"
    echo "  2. Component-wise profiling summary"
    echo "  3. Bottleneck analysis"
fi

echo "==========================================="

exit 0
