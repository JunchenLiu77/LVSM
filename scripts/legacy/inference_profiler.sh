#!/bin/bash
#SBATCH --job-name=lvsm-profiler
#SBATCH --account=aip-fsanja
#SBATCH --output=results/inference_profiling_%j.out
#SBATCH --error=results/inference_profiling_%j.err
#SBATCH --time=00-00:10:00
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=l40s:4
#SBATCH --ntasks-per-node=1

echo "=========================================="
echo "LVSM INFERENCE PROFILER"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Purpose: Comprehensive timing analysis of inference pipeline"
echo "Start time: $(date)"

# Create test dataset (50 samples for comprehensive profiling)
head -50 ~/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt > /tmp/profiling_test.txt

# Load modules
module load python/3.12
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8

# Environment setup
export OMP_NUM_THREADS=2
export IBV_FORK_SAFE=1

# Distributed setup
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

echo "Running comprehensive inference profiling..."
echo

mkdir -p results/evaluation/profiling

echo "Starting timing analysis..."
srun --exclusive uv run python -W ignore::FutureWarning -W ignore::UserWarning -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    --rdzv_backend=c10d \
    deep_profile.py \
    --config "configs/LVSM_scene_decoder_only.yaml" \
    training.dataset_path="/tmp/profiling_test.txt" \
    training.checkpoint_path="./ckpts/scene_decoder_only_256.pt" \
    training.batch_size_per_gpu=4 \
    training.num_workers=2 \
    training.prefetch_factor=2 \
    training.target_has_input=false \
    training.num_views=5 \
    training.square_crop=true \
    training.num_input_views=2 \
    training.num_target_views=3 \
    inference.if_inference=true \
    inference.compute_metrics=true \
    inference.render_video=false \
    inference_out_dir="./results/evaluation/profiling"

EXIT_CODE=$?

echo
echo "=========================================="
echo "INFERENCE PROFILING COMPLETE" 
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Inference profiling completed"
    echo
    echo "Profiling results show detailed breakdown of:"
    echo "- Overall inference pipeline timing"
    echo "- Post-processing component analysis" 
    echo "- Bottleneck identification and ranking"
    echo "- Device usage verification for metrics"
else
    echo "❌ ERROR: Inference profiling failed"
fi
