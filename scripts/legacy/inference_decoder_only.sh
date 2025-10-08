#!/bin/bash
#SBATCH --job-name=lvsm-inference-1node
#SBATCH --account=aip-fsanja
#SBATCH --output=results/inference_%j.out
#SBATCH --error=results/inference_%j.err
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=l40s:2
#SBATCH --ntasks-per-node=1

echo "=== LVSM 1-Node 2-GPU Inference Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node: $SLURMD_NODENAME" 
echo "GPUs per node: 2"
echo "Total GPUs: 2"
echo "Start time: $(date)"
echo

# Load modules (consistent with debug script)
echo "Loading modules..."
module load python/3.12
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8

# Environment variables (consistent with debug script)
echo "Setting environment variables..."
export OMP_NUM_THREADS=2
export IBV_FORK_SAFE=1

# Single-node distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=2  # 1 node * 2 GPU per node
export NODE_RANK=$SLURM_PROCID

echo "Distributed setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NODE_RANK: $NODE_RANK"
echo

# Create output directories
echo "Creating output directories..."
mkdir -p results/evaluation/test
mkdir -p results/logs

# Suppress libibverbs warnings
exec 3>&2
exec 2> >(grep -v 'libibverbs: Warning' >&3)

echo "Starting inference..."
echo "=========================================="

# Run inference on 1 node with 2 GPUs (2 total processes)
srun uv run python -W ignore::FutureWarning -W ignore::UserWarning -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    inference.py \
    --config "configs/LVSM_scene_decoder_only.yaml" \
    training.dataset_path="/home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt" \
    training.checkpoint_dir="./ckpts/scene_decoder_only_256.pt" \
    training.batch_size_per_gpu=4 \
    training.target_has_input=false \
    training.square_crop=true \
    training.num_input_views=2 \
    training.num_target_views=3 \
    inference.if_inference=true \
    inference.compute_metrics=true \
    inference.render_video=false \
    inference_out_dir="./results/evaluation/test"

# Capture exit code
INFERENCE_EXIT_CODE=$?

# Restore stderr
exec 2>&3
exec 3>&-

echo
echo "=========================================="
echo "=== Inference Job Complete ==="
echo "End time: $(date)"
echo "Exit code: $INFERENCE_EXIT_CODE"

# Show results
if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Inference completed without errors"
    echo
    echo "Results in results/evaluation/test/:"
    ls -la results/evaluation/test/ 2>/dev/null || echo "Output directory not found"
    echo
    echo "Resource usage summary:"
    echo "  Nodes used: $SLURM_NNODES"
    echo "  Total GPUs: $WORLD_SIZE"
    echo "  Job duration: $(( $(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s) )) seconds"
else
    echo "❌ ERROR: Inference failed with exit code $INFERENCE_EXIT_CODE"
    echo "Check the error log: results/inference_${SLURM_JOB_ID}.err"
fi

echo
echo "Log files:"
echo "  Output: results/inference_${SLURM_JOB_ID}.out" 
echo "  Error:  results/inference_${SLURM_JOB_ID}.err"

exit $INFERENCE_EXIT_CODE
