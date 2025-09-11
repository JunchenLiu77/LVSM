#!/bin/bash
#SBATCH --job-name=lvsm-debug
#SBATCH --account=aip-fsanja
#SBATCH --output=debug/output/debug_%j.out
#SBATCH --error=debug/output/debug_%j.err
#SBATCH --time=00-00:30:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=l40s:1
#SBATCH --ntasks-per-node=1

echo "=== LVSM Single Sample Debug Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo

# Load modules (same as inference script)
module load python/3.12
module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8

# Set environment variables (from inference script)
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export IBV_FORK_SAFE=1

# Create output directory
mkdir -p debug/output

# Run the debug script
echo "Running single sample debug..."
uv run python debug/debug_single_sample.py \
    --config configs/LVSM_scene_decoder_only.yaml \
    --checkpoint ckpts/scene_decoder_only_256.pt \
    --dataset_path /home/junchen/projects/aip-fsanja/shared/datasets/re10k_new/test/full_list.txt \
    --sample_idx 0 \
    --output_dir debug/output

# Capture exit code
DEBUG_EXIT_CODE=$?

echo
echo "=== Debug Job Complete ==="
echo "End time: $(date)"
echo "Exit code: $DEBUG_EXIT_CODE"

# Show results
if [ $DEBUG_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Debug completed without errors"
    echo
    echo "Results in debug/output/:"
    ls -la debug/output/
    echo
    if [ -f "debug/output/debug_summary.json" ]; then
        echo "Debug Summary:"
        cat debug/output/debug_summary.json
    fi
else
    echo "❌ ERROR: Debug failed with exit code $DEBUG_EXIT_CODE"
    echo "Check the error log above for details"
fi

exit $DEBUG_EXIT_CODE
