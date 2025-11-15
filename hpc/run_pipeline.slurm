#!/bin/bash
#SBATCH -A torch_pr_230_tandon_priority         # Update this for Torch
#SBATCH --job-name=refusal-dir
#SBATCH --output=logs/refusal_%j.out
#SBATCH --error=logs/refusal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ah7660@nyu.edu

# =============================================================================
# SLURM Job Script for Refusal Direction Extraction - NYU Torch Cluster
# =============================================================================
# This script is configured for the NEW Torch cluster (not Greene).
# It uses uv + virtual environments instead of Conda + overlays.
#
# Usage:
#   sbatch hpc/run_pipeline.slurm [options]
#
# Default configuration:
#   - Model: Qwen/Qwen3-8B
#   - Batch size: 16
#   - Training samples: 32
#   - Perturbation types: number_substitution, operator_replace, computation_plusminus
#
# Examples:
#   sbatch hpc/run_pipeline.slurm  # Run with defaults
#   sbatch hpc/run_pipeline.slurm --model-path meta-llama/Llama-3.2-8B-Instruct
#   sbatch hpc/run_pipeline.slurm --batch-size 32 --n-train 64
# =============================================================================

set -e  # Exit on error

# ========================
# Configuration Variables
# ========================

# Paths - UPDATE THESE if needed
PROJECT_DIR="/scratch/$USER/refusal_direction_tfl"  # Project in scratch
VENV_DIR="/scratch/$USER/venvs/refusal-dir-env"     # Virtual env in scratch
CACHE_DIR="/scratch/$USER/hf_cache"                 # HF cache in scratch
SUBMIT_DIR="$SLURM_SUBMIT_DIR"

# Singularity/Apptainer settings for Torch
SINGULARITY_BIN="/share/apps/apptainer/bin/singularity"
# Choose CUDA container - adjust based on your needs
CONTAINER_IMAGE="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
# Alternative: CONTAINER_IMAGE="/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"

# Set required Apptainer bind paths for Torch
export APPTAINER_BINDPATH="/scratch,/state/partition1,/mnt,/share/apps"

# HuggingFace cache settings
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"

# ======================
# Job Information
# ======================
echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Submitted from: $SUBMIT_DIR"
echo "=================================================="

# ======================
# Environment Setup
# ======================
echo "=== Environment Setup ==="
echo "Container: $CONTAINER_IMAGE"
echo "Project Directory: $PROJECT_DIR"
echo "Virtual Environment: $VENV_DIR"
echo "Cache Directory: $CACHE_DIR"
echo ""

# Create necessary directories
mkdir -p "$CACHE_DIR"
mkdir -p "$SUBMIT_DIR/logs"

# Verify paths exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    echo "Run: bash hpc/setup_env.sh"
    exit 1
fi

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container image not found: $CONTAINER_IMAGE"
    exit 1
fi

# ======================
# GPU Information
# ======================
echo "=== GPU Information ==="
$SINGULARITY_BIN exec --nv "$CONTAINER_IMAGE" nvidia-smi
echo ""

# ======================
# Default Configuration
# ======================
# Set default values for pipeline arguments
DEFAULT_MODEL_PATH="Qwen/Qwen3-8B"
DEFAULT_BATCH_SIZE="16"
DEFAULT_N_TRAIN="32"
DEFAULT_PERTURBATION_TYPES="number_substitution operator_replace computation_plusminus"

# ======================
# Parse Arguments
# ======================
# Get all arguments passed to sbatch after the script name
SCRIPT_ARGS="${@}"

# If no arguments provided, use defaults
if [ -z "$SCRIPT_ARGS" ]; then
    echo "No arguments provided, using defaults..."
    SCRIPT_ARGS="--model-path $DEFAULT_MODEL_PATH --batch-size $DEFAULT_BATCH_SIZE --n-train $DEFAULT_N_TRAIN --perturbation-types $DEFAULT_PERTURBATION_TYPES"
fi

echo "=== Pipeline Arguments ==="
echo "Arguments: $SCRIPT_ARGS"
echo ""

# ======================
# Run Pipeline
# ======================
echo "=== Running Pipeline ==="

# Run the pipeline inside the Singularity container
$SINGULARITY_BIN exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$VENV_DIR:$VENV_DIR" \
    --bind "$CACHE_DIR:$CACHE_DIR" \
    --bind "/scratch:/scratch" \
    --pwd "$PROJECT_DIR" \
    "$CONTAINER_IMAGE" \
    bash -c "
        # Activate virtual environment
        source $VENV_DIR/bin/activate

        # Verify environment
        echo 'Checking Python environment...'
        echo \"Python: \$(python --version)\"
        echo \"PyTorch: \$(python -c 'import torch; print(torch.__version__)')\"
        echo \"CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')\"
        echo \"GPU count: \$(python -c 'import torch; print(torch.cuda.device_count())')\"
        echo \"Transformers: \$(python -c 'import transformers; print(transformers.__version__)')\"
        echo ''

        # Run the pipeline
        echo 'Starting refusal direction extraction...'
        cd $PROJECT_DIR
        python -m pipeline.run_pipeline $SCRIPT_ARGS
    "

EXIT_CODE=$?

# ======================
# Job Summary
# ======================
echo ""
echo "=== Job Summary ==="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "Results Directory: Check $PROJECT_DIR/pipeline/runs/"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS ✓"
else
    echo "Status: FAILED ✗"
    echo "Check error log: logs/refusal_${SLURM_JOB_ID}.err"
fi

echo "=================================================="

# Print final GPU status
echo "=== Final GPU Status ==="
nvidia-smi

exit $EXIT_CODE
