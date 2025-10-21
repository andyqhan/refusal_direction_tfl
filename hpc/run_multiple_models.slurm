#!/bin/bash
#SBATCH --array=0-4
#SBATCH --job-name=refusal_array
#SBATCH --output=logs/refusal_%A_%a.out
#SBATCH --error=logs/refusal_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ah7660@nyu.edu

# Exit on error
set -e

# Define models to run
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-2-7b-chat-hf"
    "google/gemma-2b-it"
    "Qwen/Qwen-1_8B-Chat"
    "01-ai/Yi-6B-Chat"
)

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "Array Job: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Job started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# Load required modules
module purge
module load python/intel/3.8.6

# Set up directories
SCRATCH_DIR=/scratch/ah7660/refusal_direction_tfl
PROJECT_DIR=$SCRATCH_DIR/code
VENV_DIR=$SCRATCH_DIR/venv
CACHE_DIR=$SCRATCH_DIR/cache

# Create directories if they don't exist
mkdir -p $PROJECT_DIR
mkdir -p logs

cd $PROJECT_DIR

# Activate virtual environment
source $VENV_DIR/bin/activate

# Set HuggingFace cache directory
export HF_HOME=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR

echo "=================================================="
echo "Environment Info:"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

# Run the pipeline
python3 -m pipeline.run_pipeline --model_path "$MODEL"

echo "=================================================="
echo "Job completed at: $(date)"
echo "Results saved to: $PROJECT_DIR/pipeline/runs/"
echo "=================================================="

# Print GPU usage summary
nvidia-smi
