#!/bin/bash
#SBATCH --job-name=refusal_direction
#SBATCH --output=logs/refusal_%j.out
#SBATCH --error=logs/refusal_%j.err
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

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
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
RESULTS_DIR=$SCRATCH_DIR/results

# Create directories if they don't exist
mkdir -p $PROJECT_DIR
mkdir -p $VENV_DIR
mkdir -p $CACHE_DIR
mkdir -p $RESULTS_DIR
mkdir -p logs

echo "Scratch directory: $SCRATCH_DIR"
echo "Results directory: $RESULTS_DIR"

# Copy code to scratch if this is first run
if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo "Copying code to scratch..."
    rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        $SLURM_SUBMIT_DIR/ $PROJECT_DIR/
else
    echo "Code already exists in scratch, syncing updates..."
    rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        --exclude='pipeline/runs' --exclude='dataset' \
        $SLURM_SUBMIT_DIR/ $PROJECT_DIR/
fi

cd $PROJECT_DIR

# Set up virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set HuggingFace cache directory (to avoid filling up home directory)
export HF_HOME=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR

# Set Together AI API key if available
# Note: Set this in your environment before submitting the job
# export TOGETHER_API_KEY=your_key_here

# Parse model path from command line or use default
MODEL_PATH=${1:-"Qwen/Qwen3-8B"}

echo "=================================================="
echo "Running pipeline with model: $MODEL_PATH"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================================================="

# Run the pipeline
python3 -m pipeline.run_pipeline \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --eval-datasets harmbench_val,strongreject \
    --cache-layers 8

# Copy results back to submit directory if desired
# rsync -av pipeline/runs/ $SLURM_SUBMIT_DIR/pipeline/runs/

echo "=================================================="
echo "Job completed at: $(date)"
echo "Results saved to: $PROJECT_DIR/pipeline/runs/"
echo "=================================================="

# Print GPU usage summary
nvidia-smi
