#!/bin/bash
# Setup script for Greene HPC cluster
# Run this once before submitting jobs

set -e

echo "=================================================="
echo "Setting up refusal direction pipeline on Greene"
echo "=================================================="

# Configuration
SCRATCH_DIR=/scratch/ah7660/refusal_direction_tfl
PROJECT_DIR=$SCRATCH_DIR/code
VENV_DIR=$SCRATCH_DIR/venv
CACHE_DIR=$SCRATCH_DIR/cache
DATASET_DIR=$SCRATCH_DIR/dataset

echo "Creating directory structure..."
mkdir -p $SCRATCH_DIR
mkdir -p $PROJECT_DIR
mkdir -p $VENV_DIR
mkdir -p $CACHE_DIR
mkdir -p $DATASET_DIR
mkdir -p logs

echo "Copying project files to scratch..."
rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.venv' \
    . $PROJECT_DIR/

# Copy dataset if it exists locally
if [ -d "./dataset" ]; then
    echo "Copying dataset to scratch..."
    rsync -av ./dataset/ $DATASET_DIR/
else
    echo "Warning: No local dataset directory found. Dataset will be in code directory."
fi

echo "Loading Python module..."
module purge
module load python/intel/3.8.6

echo "Creating virtual environment..."
cd $PROJECT_DIR
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies (this may take a while)..."
pip install -r requirements.txt

echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Directory structure:"
echo "  Project code: $PROJECT_DIR"
echo "  Virtual env:  $VENV_DIR"
echo "  Cache:        $CACHE_DIR"
echo "  Dataset:      $DATASET_DIR"
echo ""
echo "Next steps:"
echo "  1. (Optional) Set environment variables in your ~/.bashrc:"
echo "     export TOGETHER_API_KEY='your_key_here'"
echo "     export HF_TOKEN='your_huggingface_token'"
echo ""
echo "  2. Submit a job:"
echo "     sbatch hpc/run_pipeline.slurm"
echo ""
echo "  3. Or test interactively:"
echo "     srun --gres=gpu:1 -c 8 --mem=32GB -t 2:00:00 --pty /bin/bash"
echo "     source /scratch/ah7660/refusal_direction_tfl/venv/bin/activate"
echo "     cd /scratch/ah7660/refusal_direction_tfl/code"
echo "     python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct"
echo ""
echo "=================================================="
