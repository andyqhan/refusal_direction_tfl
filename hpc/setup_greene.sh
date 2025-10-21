#!/bin/bash
# Setup script for Greene HPC cluster
# Run this once before submitting jobs

set -e

echo "=================================================="
echo "Setting up refusal direction pipeline on Greene"
echo "=================================================="

# Configuration
SCRATCH_DIR=/scratch/ah7660/refusal_direction_tfl
DEFAULT_PROJECT_DIR=$SCRATCH_DIR/code
VENV_DIR=$SCRATCH_DIR/venv
CACHE_DIR=$SCRATCH_DIR/cache
DATASET_DIR=$SCRATCH_DIR/dataset
CURRENT_DIR=$(pwd)

# Determine where the project should live on scratch. If the repository is
# already under $SCRATCH_DIR, keep it in place instead of nesting another copy.
if [ "$CURRENT_DIR" = "$SCRATCH_DIR" ]; then
    PROJECT_DIR=$SCRATCH_DIR
    CODE_SYNC_REQUIRED=false
    echo "Detected repository already in $SCRATCH_DIR; skipping code rsync."
else
    PROJECT_DIR=$DEFAULT_PROJECT_DIR
    CODE_SYNC_REQUIRED=true
fi

echo "Creating directory structure..."
mkdir -p $SCRATCH_DIR
mkdir -p $PROJECT_DIR
mkdir -p $VENV_DIR
mkdir -p $CACHE_DIR
mkdir -p $DATASET_DIR
mkdir -p logs

if [ "$CODE_SYNC_REQUIRED" = true ]; then
    echo "Copying project files to scratch..."
    rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        --exclude='*.pyc' --exclude='.venv' \
        . $PROJECT_DIR/
else
    echo "Project files already present on scratch; rsync skipped."
fi

# Copy dataset if it exists locally
if [ "$CODE_SYNC_REQUIRED" = true ] && [ -d "./dataset" ]; then
    echo "Copying dataset to scratch..."
    rsync -av ./dataset/ $DATASET_DIR/
elif [ ! -d "$PROJECT_DIR/dataset" ]; then
    echo "Warning: No local dataset directory found. Dataset will be in code directory."
fi

echo "Loading Python module..."
module purge

# Prefer newer Python modules for compatibility with recent dependencies.
# Users can override by exporting PYTHON_MODULE before running this script.
PYTHON_MODULE_CANDIDATES=(
    "${PYTHON_MODULE}"
    "python/intel/3.11.5"
    "python/intel/3.11.0"
    "python/intel/3.10.12"
    "python/intel/3.10.8"
    "python/intel/3.10.4"
    "python/3.11.9"
    "python/3.11.6"
    "python/3.10.11"
)

SELECTED_PY_MODULE=""
for candidate in "${PYTHON_MODULE_CANDIDATES[@]}"; do
    # Skip empty entries (e.g., if PYTHON_MODULE is unset)
    if [ -z "$candidate" ]; then
        continue
    fi

    if module load "$candidate" >/dev/null 2>&1; then
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            SELECTED_PY_MODULE=$candidate
            PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
            echo "Loaded Python module: $SELECTED_PY_MODULE (Python $PY_VERSION)"
            break
        else
            PY_SHORT_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
            echo "Module $candidate provides Python $PY_SHORT_VERSION; requires >= 3.10. Skipping."
            module unload "$candidate" >/dev/null 2>&1 || true
        fi
    else
        echo "Module $candidate is not available on this cluster. Skipping."
    fi
done

if [ -z "$SELECTED_PY_MODULE" ]; then
    echo "ERROR: Could not load a Python module with version >= 3.10."
    echo "Set PYTHON_MODULE to an appropriate module (e.g., python/intel/3.11.5) and re-run."
    exit 1
fi

echo "Creating virtual environment..."
cd $PROJECT_DIR
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing dependencies (this may take a while)..."
python3 -m pip install -r requirements.txt

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
