#!/bin/bash
# setup_env.sh - Set up Python environment with uv on HPC
# Run this once on a login node to create your virtual environment

set -e  # Exit on error

# Configuration
PROJECT_DIR="/scratch/$USER/refusal_direction_tfl"  # Project in scratch (more space)
VENV_DIR="/scratch/$USER/venvs/refusal-dir-env"      # Virtual env in scratch
UV_CACHE_DIR="/scratch/$USER/.uv_cache"              # Cache in scratch (avoids home quota)

echo "=== Setting up Python environment with uv ==="
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $VENV_DIR"
echo "UV cache directory: $UV_CACHE_DIR"
echo ""

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed: $(which uv)"
fi

# 2. Set uv cache location to scratch (avoid home directory quota)
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_LINK_MODE=copy  # Suppress hardlink warning on cross-filesystem installs

echo "Set UV_CACHE_DIR=$UV_CACHE_DIR"
echo ""

# 3. Create directories
mkdir -p "$PROJECT_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
mkdir -p "$UV_CACHE_DIR"

# 4. Create virtual environment in scratch space
echo "Creating virtual environment in $VENV_DIR..."
uv venv "$VENV_DIR" --python 3.11

# 5. Activate the environment
source "$VENV_DIR/bin/activate"

# 6. Install PyTorch with CUDA support first (specify CUDA version)
echo "Installing PyTorch with CUDA 12.1 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Install project dependencies
echo "Installing project dependencies from pyproject.toml..."
cd "$PROJECT_DIR"
uv pip install -e .

# Optional: Install HPC-specific dependencies (vllm, etc.)
# Uncomment if needed:
# echo "Installing HPC-specific dependencies..."
# uv pip install -e ".[hpc]"

echo ""
echo "=== Setup complete! ==="
echo "Virtual environment created at: $VENV_DIR"
echo ""
echo "To activate this environment in the future:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
