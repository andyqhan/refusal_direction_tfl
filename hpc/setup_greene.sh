#!/bin/bash
# Setup script for Greene HPC cluster using Singularity environment
# Run this once (or after requirements change) to prepare the container-based venv.

set -e

echo "=================================================="
echo "Setting up refusal direction pipeline on Greene"
echo "=================================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRATCH_DIR=/scratch/ah7660/refusal_direction_tfl
DEFAULT_PROJECT_DIR=$SCRATCH_DIR/code
CACHE_DIR=$SCRATCH_DIR/cache
DATASET_DIR=$SCRATCH_DIR/dataset
CURRENT_DIR=$(pwd)

# Detect whether the repository already lives somewhere under SCRATCH_DIR.
if [[ "$CURRENT_DIR" == "$SCRATCH_DIR" ]] || [[ "$CURRENT_DIR" == "$SCRATCH_DIR/"* ]]; then
    PROJECT_DIR=$SCRATCH_DIR
    CODE_SYNC_REQUIRED=false
    echo "Detected repository under $SCRATCH_DIR; skipping code rsync."
else
    PROJECT_DIR=$DEFAULT_PROJECT_DIR
    CODE_SYNC_REQUIRED=true
fi

# Singularity configuration (can be overridden via env vars when invoking script)
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
SINGULARITY_OVERLAY_INPUT=${SINGULARITY_OVERLAY:-/scratch/ah7660/overlay-25GB-500K.ext3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-refusal_direction}
CONTAINER_PIP_CACHE=${CONTAINER_PIP_CACHE:-/ext3/pip-cache}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

resolve_overlay_path() {
    local input_path="$1"
    local overlay_path=""
    local overlay_archive=""

    if [[ -z "$input_path" ]]; then
        echo "ERROR: No overlay path provided." >&2
        return 1
    fi

    # Normalize: if user passed .gz, strip for the working path.
    if [[ "$input_path" == *.gz ]]; then
        overlay_archive="$input_path"
        overlay_path="${input_path%.gz}"
    else
        overlay_path="$input_path"
        overlay_archive="${input_path}.gz"
    fi

    if [ -f "$overlay_path" ]; then
        echo "$overlay_path"
        return 0
    fi

    if [ -f "$overlay_archive" ]; then
        echo "Decompressing overlay archive $overlay_archive ..." >&2
        gunzip -k "$overlay_archive"
        echo "$overlay_path"
        return 0
    fi

    # Overlay not found; copy default 15GB overlay.
    local overlay_dir
    overlay_dir=$(dirname "$overlay_path")
    mkdir -p "$overlay_dir"
    local default_archive=/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz
    if [ ! -f "$default_archive" ]; then
        echo "ERROR: Default overlay archive $default_archive not found. Please specify SINGULARITY_OVERLAY manually." >&2
        return 1
    fi
    echo "Overlay not found. Copying default overlay to $overlay_archive ..." >&2
    cp "$default_archive" "$overlay_archive"
    echo "Decompressing overlay archive $overlay_archive ..." >&2
    gunzip -k "$overlay_archive"
    echo "${overlay_archive%.gz}"
    return 0
}

# -----------------------------------------------------------------------------
# Prepare directories and optional code sync
# -----------------------------------------------------------------------------

if [ -f "$HOME/.bashrc" ] && grep -q "/share/apps/anaconda3" "$HOME/.bashrc"; then
    echo "Warning: Detected conda initialization for /share/apps/anaconda3 in ~/.bashrc."
    echo "         Comment out that block as described in hpc_docs/07.5-singularity.md before continuing."
fi

echo "Creating directory structure..."
mkdir -p "$SCRATCH_DIR"
mkdir -p "$PROJECT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$DATASET_DIR"
mkdir -p logs

if [ "$CODE_SYNC_REQUIRED" = true ]; then
    echo "Copying project files to scratch..."
    rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        --exclude='*.pyc' --exclude='.venv' \
        ./ "$PROJECT_DIR/"
else
    echo "Project files already present on scratch; rsync skipped."
fi

# Copy dataset only when syncing from home to scratch and dataset exists locally.
if [ "$CODE_SYNC_REQUIRED" = true ] && [ -d "./dataset" ]; then
    echo "Copying dataset to scratch..."
    rsync -av ./dataset/ "$DATASET_DIR/"
elif [ ! -d "$PROJECT_DIR/dataset" ]; then
    echo "Warning: No dataset directory detected. The pipeline will use the code copy."
fi

# -----------------------------------------------------------------------------
# Resolve overlay and verify Singularity
# -----------------------------------------------------------------------------

OVERLAY_PATH=$(resolve_overlay_path "$SINGULARITY_OVERLAY_INPUT")
if [ $? -ne 0 ]; then
    echo "Failed to resolve Singularity overlay. Aborting."
    exit 1
fi
echo "Using overlay: $OVERLAY_PATH"

if [ ! -x "$SINGULARITY_IMAGE" ] && [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity image $SINGULARITY_IMAGE not found." >&2
    exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: 'singularity' command not found. Load the appropriate module (e.g., 'module load singularity') and rerun." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Install Python dependencies inside the container overlay
# -----------------------------------------------------------------------------

# Detect uv location on host system
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
if [ ! -x "$UV_PATH" ]; then
    echo "WARNING: uv not found at $UV_PATH. Trying common locations..."
    for loc in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/usr/local/bin/uv"; do
        if [ -x "$loc" ]; then
            UV_PATH="$loc"
            break
        fi
    done
fi

if [ ! -x "$UV_PATH" ]; then
    echo "ERROR: uv not found. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Found uv at: $UV_PATH"
echo "Installing Python environment inside Singularity overlay..."
singularity exec \
    --overlay "${OVERLAY_PATH}:rw" \
    --env PROJECT_DIR="$PROJECT_DIR" \
    --env CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    --env CONTAINER_PIP_CACHE="$CONTAINER_PIP_CACHE" \
    --env UV_PATH="$UV_PATH" \
    "${SINGULARITY_IMAGE}" /bin/bash <<'EOF'
set -e
MINIFORGE_DIR=/ext3/miniforge3
ENV_WRAPPER=/ext3/env.sh
PIP_CACHE_DIR=${CONTAINER_PIP_CACHE:-/ext3/pip-cache}
INSTALLER=/tmp/Miniforge3-Linux-x86_64.sh

unset -f which 2>/dev/null || true

if [ ! -d "$MINIFORGE_DIR" ]; then
    echo "Miniforge not found; installing to $MINIFORGE_DIR ..."
    wget --no-check-certificate -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
    rm -f "$INSTALLER"
fi

if [ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Miniforge installation failed (missing conda.sh)." >&2
    exit 1
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

if conda config --show channels | grep -q '^- defaults$'; then
    conda config --remove channels defaults || true
fi

conda config --set channel_priority flexible

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Creating conda environment '$CONDA_ENV_NAME' ..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.10
fi

cat > "$ENV_WRAPPER" <<EOW
#!/bin/bash
unset -f which 2>/dev/null || true
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:\$PATH
conda activate $CONDA_ENV_NAME
EOW
chmod +x "$ENV_WRAPPER"

source "$ENV_WRAPPER"

conda update -n base conda -y
conda clean --all --yes

mkdir -p "$PIP_CACHE_DIR"
export PIP_CACHE_DIR

# Use uv to install dependencies from pyproject.toml
# uv automatically uses uv.lock for reproducible installations
# Install with [hpc] extras for GPU-specific packages (vllm, xformers, etc.)
# UV_PATH is passed from the host system
"$UV_PATH" pip install -e "$PROJECT_DIR[hpc]"
EOF

echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Directory structure:"
echo "  Project code: $PROJECT_DIR"
echo "  Cache dir:    $CACHE_DIR"
echo "  Dataset dir:  $DATASET_DIR"
echo ""
echo "Singularity configuration:"
echo "  Image:        $SINGULARITY_IMAGE"
echo "  Overlay:      $OVERLAY_PATH"
echo "  Container venv: $CONTAINER_VENV"
echo ""
echo "Next steps:"
echo "  1. Submit jobs with: sbatch hpc/run_pipeline.slurm <MODEL_PATH>"
echo "     (Defaults to Meta-Llama-3-8B-Instruct if omitted.)"
echo "  2. Override runtime args by exporting PIPELINE_ARGS or editing the SLURM script."
echo "  3. For interactive debugging:"
echo "       srun --gres=gpu:1 --mem=32GB -c 8 -t 2:00:00 --pty /bin/bash"
echo "       singularity exec --nv --overlay ${OVERLAY_PATH}:ro ${SINGULARITY_IMAGE} \\"
echo "           /bin/bash -lc 'source ${CONTAINER_VENV}/bin/activate && python3 -m pipeline.run_pipeline --help'"
echo ""
echo "=================================================="
