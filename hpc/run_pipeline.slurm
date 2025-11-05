#!/bin/bash
#SBATCH -A pr_130_tandon_advanced
#SBATCH -p tandon_a100_2
#SBATCH --job-name=refusal_direction
#SBATCH --output=logs/refusal_%j.out
#SBATCH --error=logs/refusal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ah7660@nyu.edu

set -e

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------
SCRATCH_DIR=/scratch/ah7660/refusal_direction_tfl
DEFAULT_PROJECT_DIR=$SCRATCH_DIR/code
CACHE_DIR=$SCRATCH_DIR/cache
RESULTS_DIR=$SCRATCH_DIR/results
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# Determine project directory location.
if [ -d "$SCRATCH_DIR/.git" ]; then
    PROJECT_DIR=$SCRATCH_DIR
    CODE_SYNC_REQUIRED=false
elif [ -d "$DEFAULT_PROJECT_DIR/.git" ]; then
    PROJECT_DIR=$DEFAULT_PROJECT_DIR
    CODE_SYNC_REQUIRED=false
else
    PROJECT_DIR=$DEFAULT_PROJECT_DIR
    CODE_SYNC_REQUIRED=true
fi

echo "Project directory: $PROJECT_DIR"
echo "Submit directory:  $SUBMIT_DIR"

# Ensure directories exist
mkdir -p "$PROJECT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SUBMIT_DIR/logs"

# Sync code if necessary (typically when submitting from home directory).
if [ "$CODE_SYNC_REQUIRED" = true ]; then
    echo "Syncing project code to scratch..."
    rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        --exclude='*.pyc' --exclude='.venv' \
        "$SUBMIT_DIR/" "$PROJECT_DIR/"
else
    echo "Project already resident on scratch; skipping rsync."
fi

cd "$PROJECT_DIR"

# -----------------------------------------------------------------------------
# Singularity configuration
# -----------------------------------------------------------------------------
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
SINGULARITY_OVERLAY_INPUT=${SINGULARITY_OVERLAY:-/scratch/ah7660/overlay-25GB-500K.ext3}
ENV_WRAPPER=${ENV_WRAPPER:-/ext3/env.sh}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-refusal_direction}
PIPELINE_ARGS=${PIPELINE_ARGS:-"--generation-only"}

resolve_overlay_path() {
    local input_path="$1"
    local overlay_path=""
    local overlay_archive=""

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

    echo "ERROR: Overlay file not found at $overlay_path (or $overlay_archive)." >&2
    echo "Run hpc/setup_greene.sh to provision the overlay before submitting jobs." >&2
    return 1
}

OVERLAY_PATH=$(resolve_overlay_path "$SINGULARITY_OVERLAY_INPUT") || exit 1

if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity image $SINGULARITY_IMAGE not found." >&2
    exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: 'singularity' command not available. Load the appropriate module (e.g., module load singularity) in your job submission script." >&2
    exit 1
fi

echo "Singularity image:  $SINGULARITY_IMAGE"
echo "Singularity overlay: $OVERLAY_PATH"
echo "Conda env:          $CONDA_ENV_NAME"
echo "Env wrapper script: $ENV_WRAPPER"
echo "Pipeline args:      $PIPELINE_ARGS"

if ! singularity exec --overlay "${OVERLAY_PATH}:ro" "${SINGULARITY_IMAGE}" /bin/bash -c "[ -f ${ENV_WRAPPER} ]"; then
    echo "ERROR: Environment wrapper ${ENV_WRAPPER} not found inside overlay. Run hpc/setup_greene.sh before submitting jobs." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Model argument handling
# -----------------------------------------------------------------------------
MODEL_PATH=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}

echo "Target model:       $MODEL_PATH"

# -----------------------------------------------------------------------------
# Display environment info
# -----------------------------------------------------------------------------
echo "=================================================="
echo "Environment diagnostics"
singularity exec --nv --overlay "${OVERLAY_PATH}:ro" "${SINGULARITY_IMAGE}" /bin/bash -lc "
set -e
source ${ENV_WRAPPER}
echo \"Container Python: \$(python3 --version)\"
echo \"PyTorch: \$(python3 -c 'import torch; print(torch.__version__)')\"
echo \"CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())')\"
echo \"CUDA devices: \$(python3 -c 'import torch; print(torch.cuda.device_count())')\"
"
echo "=================================================="

# -----------------------------------------------------------------------------
# Run pipeline inside container
# -----------------------------------------------------------------------------
RUN_COMMAND="cd ${PROJECT_DIR} && \
source ${ENV_WRAPPER} && \
export HF_HOME=${CACHE_DIR} && \
export TRANSFORMERS_CACHE=${CACHE_DIR} && \
export HF_DATASETS_CACHE=${CACHE_DIR} && \
python3 -m pipeline.run_pipeline --model_path \"${MODEL_PATH}\" ${PIPELINE_ARGS}"

echo "Executing pipeline..."
singularity exec --nv --overlay "${OVERLAY_PATH}:ro" "${SINGULARITY_IMAGE}" /bin/bash -lc "$RUN_COMMAND" < /dev/null

echo "=================================================="
echo "Job completed at: $(date)"
echo "Results saved to: $PROJECT_DIR/pipeline/runs/"
echo "=================================================="

# Print GPU usage summary
nvidia-smi
