# Running Refusal Direction Extraction on HPC with SLURM

This guide explains how to run the refusal direction extraction pipeline on an HPC cluster using SLURM and Singularity containers.

## Overview

The workflow uses:
- **Singularity/Apptainer**: Provides CUDA environment and system libraries
- **uv**: Manages Python packages in a virtual environment on `/scratch`
- **SLURM**: Schedules and manages GPU jobs

This approach is **recommended by HPC staff** because:
1. The container provides a stable base system (survives OS upgrades)
2. Python packages live on the shared filesystem (not in the container)
3. No writable overlay files needed (avoids Conda reliability issues)

## Why uv + Singularity Works Great

**Does uv conflict with Singularity's file count reduction goal?**

No! Here's why:
- **Singularity's file count reduction**: Applies to the container *image* itself (the `.sif` files use SquashFS compression)
- **Your Python packages**: Live in a virtual environment on `/scratch`, *outside* the container
- **How it works**: The container provides CUDA + system libraries, while uv manages your Python dependencies on the mounted filesystem

Benefits of this approach:
- ✅ Fast package installation (uv is very fast)
- ✅ Reproducible environments (uv.lock)
- ✅ No writable overlay issues (unlike Conda)
- ✅ Easy to update packages without rebuilding containers

## Setup Instructions

### Step 1: Copy Project to HPC

On your **local machine**, sync your code to the HPC cluster:

```bash
# From your local machine
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    /Users/andyhan/Documents/school-work/work-25-q4f/pavel-research/refusal_direction_tfl \
    ah7660@torch-login.hpc.nyu.edu:~/
```

### Step 2: SSH to HPC and Set Up Environment

```bash
# SSH to the cluster
ssh ah7660@torch-login.hpc.nyu.edu

# Navigate to your project
cd ~/refusal_direction_tfl

# Make scripts executable
chmod +x hpc/setup_env.sh

# Edit the script to set correct paths
nano hpc/setup_env.sh  # Update PROJECT_DIR if needed
```

### Step 3: Run Environment Setup

This installs uv and creates a virtual environment in `/scratch` with all dependencies:

```bash
# Run the setup script
bash hpc/setup_env.sh
```

This will:
1. Install `uv` to `~/.local/bin`
2. Create a virtual environment in `/scratch/$USER/venvs/refusal-dir-env`
3. Install PyTorch with CUDA 12.1 support
4. Install all project dependencies from `pyproject.toml`

**Note**: The setup runs on a login node (no GPU needed). It takes ~5-10 minutes.

### Step 4: Verify Installation (Optional)

Test that everything works:

```bash
# Activate the environment
source /scratch/$USER/venvs/refusal-dir-env/bin/activate

# Check installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Deactivate
deactivate
```

## Running Jobs

### Basic Usage

Edit the SLURM script to set your paths:

```bash
nano hpc/run_pipeline.slurm
```

Update these variables at the top:
- `PROJECT_DIR`: Path to your project (default: `$HOME/refusal_direction_tfl`)
- `VENV_DIR`: Path to your virtual environment (default: `/scratch/$USER/venvs/refusal-dir-env`)
- `OUTPUT_DIR`: Where to save results (default: `/scratch/$USER/refusal_outputs`)

### Submit a Job

```bash
# Create logs directory
mkdir -p logs

# Submit job with model path
sbatch hpc/run_pipeline.slurm --model-path meta-llama/Llama-3.2-8B-Instruct

# With additional options
sbatch hpc/run_pipeline.slurm \
    --model-path meta-llama/Llama-3.2-8B-Instruct \
    --perturbation-types operand_swap number_substitution

# Generation only (skip selection/evaluation)
sbatch hpc/run_pipeline.slurm \
    --model-path meta-llama/Llama-3.2-8B-Instruct \
    --generation-only
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/refusal-dir_<JOB_ID>.out

# View errors
tail -f logs/refusal-dir_<JOB_ID>.err

# Cancel a job
scancel <JOB_ID>
```

### Check Results

```bash
# Results are saved to /scratch/$USER/refusal_outputs/
ls -lh /scratch/$USER/refusal_outputs/

# View artifacts
tree /scratch/$USER/refusal_outputs/<model_name>/
```

## Resource Configuration

The SLURM script defaults are:
- **GPUs**: 1 (increase for larger models)
- **Memory**: 64GB (increase for larger models like 70B)
- **Time**: 24 hours
- **CPUs**: 8 cores

To modify resources for larger models:

```bash
# Edit the SLURM directives in run_pipeline.slurm
nano hpc/run_pipeline.slurm
```

Example for large models (70B):
```bash
#SBATCH --gres=gpu:a100:2           # 2 A100 GPUs
#SBATCH --mem=256G                  # 256GB RAM
#SBATCH --cpus-per-task=16          # 16 CPU cores
```

## Container Selection

The script uses `cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif` by default.

Available CUDA containers on your HPC:
- `cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif` (default)
- `cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif` (newer)
- `cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif` (older Ubuntu)

To change containers:
```bash
# Edit CONTAINER_IMAGE in run_pipeline.slurm
CONTAINER_IMAGE="/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
```

## Troubleshooting

### Issue: "CUDA not available" in container

**Solution**: Make sure you're using `sbatch` (not `srun` directly) and that the partition has GPUs:
```bash
# Check available partitions
sinfo

# Specify GPU partition explicitly
#SBATCH --partition=gpu-a100  # or whatever your cluster calls it
```

### Issue: "Module not found" errors

**Solution**: Verify the virtual environment is activated:
```bash
# In the SLURM script, check this line:
source $VENV_DIR/bin/activate

# Or test manually:
singularity exec --nv <container> bash -c "source /scratch/$USER/venvs/refusal-dir-env/bin/activate && python -c 'import torch'"
```

### Issue: Out of memory

**Solution**: Increase memory in SLURM script:
```bash
#SBATCH --mem=128G  # or more
```

Or use gradient checkpointing / smaller batch sizes in the code.

### Issue: Job killed unexpectedly

**Solution**: Check time limits:
```bash
# Increase time limit
#SBATCH --time=48:00:00  # 48 hours
```

### Issue: Can't write to output directory

**Solution**: Make sure directories exist and are on `/scratch`:
```bash
mkdir -p /scratch/$USER/refusal_outputs
```

## File Organization

Recommended directory structure on HPC:

```
$HOME/
├── refusal_direction_tfl/          # Your code (small files, backed up)
│   ├── hpc/
│   │   ├── setup_env.sh
│   │   ├── run_pipeline.slurm
│   │   └── README_HPC.md
│   ├── pipeline/
│   ├── dataset/
│   └── pyproject.toml

/scratch/$USER/
├── venvs/
│   └── refusal-dir-env/            # Python virtual environment
├── hf_cache/                       # HuggingFace model/dataset cache
│   ├── transformers/
│   └── datasets/
└── refusal_outputs/                # Pipeline results
    ├── Llama-3.2-8B-Instruct/
    │   ├── generate_directions/
    │   ├── select_direction/
    │   ├── completions/
    │   └── direction.pt
    └── ...
```

**Why this structure?**
- `$HOME`: Code and scripts (backed up, but limited quota)
- `/scratch`: Large files like models, datasets, outputs (fast, no backups)

## Updating Dependencies

To add new packages:

```bash
# On login node, activate environment
source /scratch/$USER/venvs/refusal-dir-env/bin/activate

# Install new package
uv pip install <package-name>

# Or update pyproject.toml and reinstall
cd ~/refusal_direction_tfl
uv pip install -e .
```

## Advanced: Interactive GPU Session

For debugging, request an interactive GPU session:

```bash
# Request interactive GPU node
srun --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash

# Once on GPU node, run container
singularity shell --nv /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

# Inside container, activate environment
source /scratch/$USER/venvs/refusal-dir-env/bin/activate

# Run Python interactively
python
```

## Questions?

- HPC documentation: Check your cluster's wiki/docs
- SLURM commands: `man sbatch`, `man squeue`, `man scancel`
- Singularity docs: https://apptainer.org/docs/user/main/

## Summary

1. ✅ Use Singularity containers for CUDA + system libraries
2. ✅ Use uv to manage Python packages in `/scratch`
3. ✅ Submit jobs with `sbatch hpc/run_pipeline.slurm --model-path <path>`
4. ✅ Monitor with `squeue` and `tail -f logs/*.out`
5. ✅ Results saved to `/scratch/$USER/refusal_outputs/`
