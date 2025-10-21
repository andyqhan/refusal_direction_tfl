# Running Refusal Direction Pipeline on NYU Greene HPC

This directory contains scripts and instructions for running the refusal direction pipeline on NYU's Greene HPC cluster.

## Prerequisites

1. **Greene Access**: Ensure you have an active NYU HPC account and can SSH to Greene
   ```bash
   ssh ah7660@greene.hpc.nyu.edu
   ```

2. **HPC Documentation**: Review the comprehensive HPC documentation at:
   - `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/README.md`
   - Key guides: GPU usage, SLURM job submission, ML workflows

3. **API Keys** (Optional but recommended):
   - **Together AI API Key**: For LlamaGuard2 evaluation
   - **HuggingFace Token**: For accessing gated models (e.g., Llama models)

## Quick Start

### Step 1: Initial Setup on Greene

SSH to Greene and run the setup script:

```bash
# SSH to Greene
ssh ah7660@greene.hpc.nyu.edu

# Navigate to your home directory
cd ~

# Clone or transfer the repository to Greene
# Option A: If using git
git clone <repository-url> refusal_direction_tfl
cd refusal_direction_tfl

# Option B: Transfer from local machine (run from your Mac)
# rsync -av --exclude='.git' --exclude='venv' \
#     ~/Documents/school-work/work-25-q4f/pavel-research/refusal_direction_tfl/ \
#     ah7660@greene.hpc.nyu.edu:~/refusal_direction_tfl/

# Run setup script (one time only)
bash hpc/setup_greene.sh
```

This will:
- Create directory structure in `/scratch/ah7660/refusal_direction_tfl/`
- Copy code to scratch
- Create Python virtual environment
- Install all dependencies

### Step 2: Configure Environment Variables (Optional)

Add to your `~/.bashrc` on Greene:

```bash
# Add these lines to ~/.bashrc
export TOGETHER_API_KEY='your_together_api_key_here'
export HF_TOKEN='your_huggingface_token_here'
export HF_HOME=/scratch/ah7660/refusal_direction_tfl/cache
export TRANSFORMERS_CACHE=/scratch/ah7660/refusal_direction_tfl/cache
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Submit a Job

```bash
cd ~/refusal_direction_tfl

# Submit with default model (Llama-3-8B-Instruct)
sbatch hpc/run_pipeline.slurm

# Submit with specific model
sbatch hpc/run_pipeline.slurm meta-llama/Llama-2-7b-chat-hf

# Submit with Gemma
sbatch hpc/run_pipeline.slurm google/gemma-2b-it
```

### Step 4: Monitor Your Job

```bash
# Check job status
squeue -u ah7660

# View output (replace JOBID with your job ID)
tail -f logs/refusal_JOBID.out
tail -f logs/refusal_JOBID.err

# Check job efficiency after completion
seff JOBID
```

## Available Scripts

### `run_pipeline.slurm`
Main SLURM job script for running the refusal direction pipeline.

**Resource allocation:**
- 1 GPU (any type - scheduler will assign based on availability)
- 8 CPU cores
- 64 GB RAM
- 24 hour time limit

**Usage:**
```bash
sbatch hpc/run_pipeline.slurm [MODEL_PATH]
```

**Examples:**
```bash
# Default (Llama-3-8B-Instruct)
sbatch hpc/run_pipeline.slurm

# Llama 2
sbatch hpc/run_pipeline.slurm meta-llama/Llama-2-7b-chat-hf

# Gemma
sbatch hpc/run_pipeline.slurm google/gemma-2b-it

# Qwen
sbatch hpc/run_pipeline.slurm Qwen/Qwen-1_8B-Chat

# Yi
sbatch hpc/run_pipeline.slurm 01-ai/Yi-6B-Chat
```

### `setup_greene.sh`
One-time setup script to prepare the environment.

**What it does:**
- Creates directory structure in scratch
- Copies project code
- Sets up Python virtual environment
- Installs all dependencies

**Usage:**
```bash
bash hpc/setup_greene.sh
```

## Directory Structure on Greene

After setup, your files will be organized as:

```
/scratch/ah7660/refusal_direction_tfl/
├── code/                          # Project code
│   ├── pipeline/
│   │   └── runs/                  # Pipeline outputs (created during runs)
│   ├── dataset/
│   └── hpc/
├── venv/                          # Python virtual environment
├── cache/                         # HuggingFace model cache
└── results/                       # Additional results (if needed)

~/refusal_direction_tfl/           # Original code (in home directory)
└── logs/                          # SLURM output logs
```

**Important**:
- Use `/scratch/` for all data and models (5 TB quota)
- Keep only code in `/home/` (50 GB quota)
- Pipeline outputs go to `/scratch/ah7660/refusal_direction_tfl/code/pipeline/runs/`

## Interactive Testing

Before submitting batch jobs, test interactively:

```bash
# Request interactive GPU session
srun --gres=gpu:1 -c 8 --mem=32GB -t 2:00:00 --pty /bin/bash

# Activate environment
source /scratch/ah7660/refusal_direction_tfl/venv/bin/activate

# Navigate to code
cd /scratch/ah7660/refusal_direction_tfl/code

# Verify GPU access
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run pipeline with small test
python3 -m pipeline.run_pipeline --model_path google/gemma-2b-it

# Exit when done
exit
```

## Customizing Resource Requests

Edit `run_pipeline.slurm` to adjust resources:

### For Smaller Models (< 3B parameters)
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
```

### For Large Models (7B-13B parameters)
```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1  # Request high-memory GPU
#SBATCH --time=48:00:00
```

### For Very Large Models (> 13B parameters)
```bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1     # Request A100 GPU
#SBATCH --time=72:00:00
```

## Common Issues and Solutions

### Issue: Job Pending for Long Time

**Cause**: Requesting specific GPU types increases wait time.

**Solution**: Use generic GPU request for faster scheduling:
```bash
#SBATCH --gres=gpu:1  # Instead of gpu:v100:1
```

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions:**
1. Request GPU with more memory:
   ```bash
   #SBATCH --gres=gpu:rtx8000:1  # 48GB
   # or
   #SBATCH --gres=gpu:a100:1      # 40GB
   ```

2. Use smaller model:
   ```bash
   sbatch hpc/run_pipeline.slurm google/gemma-2b-it
   ```

### Issue: HuggingFace Model Download Fails

**Cause**: Missing authentication for gated models.

**Solution**: Set HuggingFace token:
```bash
export HF_TOKEN='your_token_here'
# Or add to ~/.bashrc
```

### Issue: Home Directory Quota Exceeded

**Cause**: Models cached in home directory instead of scratch.

**Solution**: Ensure cache environment variables are set:
```bash
export HF_HOME=/scratch/ah7660/refusal_direction_tfl/cache
export TRANSFORMERS_CACHE=/scratch/ah7660/refusal_direction_tfl/cache
```

### Issue: Job Killed Without Error

**Cause**: Exceeded memory or time limits.

**Solution**: Check with `seff JOBID` and increase limits in SLURM script.

## Retrieving Results

After job completes, results are in:
```bash
/scratch/ah7660/refusal_direction_tfl/code/pipeline/runs/{model_alias}/
```

To copy back to your local machine:

```bash
# From your Mac
rsync -av --progress \
    ah7660@greene.hpc.nyu.edu:/scratch/ah7660/refusal_direction_tfl/code/pipeline/runs/ \
    ~/Documents/school-work/work-25-q4f/pavel-research/refusal_direction_tfl/pipeline/runs/
```

Or use Data Transfer Nodes for large transfers:
```bash
rsync -av --progress \
    ah7660@dtn.hpc.nyu.edu:/scratch/ah7660/refusal_direction_tfl/code/pipeline/runs/ \
    ~/Documents/school-work/work-25-q4f/pavel-research/refusal_direction_tfl/pipeline/runs/
```

## Job Arrays for Multiple Models

To run the pipeline on multiple models in parallel:

Create `run_multiple_models.slurm`:
```bash
#!/bin/bash
#SBATCH --array=0-4
#SBATCH --job-name=refusal_array
#SBATCH --output=logs/refusal_%A_%a.out
#SBATCH --error=logs/refusal_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

# Define models
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-2-7b-chat-hf"
    "google/gemma-2b-it"
    "Qwen/Qwen-1_8B-Chat"
    "01-ai/Yi-6B-Chat"
)

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running pipeline with model: $MODEL"

# Run pipeline (rest of script same as run_pipeline.slurm)
# ... setup code ...
python3 -m pipeline.run_pipeline --model_path "$MODEL"
```

Submit:
```bash
sbatch run_multiple_models.slurm
```

## Monitoring GPU Usage

While job is running:

```bash
# Find node where job is running
squeue -u ah7660

# SSH to that node (e.g., gpu-24)
ssh gpu-24

# Monitor GPU
watch -n 1 nvidia-smi

# Exit back to login node
exit
```

## Best Practices

1. **Test Interactively First**: Always test with small epoch counts or small models before submitting long jobs

2. **Use Scratch**: Store all data, models, and results in `/scratch/` not `/home/`

3. **Monitor Efficiency**: After jobs complete, run `seff JOBID` to check resource utilization

4. **Checkpointing**: The pipeline saves checkpoints - jobs can be resumed if interrupted

5. **Email Notifications**: Update email in SLURM script to get notified when jobs complete

6. **Right-size Resources**: Don't request more than you need - it increases queue time

## Additional Resources

- **Full HPC Docs**: `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/`
- **NYU HPC Help**: hpc@nyu.edu
- **GPU Guide**: `hpc_docs/05-gpu-cuda-usage.md`
- **SLURM Guide**: `hpc_docs/04-slurm-job-submission.md`
- **ML Workflows**: `hpc_docs/06-ml-ai-workflows.md`

## Support

For issues specific to this pipeline, check:
1. Job output logs: `logs/refusal_JOBID.out`
2. Error logs: `logs/refusal_JOBID.err`
3. Main project README: `../README.md`

For HPC-related issues:
- Contact: hpc@nyu.edu
- Include: Job ID, error messages, what you've tried
