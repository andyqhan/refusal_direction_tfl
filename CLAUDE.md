# CLAUDE.md

This file provides guidance to AI agents, like Claude Code and OpenAI Codex, when working with code in this repository.

## Overview

This repository implements the research from "Refusal in Language Models Is Mediated by a Single Direction" ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)). The codebase extracts and evaluates "refusal directions" in language models - single activation vectors that mediate refusal behavior. By ablating or adding these directions, the research demonstrates it's possible to bypass or induce refusal in LLMs.

**Content Warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate. The research is defensive in nature, studying model safety mechanisms.

## Environment
- The virtual environment is located in `.venv`.

### Manual Environment Activation
```bash
source .venv/bin/activate
```

## Running the Pipeline

### Main Pipeline Command
```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```

Example model paths:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- `google/gemma-2b-it`
- `Qwen/Qwen-1_8B-Chat`
- `01-ai/Yi-6B-Chat`

### Pipeline Artifacts
All artifacts are saved to `pipeline/runs/{model_alias}/` with the following structure:
- `generate_directions/` - Candidate refusal directions (`mean_diffs.pt`)
- `select_direction/` - Direction selection results
- `direction.pt` - Final selected refusal direction
- `direction_metadata.json` - Position and layer info for selected direction
- `completions/` - Generated completions and evaluations
- `loss_evals/` - Cross-entropy loss evaluations

## Architecture

### Core Pipeline (pipeline/run_pipeline.py)

The pipeline executes 5 sequential steps:

1. **Generate Candidate Directions** (`generate_directions()`)
   - Extracts mean activation differences between harmful/harmless prompts
   - Computes differences across all layers and positions
   - Output: `mean_diffs.pt` tensor of shape `(n_positions, n_layers, d_model)`

2. **Select Best Direction** (`select_direction()`)
   - Evaluates candidate directions on validation set
   - Selects direction with best refusal score differential
   - Output: `direction.pt`, `direction_metadata.json`

3. **Generate Completions on Harmful Datasets**
   - Tests three interventions:
     - `baseline`: No intervention
     - `ablation`: Ablate refusal direction from all layers
     - `actadd`: Add negative refusal direction at selected layer
   - Evaluates on datasets specified in `Config.evaluation_datasets`

4. **Generate Completions on Harmless Dataset**
   - Tests baseline and positive actadd (induced refusal)
   - Uses random sample from harmless test split

5. **Evaluate Cross-Entropy Loss**
   - Measures impact on language modeling capabilities
   - Uses harmless completions as reference

### Model Architecture (pipeline/model_utils/)

**Factory Pattern**: `model_factory.py` routes to model-specific implementations based on model path string matching.

**Base Class** (`ModelBase`): Abstract class defining the interface for all models:
- `_load_model()`: Load HuggingFace model
- `_load_tokenizer()`: Load and configure tokenizer
- `_get_tokenize_instructions_fn()`: Returns model-specific prompt formatting function
- `_get_eoi_toks()`: End-of-instruction tokens (used to identify positions)
- `_get_refusal_toks()`: Tokens indicating refusal (e.g., "I" for "I cannot...")
- `_get_model_block_modules()`: Returns list of transformer blocks
- `_get_attn_modules()`: Returns attention submodules
- `_get_mlp_modules()`: Returns MLP submodules
- `generate_completions()`: Generate text with optional forward hooks

**Supported Models**: Each has model-specific chat templates and refusal tokens:
- `llama2_model.py`: Llama 2 Chat
- `llama3_model.py`: Llama 3 Instruct (note: uses corrected prompt format from cemiu/main merge)
- `gemma_model.py`: Gemma IT
- `qwen_model.py`: Qwen Chat
- `yi_model.py`: Yi Chat

### Hook System (pipeline/utils/hook_utils.py)

The intervention mechanism uses PyTorch forward hooks:

**Key Hook Functions**:
- `get_direction_ablation_input_pre_hook()`: Projects out refusal direction from residual stream
- `get_direction_ablation_output_hook()`: Projects out refusal direction from attention/MLP outputs
- `get_activation_addition_input_pre_hook()`: Adds scaled vector to activations
- `add_hooks()`: Context manager for temporarily registering hooks

**Ablation vs Activation Addition**:
- Ablation: Applied to ALL layers via pre-hooks on blocks + output hooks on attn/MLP
- Activation Addition: Applied to SINGLE layer via pre-hook on specific block

### Dataset Management (dataset/)

**Structure**:
- `raw/`: Downloaded raw datasets (not tracked in git)
- `processed/`: Standardized JSON format `[{'instruction': str, 'category': str}, ...]`
- `splits/`: Train/val/test splits for harmful and harmless data
- VERY IMPORTANT: Do not read the data files in these folders. They are large and contain malicious instructions. You may list them to get their names, but under no circumstances will you read their content.

**Data Sources**:
- Harmful: AdvBench, MaliciousInstruct, TDC2023, HarmBench, JailbreakBench, StrongReject
- Harmless: Alpaca (filtered for instruction-only examples)

**Loading Functions** (`load_dataset.py`):
- `load_dataset_split(harmtype, split, instructions_only)`: Load train/val/test split
- `load_dataset(dataset_name, instructions_only)`: Load specific processed dataset

**Dataset Generation**: Run `dataset/generate_datasets.ipynb` to download and process datasets (already done for this repo).

### Configuration (pipeline/config.py)

`Config` dataclass with defaults:
- `n_train=128`: Training examples per harmful/harmless set
- `n_val=32`: Validation examples
- `n_test=100`: Test examples
- `filter_train=True`: Filter out examples with unexpected refusal behavior
- `filter_val=True`: Filter validation set
- `evaluation_datasets=("jailbreakbench",)`: Harmful test sets
- `max_new_tokens=512`: Generation length
- `jailbreak_eval_methodologies=("substring_matching", "llamaguard2")`: Evaluation methods
- `ce_loss_batch_size=2`, `ce_loss_n_batches=2048`: Loss evaluation params

### Evaluation (pipeline/submodules/)

**Jailbreak Evaluation** (`evaluate_jailbreak.py`):
- `substring_matching`: Simple refusal string detection
- `llamaguard2`: Using Together AI API for safety classification

**Loss Evaluation** (`evaluate_loss.py`):
- Measures cross-entropy loss on harmless completions
- Tests whether interventions degrade language modeling

## TransformerLens Integration

This repository uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), a library designed for mechanistic interpretability. TransformerLens provides a unified API and cleaner hook system for working with transformer models.

### Using TransformerLens

All models are loaded using TransformerLens:

```bash
python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

The `TransformerLensModel` provides:
- **Unified API**: Single implementation for all supported models (Llama2, Llama3, Gemma, Qwen, Yi)
- **Named Hook Points**: Clean hook names like `blocks.{layer}.hook_resid_pre` instead of module references
- **Activation Caching**: Built-in `run_with_cache()` for efficient activation extraction
- **MI Tools**: Designed specifically for mechanistic interpretability research

### TransformerLens documentation
If you need to know how TransformerLens works, up-to-date LLM-friendly documentation, in Markdown files, can be found at `./TransformerLens_docs`. Read the `index.md` file in that directory to get some guidance on how to use them.

### TransformerLens Hook Points

Key hook points used in the pipeline:

**Residual Stream**:
- `blocks.{layer}.hook_resid_pre` - Residual stream at block input
- `blocks.{layer}.hook_resid_post` - Residual stream at block output (unused in this codebase)
- `blocks.{layer}.hook_resid_mid` - Between attention and MLP (unused)

**Attention**:
- `blocks.{layer}.attn.hook_result` - Attention output

**MLP**:
- `blocks.{layer}.hook_mlp_out` - MLP output

### How Interventions Work with TransformerLens

**Direction Ablation**: Projects out the refusal direction from activations
```python
from pipeline.utils.transformerlens_hook_utils import get_all_direction_ablation_hooks

# Creates hooks for all layers on resid_pre, attn output, and MLP output
fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, direction)
```

**Activation Addition**: Adds/subtracts a scaled direction at a specific layer
```python
from pipeline.utils.transformerlens_hook_utils import get_activation_addition_hooks

# Creates hook for single layer on resid_pre
fwd_hooks, _ = get_activation_addition_hooks(model_base, direction, coeff=-1.0, layer=15)
```

### Key TransformerLens Attributes

| Attribute | Description |
|-----------|-------------|
| `model.cfg.device` | Device where model is loaded |
| `model.cfg.n_layers` | Number of transformer layers |
| `model.cfg.d_model` | Hidden dimension size |
| `model.cfg.d_vocab` | Vocabulary size |

## Common Development Tasks

### Adding Support for New Model

1. Add model family detection to `TransformerLensModel._detect_model_family()` in `transformerlens_model.py`
2. Add chat template and refusal tokens to `_setup_model_specifics()`
3. Test with a small version of the model to verify:
   - TransformerLens can load it
   - Chat template formatting works
   - Refusal tokens are correct

### Changing Evaluation Datasets

Modify `Config.evaluation_datasets` tuple in `pipeline/config.py` or pass to Config constructor.

### Working with Directions

Load saved direction:
```python
import torch
direction = torch.load('pipeline/runs/{model_alias}/direction.pt')
```

Direction metadata:
```python
import json
with open('pipeline/runs/{model_alias}/direction_metadata.json') as f:
    metadata = json.load(f)  # {'pos': int, 'layer': int}
```

## Key Concepts

**Refusal Direction**: A single vector in the model's residual stream that represents the difference in activations between harmful and harmless prompts. The paper's key finding is that this single direction mediates refusal behavior across multiple layers.

**Positions**: The direction is computed over the last N token positions of the instruction (where N = length of end-of-instruction tokens). Position -1 is the last token before the model starts generating.

**Interventions**:
- **Baseline**: Model behavior without modification
- **Ablation**: Remove refusal direction from all components (blocks, attention, MLPs)
- **Activation Addition (actadd)**: Add/subtract refusal direction at a single layer

**Refusal Scores**: Probability assigned to refusal tokens (like "I") at the first generation position. Higher score indicates stronger refusal.

## Running on NYU HPC (Greene Cluster)

For NYU PhD students and researchers with access to Greene:

**HPC Documentation**: Complete documentation for running this pipeline on Greene is available at `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/`

**Quick Start on Greene**:
1. See `hpc/` directory in this repo for SLURM job scripts
2. Review `hpc/README.md` for detailed setup instructions
3. Submit jobs with `sbatch hpc/run_pipeline.slurm`

**Key Resources**:
- HPC Docs: `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/README.md`
- GPU Guide: `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/05-gpu-cuda-usage.md`
- ML Workflows: `~/Documents/school-work/work-25-q4f/pavel-research/hpc_docs/06-ml-ai-workflows.md`

## Important Notes

- Models are loaded with `device_map="auto"` for multi-GPU support
- All models use `eval()` mode and `requires_grad_(False)`
- Tokenizer padding is left-side for proper generation
- The Together AI API is rate-limited; evaluation may take time
- Llama 3 prompt format was fixed in PR #7 (cemiu/main merge)
