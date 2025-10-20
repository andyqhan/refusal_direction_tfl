# Testing Guide for Refusal Direction Pipeline

This document describes how to test the TransformerLens-based refusal direction pipeline.

## Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `pytest==7.4.3` - Testing framework
- `pytest-cov==4.1.0` - Coverage reporting
- `transformer-lens==2.10.0` - Core library
- Other dependencies from requirements.txt

## Quick Test Commands

### 1. Unit Tests (Fast - ~30-60 seconds)

Tests basic TransformerLens integration without requiring large models.

**With pytest:**
```bash
pytest tests/test_transformerlens_integration.py -v
```

**Manual (no pytest needed):**
```bash
python tests/test_transformerlens_integration.py
```

**What it tests:**
- Model loading with TransformerLens
- Hook creation (ablation and activation addition)
- Tokenization and chat template formatting
- Forward pass with hooks applied
- Hook application correctness

### 2. Minimal Integration Test (Medium - ~2-5 minutes)

Tests the core pipeline with a tiny model (GPT-2) and minimal data.

```bash
python tests/test_pipeline_minimal.py
```

**What it tests:**
- End-to-end direction generation
- Artifact saving
- Output shape and magnitude validation
- Variance across layers

### 3. Full Pipeline Test (Slow - ~30+ minutes, requires GPU)

Tests with a real model from the supported families.

**Smallest model (recommended for testing):**
```bash
python -m pipeline.run_pipeline --model_path gpt2
```

**Production models (requires HuggingFace authentication):**
```bash
# Gemma 2B (smallest supported production model)
python -m pipeline.run_pipeline --model_path google/gemma-2b-it

# Llama 3 8B
python -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

## Test Coverage

### Unit Tests (`test_transformerlens_integration.py`)

1. **test_model_loading**: Verifies TransformerLens model loads correctly
2. **test_direction_ablation_hook**: Tests direction projection/ablation
3. **test_activation_addition_hook**: Tests vector addition to activations
4. **test_tokenization**: Tests chat template formatting
5. **test_all_ablation_hooks_creation**: Verifies hooks for all layers
6. **test_forward_pass_with_hooks**: Tests model inference with hooks
7. **test_activation_addition_hooks_creation**: Tests single-layer hooks

### Integration Test (`test_pipeline_minimal.py`)

1. Loads GPT-2 model via TransformerLens
2. Generates refusal directions from 2 harmful + 2 harmless examples
3. Validates:
   - Output tensor shape
   - Non-zero directions
   - Reasonable magnitude
   - Variance across layers

## Expected Output

### Unit Tests Success
```
Running TransformerLens integration tests...

✓ Model loading test passed
✓ Direction ablation hook test passed
✓ Activation addition hook test passed
✓ Tokenization test passed
✓ All ablation hooks creation test passed
✓ Forward pass with hooks test passed
✓ Activation addition hooks creation test passed

✅ All tests passed!
```

### Integration Test Success
```
============================================================
MINIMAL PIPELINE INTEGRATION TEST
============================================================

[1/3] Loading model...
✓ Loaded gpt2
  - Device: cuda:0  (or cpu if no GPU)
  - Layers: 12
  - Hidden size: 768

[2/3] Generating refusal directions...
  - Harmful examples: 2
  - Harmless examples: 2
  ✓ Artifacts saved successfully

[3/3] Validating output...
  ✓ Shape correct: (1, 12, 768)
  ✓ Directions non-zero
  ✓ Direction magnitude reasonable: 0.1234
  ✓ Variance across layers: 0.0567

============================================================
✅ MINIMAL PIPELINE TEST PASSED!
============================================================
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'transformer_lens'`:
```bash
pip install transformer-lens
```

If you see `ModuleNotFoundError: No module named 'pytest'`:
```bash
pip install pytest pytest-cov
```

### CUDA/GPU Issues

If you don't have a GPU, TransformerLens will automatically use CPU. Tests will still work but may be slower.

To force CPU usage:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Memory Issues

If you run out of memory with GPT-2, try using a smaller model or reducing batch size:
- GPT-2 is ~500MB
- Use `gpt2` (124M params) instead of `gpt2-medium` or larger

### HuggingFace Authentication

For gated models (Llama, Gemma), you need a HuggingFace token:

```bash
# Option 1: Via huggingface-cli
huggingface-cli login

# Option 2: Via environment variable
export HF_TOKEN=your_token_here
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run all tests with coverage
pytest tests/ -v --cov=pipeline --cov-report=term

# Run only fast unit tests
pytest tests/test_transformerlens_integration.py -v

# Skip integration tests in CI
pytest tests/ -v -m "not integration"
```

## Comparing with Legacy Results

If you have pre-existing results from the original implementation:

```bash
# 1. Run new implementation
python -m pipeline.run_pipeline --model_path google/gemma-2b-it

# 2. Check results
ls pipeline/runs/gemma-2b-it/

# 3. Compare direction tensors
python
>>> import torch
>>> new_dir = torch.load('pipeline/runs/gemma-2b-it/direction.pt')
>>> old_dir = torch.load('pipeline/runs/gemma-2b-it/direction.pt.backup')
>>> torch.allclose(new_dir, old_dir, atol=1e-3)  # Should be very similar
```

## Performance Benchmarks

Approximate test times on a MacBook Pro (M1):

| Test | Time | GPU Required |
|------|------|--------------|
| Unit tests | 30-60s | No |
| Integration test | 2-5min | No |
| Full pipeline (GPT-2) | 5-10min | No |
| Full pipeline (Gemma 2B) | 30-60min | Recommended |
| Full pipeline (Llama 3 8B) | 1-2hrs | Yes |

## What Changed from Legacy Implementation

The cleanup removed:
- 5 model-specific implementation files (~700 lines)
- Legacy hook_utils.py (~126 lines)
- Detection code across 4 submodules (~200 lines)
- Total: ~1,000 lines of code removed

Benefits:
- Simpler codebase with single implementation path
- Cleaner API with named hook points
- Better performance via TransformerLens optimizations
- Easier to maintain and extend
