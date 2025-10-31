# GSM8K Dataset Processing

This folder contains a script to download, process, and save a modified version of the GSM8K dataset with answer perturbations.

## Overview

The `process_gsm8k.py` script performs the following steps:

1. **Downloads** the GSM8K dataset from HuggingFace (`openai/gsm8k`)
2. **Filters** for samples with clear equations in the first line (pattern: `[number] [operator] [number] = [number]`)
3. **Intelligently samples** 512 examples across 4 perturbation types (128 each):
   - `operand_swap`: 128 samples from equations with `-` or `/` operators
   - `number_substitution`: 128 samples from remaining equations
   - `operator_replace`: 128 samples from remaining equations
   - `computation_plusminus`: 128 samples from remaining equations
4. **Processes** the answers:
   - Extracts only the first line of each answer
   - Removes all `<<...>>` calculation annotations
5. **Perturbs** the equations using four different strategies
6. **Saves** the processed dataset in HuggingFace Dataset format

## Perturbation Strategies

All perturbations modify the equation in the first line of the answer. Random seed 47 ensures reproducibility.

### 1. operand_swap (128 samples)
Swaps operands in equations with `-` or `/` operators and computes the correct result.
- Original: `13 - 10 = 3`
- Perturbed: `10 - 13 = -3`

### 2. number_substitution (128 samples)
Replaces the result with a random different number from the question or answer.
- Original: `2*12 = 24`
- Perturbed: `2*12 = 6` (6 is taken from the question text)

### 3. operator_replace (128 samples)
Replaces the operator with a random different one from `{+, -, *, /}` and computes the correct result.
- Original: `8 + 4 = 12`
- Perturbed: `8 / 4 = 2`

### 4. computation_plusminus (128 samples)
Adds or subtracts a random number between 1-10 from the result.
- Original: `560/5 = 112`
- Perturbed: `560/5 = 121` (112 + 9)

## Usage

Run the processing script:

```bash
python gsm8k_dataset/process_gsm8k.py
```

The script will:
- Filter ~4,278 samples with equations from the 7,473 train samples
- Allocate 128 samples to each perturbation type
- Display progress and example outputs
- Save the final dataset

## Output

The script saves the processed dataset to `./gsm8k_dataset/processed_data/`

**Dataset columns:**
- `question`: The original math problem
- `answer`: The processed answer (first line only, no `<<...>>` annotations)
- `perturbed_answer`: The perturbed answer with modified equation
- `perturbation_type`: One of `operand_swap`, `number_substitution`, `operator_replace`, or `computation_plusminus`

**Example:**

Original answer:
```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

Processed answer:
```
Natalia sold 48/2 = 24 clips in May.
```

## Loading the Processed Dataset

To load the processed dataset:

```python
from datasets import load_from_disk

dataset = load_from_disk("./gsm8k_dataset/processed_data")
print(dataset[0])

# Filter by perturbation type
operand_swap_samples = dataset.filter(lambda x: x['perturbation_type'] == 'operand_swap')
```

## Implementation Details

- **Random seed**: 47 (set once at the start for all random operations)
- **Equation pattern**: Robust regex matching numbers with decimals, commas, and optional units
- **Helper functions**: `parse_equation()`, `compute_result()`, `extract_numbers()`, etc.
- **Total samples**: 512 (128 per perturbation type)
