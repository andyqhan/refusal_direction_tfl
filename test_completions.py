#!/usr/bin/env python3
"""
Test script to evaluate candidate correction directions on prompts.

This script loads a direction vector from mean_diffs.pt and generates completions
with activation addition intervention. Supports loading prompts from a CSV file
or sampling from the GSM8K dataset.
"""

import argparse
import csv
import json
import os
import random
import sys

# Fix for macOS threading/multiprocessing issues - MUST be set before importing torch
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
# import multiprocessing

# Set torch to single thread
# torch.set_num_threads(1)

# Fix multiprocessing start method on macOS
# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass  # Already set

from dataset.load_dataset import load_gsm8k_dataset
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test candidate correction directions on prompts from CSV or GSM8K dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--mean_diffs_path', type=str, required=True,
                        help='Path to mean_diffs.pt file')
    parser.add_argument('--layer', type=int, required=True,
                        help='Which layer to inject the vector (0-indexed)')
    parser.add_argument('--pos', type=int, required=True,
                        help='Which token position from mean_diffs (0-indexed, or negative for counting from end)')
    parser.add_argument('--coeff', type=float, default=2.0,
                        help='Addition strength coefficient')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature for generation')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save completions JSON (prints to stdout if not specified)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--csv_file', type=str,
                        help='Path to CSV file with prompts (columns: "User prompt", "Category")')
    data_group.add_argument('--use_gsm8k', action='store_true',
                        help='Use GSM8K dataset instead of CSV file')

    return parser.parse_args()


def load_direction_vector(mean_diffs_path, pos, layer):
    """
    Load mean_diffs.pt and extract the direction vector.

    Returns:
        direction: Tensor of shape [d_model]
        mean_diffs_shape: Shape of the full mean_diffs tensor
    """
    print(f"Loading mean_diffs from: {mean_diffs_path}")
    sys.stdout.flush()
    mean_diffs = torch.load(mean_diffs_path)
    mean_diffs_shape = list(mean_diffs.shape)

    print(f"Mean diffs shape: {mean_diffs_shape}")
    print(f"  - {mean_diffs_shape[0]} token positions")
    print(f"  - {mean_diffs_shape[1]} layers")
    print(f"  - {mean_diffs_shape[2]} hidden dimensions")
    sys.stdout.flush()

    # Validate indices
    n_positions, n_layers, d_model = mean_diffs.shape

    # Handle negative indexing for position
    if pos < 0:
        pos = n_positions + pos

    if not (0 <= pos < n_positions):
        raise ValueError(f"Position {pos} out of bounds (must be in range [0, {n_positions}))")

    if not (0 <= layer < n_layers):
        raise ValueError(f"Layer {layer} out of bounds (must be in range [0, {n_layers}))")

    # Extract the direction vector
    direction = mean_diffs[pos, layer, :]
    print(f"\nExtracted direction from position {pos}, layer {layer}")
    print(f"Direction shape: {direction.shape}")
    print(f"Direction norm: {direction.norm().item():.4f}")

    return direction, mean_diffs_shape


def load_csv_prompts(csv_file):
    """
    Load prompts from CSV file.

    Expected columns: "User prompt", "Category" (and optionally others)

    Returns:
        List of dicts formatted for generate_completions with 'instruction' and 'category' keys
    """
    print(f"\nLoading prompts from CSV: {csv_file}")

    formatted_dataset = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            user_prompt = row['User prompt']
            category = row.get('Category', 'unknown')

            # Format as chat (just the user prompt)
            chat_format = [{"role": "user", "content": user_prompt}]

            formatted_dataset.append({
                'instruction': chat_format,
                'category': category
            })

    print(f"Loaded {len(formatted_dataset)} prompts")

    # Print distribution of categories
    from collections import Counter
    type_counts = Counter([item['category'] for item in formatted_dataset])
    print("Category distribution:")
    for category, count in sorted(type_counts.items()):
        print(f"  {category}: {count}")

    return formatted_dataset


def sample_gsm8k_questions(n_samples=50, seed=74):
    """
    Sample questions from GSM8K dataset.

    Returns:
        List of dicts formatted for generate_completions with 'instruction' and 'category' keys
    """
    print(f"\nLoading GSM8K dataset and sampling {n_samples} questions (seed={seed})")

    # Load dataset
    dataset = load_gsm8k_dataset()
    total_samples = len(dataset['question'])
    print(f"Total GSM8K samples: {total_samples}")

    # Sample indices
    random.seed(seed)
    if n_samples > total_samples:
        raise ValueError(f"Requested {n_samples} samples but dataset only has {total_samples}")

    indices = random.sample(range(total_samples), n_samples)

    # Format for generate_completions
    # The generate_completions method expects a list of dicts with 'instruction' and 'category' keys
    # However, 'instruction' should now be in chat format (list of chat dicts)
    formatted_dataset = []

    for idx in indices:
        question = dataset['question'][idx]
        perturbation_type = dataset['perturbation_type'][idx]

        # Format as chat (just the question, no answer - we want the model to generate)
        chat_format = [{"role": "user", "content": question}]

        formatted_dataset.append({
            'instruction': chat_format,
            'category': perturbation_type
        })

    print(f"Sampled {len(formatted_dataset)} questions")

    # Print distribution of perturbation types
    from collections import Counter
    type_counts = Counter([item['category'] for item in formatted_dataset])
    print("Perturbation type distribution:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")

    return formatted_dataset


def generate_with_intervention(model_base, dataset, direction, layer, coeff, max_new_tokens, batch_size, temperature):
    """
    Generate completions with activation addition intervention.

    Returns:
        List of completion dicts with 'category', 'prompt', and 'response' keys
    """
    print(f"\nGenerating completions with intervention:")
    print(f"  Layer: {layer}")
    print(f"  Coefficient: {coeff}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch size: {batch_size}")

    # Create activation addition hooks
    fwd_pre_hooks = [(
        model_base.model_block_modules[layer],
        get_activation_addition_input_pre_hook(vector=direction, coeff=coeff)
    )]
    fwd_hooks = []

    # Generate completions
    completions = model_base.generate_completions(
        dataset=dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    print(f"Generated {len(completions)} completions")

    return completions


def save_or_print_results(completions, metadata, output_file):
    """Save results to file or print to stdout."""
    results = {
        "metadata": metadata,
        "completions": completions
    }

    if output_file:
        print(f"\nSaving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully")
    else:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
        print("\n" + "=" * 80)
        print("COMPLETIONS")
        print("=" * 80)

        for i, completion in enumerate(completions, 1):
            print(f"\n{'─' * 80}")
            print(f"Sample {i}/{len(completions)}")
            print(f"Category: {completion['category']}")
            print(f"{'─' * 80}")
            print(f"PROMPT: {completion['prompt']}")
            print(f"\nRESPONSE: {completion['response']}")


def main():
    """Main execution function."""
    args = parse_arguments()

    print("=" * 80)
    if args.csv_file:
        print(f"Testing Candidate Correction Directions on CSV: {args.csv_file}")
    else:
        print("Testing Candidate Correction Directions on GSM8K")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    sys.stdout.flush()
    model_base = construct_model_base(args.model_path)
    print(f"Model loaded successfully")
    print(f"Number of layers: {len(model_base.model_block_modules)}")
    sys.stdout.flush()

    # Load direction vector
    direction, mean_diffs_shape = load_direction_vector(
        args.mean_diffs_path,
        args.pos,
        args.layer
    )

    # Load dataset based on source
    if args.csv_file:
        dataset = load_csv_prompts(args.csv_file)
    else:
        dataset = sample_gsm8k_questions(n_samples=10, seed=74)

    # Generate completions with intervention
    completions = generate_with_intervention(
        model_base=model_base,
        dataset=dataset,
        direction=direction,
        layer=args.layer,
        coeff=args.coeff,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature
    )

    # Prepare metadata
    metadata = {
        "model_path": args.model_path,
        "mean_diffs_path": args.mean_diffs_path,
        "mean_diffs_shape": mean_diffs_shape,
        "layer": args.layer,
        "pos": args.pos,
        "coeff": args.coeff,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "data_source": args.csv_file if args.csv_file else "gsm8k",
        "num_samples": len(completions)
    }

    # Save or print results
    save_or_print_results(completions, metadata, args.output_file)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
