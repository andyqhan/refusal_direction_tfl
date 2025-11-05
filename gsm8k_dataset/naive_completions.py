"""
Generate naive completions for GSM8K dataset with perturbed answers.

This script loads the modified GSM8K dataset, uses the perturbed_answer as a
token-forced prefix, and generates continuations to see what the model does naturally.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import os


def load_gsm8k_dataset():
    """Load the GSM8K perturbation dataset from disk."""
    dataset_dir = os.path.dirname(os.path.realpath(__file__))
    gsm8k_path = os.path.join(dataset_dir, 'processed_data')
    dataset = load_from_disk(gsm8k_path)
    return dataset


def load_qwen3_model(model_path):
    """Load Qwen3 model and tokenizer."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def format_prompt_with_prefix(question, perturbed_answer, tokenizer):
    """
    Format the prompt using Qwen3's chat template with the perturbed answer as prefix.

    Returns the full prompt string that includes question + perturbed answer prefix.
    """
    # Use Qwen3's ChatML format
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": perturbed_answer}
    ]

    # Apply chat template without adding generation prompt (we want to continue from the assistant message)
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        #enable_thinking=False
    )

    return prompt


def format_prompt_normally(question, tokenizer):
    """As a control, format the prompt normally, without token forcing."""
    # Use Qwen3's ChatML format
    chat = [
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        #enable_thinking=False
    )

    return prompt


def generate_completions(model, tokenizer, dataset, n_samples=50, max_new_tokens=128):
    """
    Generate completions for n_samples from the dataset.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataset: The GSM8K dataset
        n_samples: Number of samples to process
        max_new_tokens: Number of tokens to generate

    Returns:
        List of dicts containing question, perturbed_answer, continuation, and full_output
    """
    results = []

    # Sample n_samples from the dataset
    import random
    random.seed(74)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    print(f"\nGenerating completions for {len(indices)} samples...")

    for i, idx in enumerate(indices):
        question = dataset['question'][idx]
        answer = dataset['answer'][idx]
        perturbed_answer = dataset['perturbed_answer'][idx]
        perturbation_type = dataset['perturbation_type'][idx]

        # Format the prompt with perturbed answer as prefix
        full_prompt = format_prompt_with_prefix(question, perturbed_answer, tokenizer)

        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic results
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (the continuation)
        continuation = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)

        # Decode the full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # unforced_prompt = format_prompt_normally(question, tokenizer)
        # unforced_inputs = tokenizer(unforced_prompt, return_tensors="pt").to(model.device)
        # with torch.no_grad():
        #     unforced_outputs = model.generate(
        #         **unforced_inputs,
        #         max_new_tokens=max_new_tokens,
        #         do_sample=False,
        #         pad_token_id=tokenizer.eos_token_id,
        #         eos_token_id=tokenizer.eos_token_id,
        #     )

        # unforced_full_output = tokenizer.decode(unforced_outputs[0], skip_special_tokens=False)

        results.append({
            'index': idx,
            'question': question,
            'correct_answer': answer,
            'perturbed_answer': perturbed_answer,
            'perturbation_type': perturbation_type,
            'continuation': continuation,
            'full_output': full_output,
            # 'unforced_full_output': unforced_full_output,
        })

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(indices)} samples")

    return results


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_examples(results, n_examples=3):
    """Print a few example results."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE COMPLETIONS (showing {n_examples} examples)")
    print(f"{'='*80}\n")

    for i, result in enumerate(results[:n_examples]):
        print(f"Example {i+1}:")
        print(f"Perturbation type: {result['perturbation_type']}")
        print(f"\nQuestion: {result['question'][:100]}...")
        print(f"\nCorrect answer: {result['correct_answer']}")
        print(f"\nPerturbed answer (forced prefix): {result['perturbed_answer']}")
        print(f"\nModel continuation: {result['continuation'][:200]}...")
        print(f"\n{'-'*80}\n")


def main():
    # Configuration
    model_path = "google/gemma-3-1b-it"
    n_samples = 15
    max_new_tokens = 256
    output_path = os.path.join(os.path.dirname(__file__), 'naive_completions_results.json')

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_gsm8k_dataset()
    print(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    model, tokenizer = load_qwen3_model(model_path)

    # Generate completions
    results = generate_completions(
        model,
        tokenizer,
        dataset,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens
    )

    # Save results
    save_results(results, output_path)

    # Print examples
    print_examples(results, n_examples=3)

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
