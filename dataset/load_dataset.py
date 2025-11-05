import os
import json
import random
from datasets import load_from_disk

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

# GSM8K perturbation dataset functions
def load_gsm8k_dataset():
    """Load the GSM8K perturbation dataset from disk."""
    gsm8k_path = os.path.join(dataset_dir_path, '../gsm8k_dataset/processed_data')
    dataset = load_from_disk(gsm8k_path)
    return dataset

def sample_gsm8k_data(dataset, n_samples, perturbation_types=None):
    """
    Sample data uniformly from specified perturbation types.

    Args:
        dataset: The GSM8K dataset
        n_samples: Number of samples to collect
        perturbation_types: List of perturbation types to sample from, or None for all types
                           Valid types: 'operand_swap', 'number_substitution', 'operator_replace', 'computation_plusminus'

    Returns two lists of chat-formatted samples:
    - baseline_samples: questions with correct answers
    - perturbed_samples: questions with perturbed answers

    Each sample is a list of chat dicts: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    """
    # Get all available perturbation types
    all_types = list(set(dataset['perturbation_type']))

    # Filter to requested types, or use all if None
    if perturbation_types is None:
        selected_types = all_types
    else:
        # Validate that requested types exist
        invalid_types = [t for t in perturbation_types if t not in all_types]
        if invalid_types:
            raise ValueError(f"Invalid perturbation types: {invalid_types}. Valid types are: {all_types}")
        selected_types = perturbation_types

    n_per_type = n_samples // len(selected_types)

    baseline_samples = []
    perturbed_samples = []

    # Sample uniformly from each selected perturbation type
    for ptype in selected_types:
        # Filter dataset by perturbation type
        indices = [i for i, t in enumerate(dataset['perturbation_type']) if t == ptype]
        sampled_indices = random.sample(indices, min(n_per_type, len(indices)))

        for idx in sampled_indices:
            question = dataset['question'][idx]
            answer = dataset['answer'][idx]
            perturbed_answer = dataset['perturbed_answer'][idx]

            # Create chat-formatted samples
            baseline_samples.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            perturbed_samples.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": perturbed_answer}
            ])

    return baseline_samples, perturbed_samples
