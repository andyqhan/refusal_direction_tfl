"""
GSM8K Dataset Processing Script

This script downloads the GSM8K dataset, processes it, and saves a modified version.
"""

import os
# Set tokenizers parallelism to false to avoid fork warnings
# This must be set before importing any HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import random
from datasets import load_dataset
from typing import Tuple, Optional, List


# Regex pattern to match equations: [number] [optional unit] [operator] [number] [optional unit] = [number]
# Pattern for numbers: integer or decimal (1234, 1,234, 12.34, 1,234.56, .25)
NUMBER_PATTERN = r'(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)'
EQUATION_PATTERN = rf'{NUMBER_PATTERN}\s*[a-zA-Z]*\s*[\+\-\*/]\s*{NUMBER_PATTERN}\s*[a-zA-Z]*\s*=\s*-?{NUMBER_PATTERN}'


def has_equation_pattern(text: str) -> bool:
    """Check if text contains an equation pattern."""
    return bool(re.search(EQUATION_PATTERN, text))


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    # Match numbers including decimals and those with commas
    pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(pattern, text)
    numbers = []
    for match in matches:
        try:
            # Remove commas and convert to float
            num = float(match.replace(',', ''))
            numbers.append(num)
        except ValueError:
            continue
    return numbers


def parse_equation(text: str) -> Optional[Tuple[float, str, float, float, str, str, str, str]]:
    """
    Parse an equation from text.

    Returns:
        Tuple of (operand1, operator, operand2, result, prefix, suffix, unit1, unit2) or None if parsing fails
        prefix is the text before the equation, suffix is the text after
    """
    # Use NUMBER_PATTERN for more accurate parsing
    pattern = rf'(.*?)({NUMBER_PATTERN})\s*([a-zA-Z]*)\s*([\+\-\*/])\s*({NUMBER_PATTERN})\s*([a-zA-Z]*)\s*=\s*(-?{NUMBER_PATTERN})(.*)'
    match = re.search(pattern, text)
    if not match:
        return None

    prefix = match.group(1)
    operand1_str = match.group(2)
    unit1 = match.group(3)
    operator = match.group(4)
    operand2_str = match.group(5)
    unit2 = match.group(6)
    result_str = match.group(7)
    suffix = match.group(8)

    try:
        operand1 = float(operand1_str.replace(',', ''))
        operand2 = float(operand2_str.replace(',', ''))
        result = float(result_str.replace(',', ''))

        return (operand1, operator, operand2, result, prefix, suffix, unit1, unit2)
    except ValueError:
        return None


def compute_result(operand1: float, operator: str, operand2: float) -> float:
    """Compute the result of an operation."""
    if operator == '+':
        return operand1 + operand2
    elif operator == '-':
        return operand1 - operand2
    elif operator == '*':
        return operand1 * operand2
    elif operator == '/':
        if operand2 == 0:
            return 0  # Avoid division by zero
        return operand1 / operand2
    return 0


def format_number(num: float) -> str:
    """Format a number, removing trailing zeros and limiting decimal places."""
    if num == int(num):
        return str(int(num))
    # Round to 6 decimal places to avoid very long decimals
    rounded = round(num, 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded).rstrip('0').rstrip('.')


def operand_swap(equation: str, full_answer: str) -> str:
    """
    Swap operands for - or / operators and compute correct result.

    Args:
        equation: The equation string (first line, processed)
        full_answer: The full original answer (needed for context)

    Returns:
        Modified equation with swapped operands
    """
    parsed = parse_equation(equation)
    if not parsed:
        return equation

    operand1, operator, operand2, result, prefix, suffix, unit1, unit2 = parsed

    if operator not in ['-', '/']:
        return equation

    # Swap operands and compute new result
    new_result = compute_result(operand2, operator, operand1)

    # Reconstruct equation with proper spacing
    unit1_str = f' {unit1}' if unit1 else ''
    unit2_str = f' {unit2}' if unit2 else ''
    new_equation = f"{prefix}{format_number(operand2)}{unit2_str} {operator} {format_number(operand1)}{unit1_str} = {format_number(new_result)}{suffix}"

    return ensure_period(new_equation)


def number_substitution(equation: str, full_answer: str, question: str) -> str:
    """
    Replace result with a random different number from question/answer.

    Args:
        equation: The equation string (first line, processed)
        full_answer: The full original answer
        question: The question text

    Returns:
        Modified equation with substituted number
    """
    parsed = parse_equation(equation)
    if not parsed:
        return equation

    operand1, operator, operand2, result, prefix, suffix, unit1, unit2 = parsed

    # Extract all numbers from question and full answer
    all_numbers = extract_numbers(question + ' ' + full_answer)

    # Filter out the current result (with tolerance for floating point comparison)
    different_numbers = [num for num in all_numbers if abs(num - result) > 0.0001]

    if not different_numbers:
        # Fallback: use operand1 or operand2 if they're different from result
        different_numbers = [num for num in [operand1, operand2] if abs(num - result) > 0.0001]

    if not different_numbers:
        # Last resort: add/subtract 1 from result
        different_numbers = [result + 1, result - 1]

    # Pick a random different number
    new_result = random.choice(different_numbers)

    # Reconstruct equation with proper spacing
    unit1_str = f' {unit1}' if unit1 else ''
    unit2_str = f' {unit2}' if unit2 else ''
    new_equation = f"{prefix}{format_number(operand1)}{unit1_str} {operator} {format_number(operand2)}{unit2_str} = {format_number(new_result)}{suffix}"

    return ensure_period(new_equation)


def operator_replace(equation: str, full_answer: str) -> str:
    """
    Replace operator with a random different one and compute correct result.

    Args:
        equation: The equation string (first line, processed)
        full_answer: The full original answer

    Returns:
        Modified equation with replaced operator
    """
    parsed = parse_equation(equation)
    if not parsed:
        return equation

    operand1, operator, operand2, result, prefix, suffix, unit1, unit2 = parsed

    # Choose a different operator
    operators = ['+', '-', '*', '/']
    different_operators = [op for op in operators if op != operator]
    new_operator = random.choice(different_operators)

    # Compute new result
    new_result = compute_result(operand1, new_operator, operand2)

    # Reconstruct equation with proper spacing
    unit1_str = f' {unit1}' if unit1 else ''
    unit2_str = f' {unit2}' if unit2 else ''
    new_equation = f"{prefix}{format_number(operand1)}{unit1_str} {new_operator} {format_number(operand2)}{unit2_str} = {format_number(new_result)}{suffix}"

    return ensure_period(new_equation)


def computation_plusminus(equation: str, full_answer: str) -> str:
    """
    Add or subtract a random number (1-10) from the result.

    Args:
        equation: The equation string (first line, processed)
        full_answer: The full original answer

    Returns:
        Modified equation with perturbed result
    """
    parsed = parse_equation(equation)
    if not parsed:
        return equation

    operand1, operator, operand2, result, prefix, suffix, unit1, unit2 = parsed

    # Generate random offset between 1 and 10
    offset = random.randint(1, 10)
    # Randomly choose to add or subtract
    sign = random.choice([-1, 1])
    new_result = result + (sign * offset)

    # Reconstruct equation with proper spacing
    unit1_str = f' {unit1}' if unit1 else ''
    unit2_str = f' {unit2}' if unit2 else ''
    new_equation = f"{prefix}{format_number(operand1)}{unit1_str} {operator} {format_number(operand2)}{unit2_str} = {format_number(new_result)}{suffix}"

    return ensure_period(new_equation)


def ensure_period(text: str) -> str:
    """Add a period to the end of text if it doesn't have one."""
    text = text.strip()
    if text and not text.endswith('.'):
        text += '.'
    return text


def extract_final_answer(answer: str) -> str:
    """
    Extract the final answer from the answer string.
    The final answer is the line prepended by ####.

    Args:
        answer: The full answer string

    Returns:
        The final answer (without ####), or empty string if not found
    """
    lines = answer.split('\n')
    for line in lines:
        if line.strip().startswith('####'):
            # Remove #### and return the rest
            return line.strip().replace('####', '').strip()
    return ""


def process_answer(answer: str) -> str:
    """
    Process an answer by:
    1. Keeping only the first line
    2. Removing all <<...>> patterns
    3. Ensuring it ends with a period

    Args:
        answer: The original answer string

    Returns:
        The processed answer string
    """
    # Keep only the first line
    first_line = answer.split('\n')[0]

    # Remove all <<...>> patterns
    processed = re.sub(r'<<[^>]*>>', '', first_line)

    # Ensure it ends with a period
    processed = ensure_period(processed)

    return processed


def main():
    """Main function to process the GSM8K dataset."""

    print("Loading GSM8K dataset...")
    # Set random seed for reproducibility at the very start
    random.seed(47)

    # Load the dataset
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds['train']

    print(f"Total train samples: {len(train_ds)}")

    # Step 1: Filter for samples with equations in the first line
    print("Filtering for samples with equations...")
    equation_indices = []
    for idx in range(len(train_ds)):
        answer = train_ds[idx]['answer']
        first_line = answer.split('\n')[0]
        cleaned = re.sub(r'<<[^>]*>>', '', first_line)
        if has_equation_pattern(cleaned):
            equation_indices.append(idx)

    print(f"Found {len(equation_indices)} samples with equations")

    # Step 2: Further filter for equations with - or / (for operand_swap)
    print("Filtering for equations with - or / operators...")
    minus_divide_indices = []
    other_indices = []

    for idx in equation_indices:
        answer = train_ds[idx]['answer']
        first_line = answer.split('\n')[0]
        cleaned = re.sub(r'<<[^>]*>>', '', first_line)
        parsed = parse_equation(cleaned)
        if parsed and parsed[1] in ['-', '/']:
            minus_divide_indices.append(idx)
        else:
            other_indices.append(idx)

    print(f"Found {len(minus_divide_indices)} with - or /")
    print(f"Found {len(other_indices)} with + or *")

    # Step 3: Sample 128 from each group
    # Sample 128 for operand_swap (from - and / operators)
    if len(minus_divide_indices) < 128:
        print(f"Warning: Only {len(minus_divide_indices)} samples with - or /, need 128")
        operand_swap_indices = minus_divide_indices
    else:
        operand_swap_indices = random.sample(minus_divide_indices, 128)

    # Sample 128*3 = 384 from the remaining for the other three perturbations
    remaining_indices = [idx for idx in equation_indices if idx not in operand_swap_indices]
    if len(remaining_indices) < 384:
        print(f"Warning: Only {len(remaining_indices)} remaining samples, need 384")
        other_perturbation_indices = remaining_indices
    else:
        other_perturbation_indices = random.sample(remaining_indices, 384)

    # Split the 384 into 3 groups of 128 each
    number_sub_indices = other_perturbation_indices[:128]
    operator_replace_indices = other_perturbation_indices[128:256]
    plusminus_indices = other_perturbation_indices[256:384]

    print(f"\nPerturbation allocation:")
    print(f"  operand_swap: {len(operand_swap_indices)}")
    print(f"  number_substitution: {len(number_sub_indices)}")
    print(f"  operator_replace: {len(operator_replace_indices)}")
    print(f"  computation_plusminus: {len(plusminus_indices)}")
    print(f"  Total: {len(operand_swap_indices) + len(number_sub_indices) + len(operator_replace_indices) + len(plusminus_indices)}")

    # Step 4: Process each group with its perturbation
    print("\nProcessing samples...")
    processed_samples = []

    # Process operand_swap group
    for idx in operand_swap_indices:
        example = train_ds[idx]
        processed_answer = process_answer(example['answer'])
        perturbed = operand_swap(processed_answer, example['answer'])
        processed_samples.append({
            'question': example['question'],
            'answer': processed_answer,
            'perturbed_answer': perturbed,
            'perturbation_type': 'operand_swap',
            'full_answer': example['answer'],
            'final_answer': extract_final_answer(example['answer'])
        })

    # Process number_substitution group
    for idx in number_sub_indices:
        example = train_ds[idx]
        processed_answer = process_answer(example['answer'])
        perturbed = number_substitution(processed_answer, example['answer'], example['question'])
        processed_samples.append({
            'question': example['question'],
            'answer': processed_answer,
            'perturbed_answer': perturbed,
            'perturbation_type': 'number_substitution',
            'full_answer': example['answer'],
            'final_answer': extract_final_answer(example['answer'])
        })

    # Process operator_replace group
    for idx in operator_replace_indices:
        example = train_ds[idx]
        processed_answer = process_answer(example['answer'])
        perturbed = operator_replace(processed_answer, example['answer'])
        processed_samples.append({
            'question': example['question'],
            'answer': processed_answer,
            'perturbed_answer': perturbed,
            'perturbation_type': 'operator_replace',
            'full_answer': example['answer'],
            'final_answer': extract_final_answer(example['answer'])
        })

    # Process computation_plusminus group
    for idx in plusminus_indices:
        example = train_ds[idx]
        processed_answer = process_answer(example['answer'])
        perturbed = computation_plusminus(processed_answer, example['answer'])
        processed_samples.append({
            'question': example['question'],
            'answer': processed_answer,
            'perturbed_answer': perturbed,
            'perturbation_type': 'computation_plusminus',
            'full_answer': example['answer'],
            'final_answer': extract_final_answer(example['answer'])
        })

    # Convert to HuggingFace dataset
    from datasets import Dataset
    processed_ds = Dataset.from_list(processed_samples)

    print("Saving processed dataset...")
    # Save the processed dataset
    output_path = "./gsm8k_dataset/processed_data"
    processed_ds.save_to_disk(output_path)

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(processed_ds)}")
    print(f"Columns: {processed_ds.column_names}")

    # Print summary statistics
    print("\nPerturbation type distribution:")
    from collections import Counter
    perturbation_counts = Counter(processed_ds['perturbation_type'])
    for ptype, count in perturbation_counts.items():
        print(f"  {ptype}: {count}")

    print("\nExample entries (one from each perturbation type):")
    for ptype in ['operand_swap', 'number_substitution', 'operator_replace', 'computation_plusminus']:
        examples = [ex for ex in processed_samples if ex['perturbation_type'] == ptype]
        if examples:
            ex = examples[0]
            print(f"\n{ptype}:")
            print(f"  Question: {ex['question'][:80]}...")
            print(f"  Original: {ex['answer']}")
            print(f"  Perturbed: {ex['perturbed_answer']}")


if __name__ == "__main__":
    main()
