#!/usr/bin/env python3
"""
Interactive chatbot interface with steered concept vectors.

This script loads a direction vector from mean_diffs.pt and creates an
interactive terminal chatbot where you can chat with the model with
activation addition intervention applied.
"""

import argparse
import os
import re
import sys
import threading
import time
from typing import Optional, List

import torch

try:
    from prompt_toolkit import prompt as prompt_input
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import ANSI
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    print("Warning: prompt_toolkit not available. Install with: pip install prompt_toolkit")
    print("Falling back to basic input (no arrow key support).\n")

import warnings
# Suppress the torch.tensor() copy warning from transformers
warnings.filterwarnings('ignore', message='To copy construct from a tensor.*', category=UserWarning)

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, add_hooks


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chatbot with steered concept vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mean_diffs_path', type=str, required=True,
                        help='Path to mean_diffs.pt file (e.g., ./pipeline/runs/<model-name>/generate_directions/mean_diffs.pt)')
    parser.add_argument('--layer', type=int, required=True,
                        help='Which layer to inject the vector (0-indexed)')
    parser.add_argument('--pos', type=int, required=True,
                        help='Which token position from mean_diffs (0-indexed, or negative for counting from end)')
    parser.add_argument('--coeff', type=float, required=True,
                        help='Steering coefficient (strength of the intervention)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model (if not provided, will be inferred from mean_diffs_path)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7, use 0.0 for greedy/deterministic)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of parallel responses to generate for each prompt')
    parser.add_argument('--last-token-only', action='store_true', default=True,
                        help='Only apply steering to the last token position (default: True)')
    parser.add_argument('--all-tokens', dest='last_token_only', action='store_false',
                        help='Apply steering to all token positions (overrides --last-token-only)')

    return parser.parse_args()


def extract_model_path_from_mean_diffs_path(mean_diffs_path: str) -> Optional[str]:
    """
    Try to infer the model path from the mean_diffs.pt path.

    Expected format: ./pipeline/runs/<model-name>/generate_directions/mean_diffs.pt

    Returns the model name/path component, or None if extraction fails.
    """
    # Normalize path
    mean_diffs_path = os.path.normpath(mean_diffs_path)

    # Try to match pattern: */runs/<model-name>/*
    match = re.search(r'/runs/([^/]+)/', mean_diffs_path)
    if match:
        return match.group(1)

    return None


def load_direction_vector(mean_diffs_path, pos, layer):
    """
    Load mean_diffs.pt and extract the direction vector.

    Returns:
        direction: Tensor of shape [d_model]
        mean_diffs_shape: Shape of the full mean_diffs tensor
    """
    print(f"Loading mean_diffs from: {mean_diffs_path}")
    mean_diffs = torch.load(mean_diffs_path)
    mean_diffs_shape = list(mean_diffs.shape)

    print(f"Mean diffs shape: {mean_diffs_shape}")
    print(f"  - {mean_diffs_shape[0]} token positions")
    print(f"  - {mean_diffs_shape[1]} layers")
    print(f"  - {mean_diffs_shape[2]} hidden dimensions")

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


class Spinner:
    """A simple spinner to show progress during generation."""

    def __init__(self, message="Generating"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        """Internal method to run the spinner animation."""
        idx = 0
        while self.running:
            sys.stdout.write(f'\r\033[1;32mAssistant:\033[0m {self.spinner_chars[idx]} {self.message}...')
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()


def generate_response(model_base, conversation_history, direction, layer, coeff, max_new_tokens,
                      temperature, batch_size, last_token_only, steering_start_pos=None):
    """
    Generate response(s) to the user's message with steering applied.

    Args:
        model_base: The ModelBase instance
        conversation_history: Full conversation history (list of dicts with 'role' and 'content')
        direction: Direction vector to steer with
        layer: Layer to apply steering
        coeff: Steering coefficient
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Number of parallel responses to generate
        last_token_only: Whether to apply steering only to last token
        steering_start_pos: Position to start steering from (None = calculate from first message)

    Returns:
        Tuple of (responses, steering_start_pos) where responses is a list of strings
    """
    # If steering_start_pos not provided and last_token_only is True, calculate it
    if steering_start_pos is None and last_token_only:
        # Tokenize just the first user message to get its length
        first_message = [conversation_history[0]]  # Just the first user message
        first_tokenized = model_base.tokenize_instructions_fn(instructions=[first_message])
        # The steering should start from the last token of the first message
        steering_start_pos = first_tokenized.input_ids.shape[1] - 1

    # Tokenize the full conversation history and replicate for batch_size
    tokenized = model_base.tokenize_instructions_fn(instructions=[conversation_history] * batch_size)

    # Create activation addition hooks
    fwd_pre_hooks = [(
        model_base.model_block_modules[layer],
        get_activation_addition_input_pre_hook(vector=direction, coeff=coeff,
                                               last_token_only=last_token_only,
                                               steering_start_pos=steering_start_pos if last_token_only else None)
    )]

    # Generate with hooks
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
        if temperature == 0.0:
            # Greedy sampling
            generation_toks = model_base.model.generate(
                input_ids=tokenized.input_ids.to(model_base.model.device),
                attention_mask=tokenized.attention_mask.to(model_base.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id,
            )
        else:
            # Temperature sampling
            generation_toks = model_base.model.generate(
                input_ids=tokenized.input_ids.to(model_base.model.device),
                attention_mask=tokenized.attention_mask.to(model_base.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=model_base.tokenizer.pad_token_id,
            )

    # Decode all generated responses
    generated_tokens = generation_toks[:, tokenized.input_ids.shape[-1]:]
    responses = []
    for i in range(batch_size):
        response = model_base.tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip()
        responses.append(response)

    return responses, steering_start_pos


def chat_loop(model_base, direction, layer, coeff, max_new_tokens, temperature, batch_size, last_token_only):
    """
    Run the interactive chat loop.
    """
    print("\n" + "=" * 80)
    print("CHATBOT READY")
    print("=" * 80)
    print("\nType your messages and press Enter. Type 'quit', 'exit', or Ctrl+C to exit.")
    print("Type 'clear' to clear the screen and reset conversation history.")
    if batch_size > 1:
        print("Type '/r N' to select response N to add to conversation history.\n")
    else:
        print()

    # Initialize conversation history and input history
    conversation_history = []
    last_responses = []  # Store last batch of responses for /r command
    steering_start_pos = None  # Track where steering starts (for --last-token-only mode)
    if PROMPT_TOOLKIT_AVAILABLE:
        input_history = InMemoryHistory()

    # Warn if using batch_size > 1 with temperature = 0
    if batch_size > 1 and temperature == 0.0:
        print("\033[1;33mNote:\033[0m batch_size > 1 with temperature=0.0 will produce identical responses.")
        print("Consider using --temperature > 0 for diverse responses.\n")

    while True:
        try:
            # Get user input
            if PROMPT_TOOLKIT_AVAILABLE:
                user_input = prompt_input(ANSI("\n\033[1;34mYou:\033[0m "), history=input_history).strip()
            else:
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()

            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                conversation_history = []  # Reset conversation history
                last_responses = []
                steering_start_pos = None  # Reset steering position
                print("\n" + "=" * 80)
                print("CONVERSATION RESET")
                print("=" * 80)
                print("\nConversation history cleared. Starting fresh.\n")
                continue
            elif user_input.startswith('/r '):
                # Handle response selection command
                if batch_size == 1:
                    print("\033[1;33mNote:\033[0m /r command only works with --batch_size > 1")
                    continue
                if not last_responses:
                    print("\033[1;33mNote:\033[0m No responses available. Generate responses first.")
                    continue

                try:
                    response_num = int(user_input[3:].strip())
                    if 1 <= response_num <= len(last_responses):
                        # Remove the previous assistant message (if any)
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            conversation_history.pop()
                        # Add selected response
                        conversation_history.append({"role": "assistant", "content": last_responses[response_num - 1]})
                        print(f"\033[1;32m✓\033[0m Response {response_num} added to conversation history.")
                    else:
                        print(f"\033[1;31mError:\033[0m Response number must be between 1 and {len(last_responses)}")
                except ValueError:
                    print(f"\033[1;31mError:\033[0m Invalid response number. Use: /r N (e.g., /r 2)")
                continue
            elif not user_input:
                continue

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Start spinner
            spinner = Spinner(message="Generating")
            spinner.start()

            try:
                # Generate response(s)
                old_steering_pos = steering_start_pos
                responses, steering_start_pos = generate_response(
                    model_base=model_base,
                    conversation_history=conversation_history,
                    direction=direction,
                    layer=layer,
                    coeff=coeff,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    batch_size=batch_size,
                    last_token_only=last_token_only,
                    steering_start_pos=steering_start_pos
                )
                # Show steering position on first turn if using last-token-only
                if old_steering_pos is None and steering_start_pos is not None and last_token_only:
                    print(f"\033[1;90m[Steering from token position {steering_start_pos} onward]\033[0m")
            finally:
                # Stop spinner
                spinner.stop()

            # Store responses for /r command
            last_responses = responses

            # Display response(s)
            if batch_size == 1:
                print("\033[1;32mAssistant:\033[0m " + responses[0])
                # Add assistant response to conversation history
                conversation_history.append({"role": "assistant", "content": responses[0]})
            else:
                # Display multiple responses
                print()
                for i, response in enumerate(responses, 1):
                    print(f"\033[1;32mAssistant (response {i}/{batch_size}):\033[0m")
                    print(response)
                    if i < batch_size:
                        print()  # Add blank line between responses

                # Add first response to conversation history (for multi-turn)
                conversation_history.append({"role": "assistant", "content": responses[0]})
                if batch_size > 1:
                    print(f"\n\033[1;33mNote:\033[0m Response 1 added to conversation history. Use '/r N' to select a different response.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            # Handle Ctrl+D
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {str(e)}")
            print("Continuing chat...")


def main():
    """Main execution function."""
    args = parse_arguments()

    print("=" * 80)
    print("Interactive Chatbot with Steered Concept Vectors")
    print("=" * 80)

    # Determine model path
    model_path = args.model_path
    if model_path is None:
        print("\nAttempting to infer model path from mean_diffs_path...")
        model_path = extract_model_path_from_mean_diffs_path(args.mean_diffs_path)
        if model_path is None:
            print("\n\033[1;31mError:\033[0m Could not infer model path from mean_diffs_path.")
            print("Please provide --model_path explicitly.")
            sys.exit(1)
        print(f"Inferred model path: {model_path}")

    # Load model
    print(f"\nLoading model from: {model_path}")
    sys.stdout.flush()
    try:
        model_base = construct_model_base(model_path)
        print(f"✓ Model loaded successfully")
        print(f"✓ Number of layers: {len(model_base.model_block_modules)}")
    except Exception as e:
        print(f"\n\033[1;31mError loading model:\033[0m {str(e)}")
        sys.exit(1)

    # Load direction vector
    try:
        direction, mean_diffs_shape = load_direction_vector(
            args.mean_diffs_path,
            args.pos,
            args.layer
        )
        print(f"✓ Direction vector loaded successfully")
    except Exception as e:
        print(f"\n\033[1;31mError loading direction vector:\033[0m {str(e)}")
        sys.exit(1)

    # Display configuration
    print("\n" + "─" * 80)
    print("Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Layer: {args.layer}")
    print(f"  Position: {args.pos}")
    print(f"  Coefficient: {args.coeff}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Last token only: {args.last_token_only}")
    print(f"  Input mode: {'prompt_toolkit (readline support)' if PROMPT_TOOLKIT_AVAILABLE else 'basic input'}")
    print("─" * 80)

    # Start chat loop
    chat_loop(
        model_base=model_base,
        direction=direction,
        layer=args.layer,
        coeff=args.coeff,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        last_token_only=args.last_token_only
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
