import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase


def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules=None, batch_size=1, positions=[-1]):
    """
    Get mean activations using TransformerLens's run_with_cache.

    This uses TransformerLens's built-in activation caching for efficient extraction.

    Args:
        model: HookedTransformer model
        tokenizer: Tokenizer
        instructions: List of instruction strings
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: Unused, kept for API compatibility
        batch_size: Batch size for processing
        positions: List of positions to extract (negative indices from end)

    Returns:
        Mean activations tensor of shape (n_positions, n_layers, d_model)
    """
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.cfg.n_layers
    n_samples = len(instructions)
    d_model = model.cfg.d_model

    # Store mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.cfg.device)

    n_batches = (len(instructions) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(instructions), batch_size),
                  desc=f"Extracting activations ({len(instructions)} examples)",
                  unit="batch",
                  total=n_batches):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        # Use TransformerLens's run_with_cache to get all activations
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                inputs.input_ids.to(model.cfg.device),
                prepend_bos=False,  # Don't add BOS token (already in input)
            )

        # Extract activations from residual stream at specified positions
        for pos_idx, pos in enumerate(positions):
            for layer in range(n_layers):
                # Get residual stream activations at this layer
                hook_name = f"blocks.{layer}.hook_resid_pre"
                layer_activations = cache[hook_name]  # Shape: (batch, seq, d_model)

                # Extract activations at the specified position and accumulate mean
                # pos is negative, so layer_activations[:, pos, :] gets the right position
                mean_activations[pos_idx, layer] += (1.0 / n_samples) * layer_activations[:, pos, :].sum(dim=0).to(torch.float64)

    return mean_activations


def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1]):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir, batch_size=1):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, model_base.model_block_modules, batch_size=batch_size, positions=list(range(-len(model_base.eoi_toks), 0)))

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs