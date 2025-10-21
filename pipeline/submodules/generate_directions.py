import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase


def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules=None, batch_size=1, positions=[-1], cache_layers_arg=None):
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
        cache_layers_arg: Control which layers to cache (int, "final", "threefourths", "half", or None for all)

    Returns:
        Mean activations tensor of shape (n_positions, n_layers, d_model)
    """
    # Import the helper function from run_pipeline
    from pipeline.run_pipeline import compute_layers_to_cache

    # Clear cache for the appropriate device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n_positions = len(positions)
    n_layers = model.cfg.n_layers
    n_samples = len(instructions)
    d_model = model.cfg.d_model

    # Determine which layers to cache
    layers_to_cache_list = compute_layers_to_cache(cache_layers_arg, n_layers)

    # Store mean activations in high-precision to avoid numerical issues
    # MPS doesn't support float64, use float32 instead
    precision_dtype = torch.float32 if model.cfg.device == 'mps' else torch.float64
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=precision_dtype, device=model.cfg.device)

    n_batches = (len(instructions) + batch_size - 1) // batch_size

    # Determine which layers to iterate over for extraction
    if layers_to_cache_list is None:
        # Cache and extract all layers
        layers_to_extract = range(n_layers)
        names_filter = lambda name: "hook_resid_pre" in name
    else:
        # Cache only specific layers
        layers_to_extract = layers_to_cache_list
        layers_set = set(layers_to_cache_list)
        names_filter = lambda name: "hook_resid_pre" in name and any(f"blocks.{layer}." in name for layer in layers_set)

    for i in tqdm(range(0, len(instructions), batch_size),
                  desc=f"Extracting activations ({len(instructions)} examples)",
                  unit="batch",
                  total=n_batches):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        # Use TransformerLens's run_with_cache to get activations
        # Only cache residual stream activations (hook_resid_pre) at selected layers
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                inputs.input_ids.to(model.cfg.device),
                prepend_bos=False,  # Don't add BOS token (already in input)
                names_filter=names_filter,  # Only cache what we need
            )

        # Extract activations from residual stream at specified positions
        for pos_idx, pos in enumerate(positions):
            for layer in layers_to_extract:
                # Get residual stream activations at this layer
                hook_name = f"blocks.{layer}.hook_resid_pre"
                layer_activations = cache[hook_name]  # Shape: (batch, seq, d_model)

                # Extract activations at the specified position and accumulate mean
                # pos is negative, so layer_activations[:, pos, :] gets the right position
                mean_activations[pos_idx, layer] += (1.0 / n_samples) * layer_activations[:, pos, :].sum(dim=0).to(precision_dtype)

    return mean_activations


def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1], cache_layers=None):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions, cache_layers_arg=cache_layers)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions, cache_layers_arg=cache_layers)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir, batch_size=1, cache_layers=None):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, model_base.model_block_modules, batch_size=batch_size, positions=list(range(-len(model_base.eoi_toks), 0)), cache_layers=cache_layers)

    # TransformerLens uses .cfg instead of .config
    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.cfg.n_layers, model_base.model.cfg.d_model)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs