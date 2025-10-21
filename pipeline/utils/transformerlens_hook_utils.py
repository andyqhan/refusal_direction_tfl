
import torch
from typing import List, Tuple, Callable, Dict
from jaxtyping import Float
from torch import Tensor


def get_direction_ablation_hook(direction: Tensor):
    """
    Create a hook function that ablates (projects out) a direction from activations.

    This hook removes the component of the activation that aligns with the given direction,
    effectively preventing the model from using that direction for computation.

    Args:
        direction: The direction vector to ablate (d_model,)

    Returns:
        Hook function compatible with TransformerLens
    """
    def hook_fn(activation: Float[Tensor, "batch seq d_model"], hook):
        """
        Hook function that projects out the direction from the activation.

        Args:
            activation: The activation tensor from the model
            hook: HookPoint object (provided by TransformerLens)

        Returns:
            Modified activation with direction projected out
        """
        # Normalize the direction
        dir_normalized = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        dir_normalized = dir_normalized.to(activation)

        # Project out the direction: activation -= (activation @ direction) * direction
        projection = (activation @ dir_normalized).unsqueeze(-1) * dir_normalized
        activation = activation - projection

        return activation

    return hook_fn


def get_activation_addition_hook(vector: Float[Tensor, "d_model"], coeff: float):
    """
    Create a hook function that adds a scaled vector to activations.

    This hook adds a constant vector (scaled by coeff) to the activation at every position,
    effectively steering the model in a particular direction.

    Args:
        vector: The vector to add (d_model,)
        coeff: Coefficient to scale the vector (positive to add, negative to subtract)

    Returns:
        Hook function compatible with TransformerLens
    """
    def hook_fn(activation: Float[Tensor, "batch seq d_model"], hook):
        """
        Hook function that adds a scaled vector to the activation.

        Args:
            activation: The activation tensor from the model
            hook: HookPoint object (provided by TransformerLens)

        Returns:
            Modified activation with vector added
        """
        vector_on_device = vector.to(activation)
        activation = activation + coeff * vector_on_device

        return activation

    return hook_fn


def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
) -> Tuple[List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
    """
    Generate hook specifications for ablating a direction across all layers.

    In the original implementation, ablation hooks are applied to:
    - Residual stream at the start of each block (hook_resid_pre)
    - Attention outputs (hook_result)
    - MLP outputs (hook_mlp_out)

    Args:
        model_base: The model instance
        direction: The direction vector to ablate

    Returns:
        Tuple of (fwd_hooks list, empty list) for compatibility with original API.
        In TransformerLens, we only need fwd_hooks (no pre-hooks).
    """
    n_layers = model_base.model.cfg.n_layers

    # Create hook specifications as (hook_name, hook_function) tuples
    fwd_hooks = []

    # Ablate from residual stream at start of each block
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_pre"
        fwd_hooks.append((hook_name, get_direction_ablation_hook(direction)))

    # Return (fwd_hooks, empty_list) for API compatibility
    # The original code expects (fwd_pre_hooks, fwd_hooks) but in TransformerLens
    # we handle everything through run_with_hooks which takes a single list
    return fwd_hooks, []


def get_activation_addition_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
    coeff: float,
    layer: int,
) -> Tuple[List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
    """
    Generate hook specification for adding a direction at a specific layer.

    Activation addition is applied only at the residual stream of a single layer.

    Args:
        model_base: The model instance
        direction: The direction vector to add
        coeff: Coefficient for scaling (negative to subtract)
        layer: Which layer to apply the intervention

    Returns:
        Tuple of (fwd_hooks list, empty list) for compatibility with original API
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    fwd_hooks = [(hook_name, get_activation_addition_hook(direction, coeff))]

    # Return (fwd_hooks, empty_list) for API compatibility
    return fwd_hooks, []


def convert_hooks_to_transformerlens_format(
    fwd_pre_hooks: List[Tuple],
    fwd_hooks: List[Tuple]
) -> List[Tuple[str, Callable]]:
    """
    Convert legacy hook format to TransformerLens format.

    The legacy code uses (module, hook_fn) tuples, while TransformerLens uses
    (hook_name_string, hook_fn) tuples.

    This function is a helper for gradual migration.

    Args:
        fwd_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
        fwd_hooks: List of (module, hook_fn) tuples for forward hooks

    Returns:
        List of (hook_name, hook_fn) tuples for TransformerLens
    """
    # For now, we assume the hooks are already in TransformerLens format
    # This function can be extended if we need to support gradual migration
    return fwd_pre_hooks + fwd_hooks


def prepare_hooks_for_run_with_hooks(fwd_hooks: List[Tuple[str, Callable]]) -> Dict[str, Callable]:
    """
    Convert a list of (hook_name, hook_fn) tuples to a dictionary.

    TransformerLens's run_with_hooks can accept either:
    - A list of (hook_name, hook_fn) tuples
    - A dictionary mapping hook_name -> hook_fn

    This helper creates the dictionary format.

    Args:
        fwd_hooks: List of (hook_name, hook_fn) tuples

    Returns:
        Dictionary mapping hook names to hook functions
    """
    return {hook_name: hook_fn for hook_name, hook_fn in fwd_hooks}
