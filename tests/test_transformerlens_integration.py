"""
Unit tests for TransformerLens integration.

These tests verify that the TransformerLens implementation works correctly
without requiring large models or extensive computation.
"""

try:
    import pytest
except ImportError:
    pytest = None

import torch
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.transformerlens_hook_utils import (
    get_direction_ablation_hook,
    get_activation_addition_hook,
    get_all_direction_ablation_hooks,
    get_activation_addition_hooks
)

# Use a tiny test model for fast tests - GPT-2 is small, public, and needs no auth
TEST_MODEL = "gpt2"


def test_model_loading():
    """Test that TransformerLensModel loads correctly."""
    model_base = construct_model_base(TEST_MODEL)

    assert model_base is not None
    assert hasattr(model_base.model, 'run_with_cache')
    assert hasattr(model_base.model, 'run_with_hooks')
    assert model_base.model.cfg.d_model == 768  # GPT-2 hidden size
    print("✓ Model loading test passed")


def test_direction_ablation_hook():
    """Test direction ablation hook creation and application."""
    direction = torch.randn(768)  # GPT-2 hidden size
    hook_fn = get_direction_ablation_hook(direction)

    # Test hook on dummy activation
    activation = torch.randn(2, 10, 768)  # (batch, seq, d_model)

    # Create a mock hook object
    from unittest.mock import MagicMock
    mock_hook = MagicMock()

    result = hook_fn(activation, mock_hook)

    assert result.shape == activation.shape
    assert not torch.allclose(result, activation)  # Should be modified

    # Verify direction was projected out (approximately)
    direction_normalized = direction / direction.norm()
    projection = (result @ direction_normalized).abs().mean()
    assert projection < (activation @ direction_normalized).abs().mean()

    print("✓ Direction ablation hook test passed")


def test_activation_addition_hook():
    """Test activation addition hook creation and application."""
    vector = torch.randn(768)
    coeff = 1.5
    hook_fn = get_activation_addition_hook(vector, coeff)

    activation = torch.randn(2, 10, 768)

    from unittest.mock import MagicMock
    mock_hook = MagicMock()

    result = hook_fn(activation, mock_hook)

    assert result.shape == activation.shape
    assert not torch.allclose(result, activation)

    # Verify the vector was added with correct coefficient
    # result ≈ activation + coeff * vector
    expected = activation + coeff * vector
    assert torch.allclose(result, expected, atol=1e-5)

    print("✓ Activation addition hook test passed")


def test_tokenization():
    """Test chat template formatting and tokenization."""
    model_base = construct_model_base(TEST_MODEL)

    instructions = ["Hello, how are you?", "What is 2+2?"]
    result = model_base.tokenize_instructions_fn(instructions=instructions)

    assert 'input_ids' in result
    assert 'attention_mask' in result
    assert result.input_ids.shape[0] == 2  # Batch size
    assert result.input_ids.shape[1] > 0  # Sequence length

    print("✓ Tokenization test passed")


def test_all_ablation_hooks_creation():
    """Test creating ablation hooks for all layers."""
    model_base = construct_model_base(TEST_MODEL)
    direction = torch.randn(model_base.model.cfg.d_model)

    fwd_hooks, fwd_pre_hooks = get_all_direction_ablation_hooks(model_base, direction)

    # Should create hooks for: resid_pre + attn_result + mlp_out for each layer
    n_layers = model_base.model.cfg.n_layers
    expected_hooks = n_layers * 3  # resid_pre, attn, mlp for each layer

    assert len(fwd_hooks) == expected_hooks
    assert len(fwd_pre_hooks) == 0  # All hooks are forward hooks in TransformerLens

    # Verify hook names are correct
    hook_names = [name for name, _ in fwd_hooks]
    assert any('hook_resid_pre' in name for name in hook_names)
    assert any('attn.hook_result' in name for name in hook_names)
    assert any('hook_mlp_out' in name for name in hook_names)

    print("✓ All ablation hooks creation test passed")


def test_forward_pass_with_hooks():
    """Test forward pass with hooks applied."""
    model_base = construct_model_base(TEST_MODEL)

    # Create a dummy direction
    direction = torch.randn(model_base.model.cfg.d_model)

    # Get hooks
    fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, direction)

    # Tokenize input
    instructions = ["Test input"]
    inputs = model_base.tokenize_instructions_fn(instructions=instructions)

    # Run with hooks
    logits = model_base.model.run_with_hooks(
        inputs.input_ids.to(model_base.model.cfg.device),
        fwd_hooks=fwd_hooks,
        prepend_bos=False,
    )

    assert logits is not None
    assert logits.shape[0] == 1  # Batch size
    assert logits.shape[-1] == model_base.model.cfg.d_vocab  # Vocab size

    print("✓ Forward pass with hooks test passed")


def test_activation_addition_hooks_creation():
    """Test creating activation addition hooks for a specific layer."""
    model_base = construct_model_base(TEST_MODEL)
    direction = torch.randn(model_base.model.cfg.d_model)
    coeff = -0.5
    layer = 5

    fwd_hooks, fwd_pre_hooks = get_activation_addition_hooks(model_base, direction, coeff, layer)

    # Should create one hook for the specified layer
    assert len(fwd_hooks) == 1
    assert len(fwd_pre_hooks) == 0

    # Verify hook name includes the layer
    hook_name, _ = fwd_hooks[0]
    assert f"blocks.{layer}" in hook_name
    assert "hook_resid_pre" in hook_name

    print("✓ Activation addition hooks creation test passed")


if __name__ == "__main__":
    # Run tests manually if pytest is not installed
    print("Running TransformerLens integration tests...\n")

    try:
        test_model_loading()
        test_direction_ablation_hook()
        test_activation_addition_hook()
        test_tokenization()
        test_all_ablation_hooks_creation()
        test_forward_pass_with_hooks()
        test_activation_addition_hooks_creation()

        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
