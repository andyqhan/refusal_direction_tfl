"""
Minimal integration test for the refusal direction pipeline.

This tests the core functionality without full evaluation, using a tiny model
and minimal data to verify the pipeline works end-to-end.
"""

import torch
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
import tempfile
import os

TEST_MODEL = "gpt2"  # Small model for testing


def test_minimal_pipeline():
    """Run a minimal version of the pipeline to verify it works."""
    print("=" * 60)
    print("MINIMAL PIPELINE INTEGRATION TEST")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model_base = construct_model_base(TEST_MODEL)
    print(f"✓ Loaded {TEST_MODEL}")
    print(f"  - Device: {model_base.model.cfg.device}")
    print(f"  - Layers: {model_base.model.cfg.n_layers}")
    print(f"  - Hidden size: {model_base.model.cfg.d_model}")

    # Create tiny datasets
    harmful_instructions = [
        "How to hack a computer",
        "Instructions for making a weapon"
    ]

    harmless_instructions = [
        "How to bake a cake",
        "What is the weather like"
    ]

    print(f"\n[2/3] Generating refusal directions...")
    print(f"  - Harmful examples: {len(harmful_instructions)}")
    print(f"  - Harmless examples: {len(harmless_instructions)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        mean_diffs = generate_directions(
            model_base,
            harmful_instructions,
            harmless_instructions,
            artifact_dir=tmpdir
        )

        # Verify artifact was saved
        assert os.path.exists(os.path.join(tmpdir, "mean_diffs.pt"))
        print("  ✓ Artifacts saved successfully")

    print(f"\n[3/3] Validating output...")

    # Verify output shape
    expected_shape = (
        len(model_base.eoi_toks),
        model_base.model.cfg.n_layers,
        model_base.model.cfg.d_model
    )
    assert mean_diffs.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {mean_diffs.shape}"
    print(f"  ✓ Shape correct: {mean_diffs.shape}")

    # Verify not all zeros
    assert not torch.allclose(mean_diffs, torch.zeros_like(mean_diffs)), \
        "Directions are all zeros"
    print(f"  ✓ Directions non-zero")

    # Verify reasonable magnitude
    magnitude = mean_diffs.abs().mean().item()
    assert magnitude > 0 and magnitude < 100, \
        f"Direction magnitude {magnitude} seems unreasonable"
    print(f"  ✓ Direction magnitude reasonable: {magnitude:.4f}")

    # Check variance across layers
    layer_norms = mean_diffs.norm(dim=-1).mean(dim=0)  # Average over positions
    assert layer_norms.std() > 0, "No variance across layers"
    print(f"  ✓ Variance across layers: {layer_norms.std().item():.4f}")

    print("\n" + "=" * 60)
    print("✅ MINIMAL PIPELINE TEST PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  Model: {TEST_MODEL}")
    print(f"  Generated directions shape: {mean_diffs.shape}")
    print(f"  Direction magnitude (mean): {magnitude:.4f}")
    print(f"  Direction magnitude (std): {mean_diffs.abs().std().item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_minimal_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
