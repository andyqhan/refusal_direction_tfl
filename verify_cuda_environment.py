"""Utility script to sanity-check CUDA (or MPS) compatibility for the refusal direction pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from rich.console import Console
from rich.table import Table

# Ensure project root is importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.transformerlens_hook_utils import get_activation_addition_hooks


console = Console()


def detect_preferred_device() -> str:
    """Pick the device the pipeline will use."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_device_summary() -> Table:
    """Build a table summarising Torch + device availability."""
    table = Table(title="Torch Device Summary")
    table.add_column("Capability", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="magenta")

    cuda_available = torch.cuda.is_available()
    table.add_row("CUDA available", "✅" if cuda_available else "❌", str(cuda_available))
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        table.add_row("CUDA device", "ℹ️", props.name)
        table.add_row("Total memory (GB)", "ℹ️", f"{props.total_memory / (1024 ** 3):.1f}")
        table.add_row("CUDA version", "ℹ️", str(torch.version.cuda))
    else:
        table.add_row("CUDA device", "ℹ️", "N/A")
        table.add_row("CUDA version", "ℹ️", str(torch.version.cuda))

    mps_available = torch.backends.mps.is_available()
    table.add_row("MPS available", "✅" if mps_available else "❌", str(mps_available))

    table.add_row("Torch version", "ℹ️", torch.__version__)
    table.add_row("Preferred device", "ℹ️", detect_preferred_device())

    return table


def run_minimal_generation(
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    num_prompts: int,
    use_hooks: bool,
) -> None:
    """Load the model and run a short generation pass to shake out device issues."""
    console.print(f"[bold]Loading model:[/bold] [yellow]{model_path}[/yellow]")
    model_base = construct_model_base(model_path)

    preferred = detect_preferred_device()
    actual_device = model_base.model.cfg.device
    if preferred != actual_device:
        console.print(
            f"[yellow]⚠ Device preference mismatch: expected {preferred}, model reports {actual_device}[/yellow]"
        )
    else:
        console.print(f"[green]✓ Model reports device '{actual_device}'[/green]")

    prompts = [
        "List three use cases for responsible AI safety evaluations.",
        "Summarise how activation steering works in transformer models.",
        "What are common pitfalls when porting research pipelines to CUDA?",
        "Explain why modal cloud GPUs can accelerate benchmarking workloads.",
    ]
    prompts = (prompts * ((num_prompts + len(prompts) - 1) // len(prompts)))[:num_prompts]
    dataset = [{"instruction": p, "category": "cuda-check"} for p in prompts]

    hook_label = "with hooks" if use_hooks else "baseline"
    console.print(f"[bold]Running generation ({hook_label})…[/bold]")
    start = time.time()

    fwd_pre_hooks: List = []
    fwd_hooks: List = []
    if use_hooks:
        dummy_direction = torch.randn(model_base.model.cfg.d_model, device=model_base.model.cfg.device, dtype=model_base.dtype)
        dummy_direction = dummy_direction / (dummy_direction.norm() + 1e-8)
        fwd_hooks, _ = get_activation_addition_hooks(model_base, dummy_direction, coeff=-0.1, layer=model_base.model.cfg.n_layers // 2)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    elapsed = time.time() - start
    total_tokens = sum(len(model_base.tokenizer.encode(item["response"])) for item in completions)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    console.print(
        f"[green]✓ Completed generation in {elapsed:.2f}s ({total_tokens} tokens, {throughput:.1f} tok/s)[/green]"
    )
    sample = completions[0]["response"] if completions else "<no completions>"
    console.print(f"[dim]Sample completion:[/dim] {sample[:200]}{'…' if len(sample) > 200 else ''}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CUDA/MPS compatibility for the refusal direction pipeline.")
    parser.add_argument("--model-path", default="google/gemma-2b-it", help="HF model identifier to load for the smoke test.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the generation smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens for the smoke test generation.")
    parser.add_argument("--num-prompts", type=int, default=4, help="How many prompts to generate during the smoke test.")
    parser.add_argument("--skip-generation", action="store_true", help="Only report device info without running a generation test.")
    parser.add_argument("--with-hooks", action="store_true", help="Include a dummy activation-addition hook to exercise hook code paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(format_device_summary())

    preferred_device = detect_preferred_device()
    if preferred_device == "cpu":
        console.print(
            "[yellow]⚠ Neither CUDA nor MPS is available. The pipeline will run on CPU and may be extremely slow.[/yellow]"
        )

    if torch.cuda.is_available():
        # Warm up CUDA to catch driver issues early.
        console.print("[bold]Warming up CUDA…[/bold]")
        _ = torch.rand(1, device="cuda")
        torch.cuda.synchronize()
        console.print("[green]✓ CUDA warm-up succeeded[/green]")

    if not args.skip_generation:
        run_minimal_generation(
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_prompts=args.num_prompts,
            use_hooks=args.with_hooks,
        )
    else:
        console.print("[cyan]Skipping generation test as requested.[/cyan]")


if __name__ == "__main__":
    main()
