"""
Performance test script using the EXACT same setup as the pipeline.
"""

import torch
import time
from rich.console import Console
from rich.table import Table

# Import the actual model class used in the pipeline
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.transformerlens_hook_utils import get_all_direction_ablation_hooks

console = Console()

def test_generation_speed(model_path="google/gemma-2b-it", n_prompts=4, max_new_tokens=64):
    """Test generation speed using the exact pipeline setup."""

    console.print(f"\n[bold cyan]Performance Test: {model_path}[/bold cyan]")
    console.print(f"Prompts: {n_prompts}, Max new tokens: {max_new_tokens}\n")

    # Load model exactly like the pipeline does
    console.print("[bold]Loading model (same as pipeline)...[/bold]")
    model_base = construct_model_base(model_path)

    # Check what device we're actually on
    device = model_base.model.cfg.device
    console.print(f"[green]Model device: {device}[/green]")
    console.print(f"[green]Model dtype: {model_base.dtype}[/green]")
    console.print(f"[green]Number of layers: {model_base.model.cfg.n_layers}[/green]\n")

    # Create test dataset in the format the pipeline uses
    test_dataset = [
        {"instruction": "Write a short poem about cats.", "category": "test"}
        for _ in range(n_prompts)
    ]

    results = {}

    # Test 1: Baseline generation (no hooks)
    console.print("[bold cyan]Test 1: Baseline generation (no hooks)[/bold cyan]")
    start = time.time()

    completions_baseline = model_base.generate_completions(
        test_dataset,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        max_new_tokens=max_new_tokens,
        batch_size=n_prompts  # Do all at once
    )

    elapsed_baseline = time.time() - start

    # Count actual tokens generated
    total_tokens_baseline = sum(
        len(model_base.tokenizer.encode(c['response']))
        for c in completions_baseline
    )
    throughput_baseline = total_tokens_baseline / elapsed_baseline

    console.print(f"[green]✓ Time: {elapsed_baseline:.2f}s")
    console.print(f"[green]✓ Tokens generated: {total_tokens_baseline}")
    console.print(f"[green]✓ Throughput: {throughput_baseline:.1f} tok/s[/green]\n")

    results['baseline'] = {
        'time': elapsed_baseline,
        'tokens': total_tokens_baseline,
        'throughput': throughput_baseline
    }

    # Test 2: Generation with ablation hooks (hooks on all layers)
    console.print("[bold cyan]Test 2: With ablation hooks (all layers)[/bold cyan]")

    # Create a dummy direction for ablation
    dummy_direction = torch.randn(model_base.model.cfg.d_model, device=device, dtype=model_base.dtype)
    dummy_direction = dummy_direction / dummy_direction.norm()

    # Get hooks exactly like the pipeline does
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, dummy_direction)

    console.print(f"[dim]Number of hooks: {len(fwd_pre_hooks) + len(fwd_hooks)}[/dim]")

    start = time.time()

    completions_ablation = model_base.generate_completions(
        test_dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens,
        batch_size=n_prompts
    )

    elapsed_ablation = time.time() - start

    total_tokens_ablation = sum(
        len(model_base.tokenizer.encode(c['response']))
        for c in completions_ablation
    )
    throughput_ablation = total_tokens_ablation / elapsed_ablation

    console.print(f"[green]✓ Time: {elapsed_ablation:.2f}s")
    console.print(f"[green]✓ Tokens generated: {total_tokens_ablation}")
    console.print(f"[green]✓ Throughput: {throughput_ablation:.1f} tok/s[/green]")
    console.print(f"[yellow]⚠ Slowdown vs baseline: {elapsed_ablation/elapsed_baseline:.2f}x[/yellow]\n")

    results['ablation'] = {
        'time': elapsed_ablation,
        'tokens': total_tokens_ablation,
        'throughput': throughput_ablation,
        'slowdown': elapsed_ablation / elapsed_baseline
    }

    # Test 3: Just caching (like direction generation)
    console.print("[bold cyan]Test 3: Activation caching (direction generation)[/bold cyan]")

    # Tokenize
    instructions = [x['instruction'] for x in test_dataset]
    tokenized = model_base.tokenize_instructions_fn(instructions=instructions)
    input_ids = tokenized.input_ids.to(device)

    start = time.time()

    with torch.no_grad():
        _, cache = model_base.model.run_with_cache(input_ids)

    elapsed_cache = time.time() - start

    tokens_processed = input_ids.numel()
    throughput_cache = tokens_processed / elapsed_cache

    console.print(f"[green]✓ Time: {elapsed_cache:.2f}s")
    console.print(f"[green]✓ Tokens processed: {tokens_processed}")
    console.print(f"[green]✓ Throughput: {throughput_cache:.1f} tok/s[/green]\n")

    results['caching'] = {
        'time': elapsed_cache,
        'tokens': tokens_processed,
        'throughput': throughput_cache
    }

    # Summary table
    table = Table(title="Performance Summary")
    table.add_column("Test", style="cyan")
    table.add_column("Time (s)", style="yellow")
    table.add_column("Tokens", style="magenta")
    table.add_column("Throughput", style="green")
    table.add_column("Notes", style="dim")

    table.add_row(
        "Baseline",
        f"{results['baseline']['time']:.2f}",
        str(results['baseline']['tokens']),
        f"{results['baseline']['throughput']:.1f} tok/s",
        "No hooks"
    )

    table.add_row(
        "Ablation",
        f"{results['ablation']['time']:.2f}",
        str(results['ablation']['tokens']),
        f"{results['ablation']['throughput']:.1f} tok/s",
        f"{results['ablation']['slowdown']:.2f}x slower"
    )

    table.add_row(
        "Caching",
        f"{results['caching']['time']:.2f}",
        str(results['caching']['tokens']),
        f"{results['caching']['throughput']:.1f} tok/s",
        "Forward only"
    )

    console.print(table)

    # Analysis
    console.print("\n[bold]Analysis:[/bold]")

    if results['baseline']['throughput'] < 10:
        console.print("[red]⚠ VERY SLOW baseline generation (<10 tok/s)[/red]")
        console.print("[red]  → MPS might not be working properly with TransformerLens[/red]")
    elif results['baseline']['throughput'] < 50:
        console.print("[yellow]⚠ Slow baseline generation (<50 tok/s)[/yellow]")
        console.print("[yellow]  → This might be expected for MPS with larger models[/yellow]")
    else:
        console.print("[green]✓ Good baseline generation speed[/green]")

    if results['ablation']['slowdown'] > 2:
        console.print(f"[red]⚠ Hooks add significant overhead ({results['ablation']['slowdown']:.1f}x)[/red]")
    else:
        console.print(f"[green]✓ Hooks overhead is reasonable ({results['ablation']['slowdown']:.1f}x)[/green]")

    caching_speedup = results['caching']['throughput'] / results['baseline']['throughput']
    console.print(f"[cyan]ℹ Caching is {caching_speedup:.1f}x faster than generation (expected)[/cyan]")

    # Estimate for full pipeline
    console.print("\n[bold]Pipeline Estimate:[/bold]")
    total_completions = 100  # jailbreakbench test set
    interventions = 3  # baseline, ablation, actadd

    # Assume actadd generates full responses, baseline/ablation refuse (shorter)
    # Let's say baseline/ablation average 50 tokens, actadd averages max_new_tokens
    avg_tokens_refusing = 50
    avg_tokens_complying = max_new_tokens

    time_baseline = (total_completions * avg_tokens_refusing) / results['baseline']['throughput']
    time_ablation = (total_completions * avg_tokens_refusing) / results['ablation']['throughput']
    time_actadd = (total_completions * avg_tokens_complying) / results['ablation']['throughput']  # actadd uses hooks too

    total_estimated = time_baseline + time_ablation + time_actadd

    console.print(f"Estimated time for 100 examples × 3 interventions:")
    console.print(f"  Baseline: {time_baseline/60:.1f} min")
    console.print(f"  Ablation: {time_ablation/60:.1f} min")
    console.print(f"  Actadd: {time_actadd/60:.1f} min (assumes full {max_new_tokens} tok responses)")
    console.print(f"  [bold]Total: ~{total_estimated/60:.1f} minutes[/bold]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google/gemma-2b-it")
    parser.add_argument("--n_prompts", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    test_generation_speed(args.model_path, args.n_prompts, args.max_new_tokens)
