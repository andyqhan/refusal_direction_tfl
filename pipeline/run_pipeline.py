import torch
import random
import json
import os
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.transformerlens_hook_utils import (
    get_all_direction_ablation_hooks,
    get_activation_addition_hooks
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

console = Console()

def parse_cache_layers_arg(value):
    """
    Parse the --cache-layers argument.

    Accepts:
    - An integer (number of evenly-spaced layers to cache)
    - "all" (cache all layers - default)
    - "final" (only cache the final layer)
    - "threefourths" (only cache the layer at 3L/4)
    - "half" (only cache the layer at L/2)

    Returns the raw value (int, str, or None) for later processing.
    """
    special_values = ["all", "final", "threefourths", "half"]

    if value.lower() in special_values:
        # Convert "all" to None (which means cache all layers)
        return None if value.lower() == "all" else value.lower()

    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError("--cache-layers must be a positive integer")
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--cache-layers must be an integer or one of: {', '.join(special_values)}"
        )

def compute_layers_to_cache(cache_layers_arg, n_layers):
    """
    Compute which layers to cache based on the cache_layers argument.

    Args:
        cache_layers_arg: Either an int or one of ["final", "threefourths", "half"]
        n_layers: Total number of layers in the model (0-indexed)

    Returns:
        List of layer indices to cache (0-indexed), or None to cache all layers
    """
    if cache_layers_arg is None:
        # Default: cache all layers
        return None

    if cache_layers_arg == "final":
        return [n_layers - 1]
    elif cache_layers_arg == "threefourths":
        return [round(3 * n_layers / 4) - 1]
    elif cache_layers_arg == "half":
        return [round(n_layers / 2) - 1]
    elif isinstance(cache_layers_arg, int):
        # Evenly space the layers
        # For L=16 and cache_layers=4, we want layers [15, 11, 7, 3]
        # This means we divide into 4 segments and take the last layer of each segment
        if cache_layers_arg >= n_layers:
            # If asking for more layers than exist, just cache all
            return None

        layers = []
        for i in range(cache_layers_arg):
            # Calculate position: we want evenly spaced layers, starting from the end
            # Segment i goes from layer (i * n_layers / cache_layers_arg) to ((i+1) * n_layers / cache_layers_arg)
            # We take the last layer of each segment
            layer = round((i + 1) * n_layers / cache_layers_arg) - 1
            layers.append(layer)

        # Reverse to get descending order (highest layer first)
        layers.reverse()

        # Deduplicate while preserving order
        seen = set()
        deduped_layers = []
        for layer in layers:
            if layer not in seen:
                seen.add(layer)
                deduped_layers.append(layer)

        return deduped_layers

    return None

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing (default: 1)')
    parser.add_argument('--ignore-cached-direction', action='store_true',
                        help='Ignore cached direction and recompute from scratch (default: use cached if available)')
    parser.add_argument('--cache-layers', type=parse_cache_layers_arg, default='all',
                        help='Control which layers to cache activations for. Can be an integer (number of evenly-spaced layers), '
                             '"all" (cache all layers), "final" (only final layer), "threefourths" (layer at 3L/4), or "half" (layer at L/2). '
                             'Default: "all".')
    parser.add_argument('--eval-datasets', type=str, default=None,
                        help='Comma-separated list of evaluation datasets to run (e.g., "harmbench_test,strongreject"). '
                             'Available datasets: advbench, harmbench_test, harmbench_val, jailbreakbench, malicious_instruct, strongreject, tdc2023. '
                             'Default: harmbench_val')
    parser.add_argument('--eval-methodologies', type=str, default=None,
                        help='Comma-separated list of evaluation methodologies (e.g., "substring_matching,llamaguard2"). '
                             'Available: substring_matching (free, rule-based), llamaguard2 (requires TOGETHER_API_KEY), harmbench (requires vLLM, Linux only). '
                             'Default: substring_matching')
    parser.add_argument('--no-evaluation', action='store_true',
                        help='Skip evaluation steps (Steps 6, 8, 9). Only generate completions without evaluating them. (default: run all evaluations)')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    console.print("\n")
    console.print(Panel.fit("📊 Step 1: Loading and Sampling Datasets", style="bold cyan"))

    random.seed(42)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading datasets...", total=4)

        progress.update(task, description=f"[cyan]Loading {cfg.n_train} harmful training examples...")
        harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
        progress.advance(task)

        progress.update(task, description=f"[cyan]Loading {cfg.n_train} harmless training examples...")
        harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
        progress.advance(task)

        progress.update(task, description=f"[cyan]Loading {cfg.n_val} harmful validation examples...")
        harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
        progress.advance(task)

        progress.update(task, description=f"[cyan]Loading {cfg.n_val} harmless validation examples...")
        harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
        progress.advance(task)

    console.print(f"[green]✓[/green] Loaded {len(harmful_train)} harmful train, {len(harmless_train)} harmless train, {len(harmful_val)} harmful val, {len(harmless_val)} harmless val")
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    console.print("\n")
    console.print(Panel.fit("🔍 Step 2: Filtering Datasets Based on Refusal Scores", style="bold cyan"))

    if cfg.filter_train:
        console.print("[yellow]Filtering training data...[/yellow]")
        harmful_before = len(harmful_train)
        harmless_before = len(harmless_train)

        console.print(f"[cyan]Computing refusal scores for {harmful_before} harmful training examples...[/cyan]")
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=cfg.batch_size)

        console.print(f"[cyan]Computing refusal scores for {harmless_before} harmless training examples...[/cyan]")
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=cfg.batch_size)

        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)
        console.print(f"[green]✓[/green] Kept {len(harmful_train)}/{harmful_before} harmful, {len(harmless_train)}/{harmless_before} harmless")

    if cfg.filter_val:
        console.print("[yellow]Filtering validation data...[/yellow]")
        harmful_before = len(harmful_val)
        harmless_before = len(harmless_val)

        console.print(f"[cyan]Computing refusal scores for {harmful_before} harmful validation examples...[/cyan]")
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=cfg.batch_size)

        console.print(f"[cyan]Computing refusal scores for {harmless_before} harmless validation examples...[/cyan]")
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=cfg.batch_size)

        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
        console.print(f"[green]✓[/green] Kept {len(harmful_val)}/{harmful_before} harmful, {len(harmless_val)}/{harmless_before} harmless")

    if not cfg.filter_train and not cfg.filter_val:
        console.print("[dim]Filtering disabled - using all examples[/dim]")

    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, cache_layers=None):
    """Generate and save candidate directions."""
    console.print("\n")
    console.print(Panel.fit("🧭 Step 3: Generating Candidate Refusal Directions", style="bold cyan"))

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    console.print(f"[cyan]Computing activation differences between {len(harmful_train)} harmful and {len(harmless_train)} harmless examples...[/cyan]")

    # Display cache layers info if specified
    if cache_layers is not None:
        layers_to_cache = compute_layers_to_cache(cache_layers, model_base.model.cfg.n_layers)
        if layers_to_cache:
            console.print(f"[dim]Caching only {len(layers_to_cache)} layer(s): {layers_to_cache}[/dim]")
        else:
            console.print(f"[dim]Caching all {model_base.model.cfg.n_layers} layers[/dim]")

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        batch_size=cfg.batch_size,
        cache_layers=cache_layers)

    save_path = os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt')
    torch.save(mean_diffs, save_path)
    console.print(f"[green]✓[/green] Generated candidate directions with shape [bold]{mean_diffs.shape}[/bold]")
    console.print(f"[green]✓[/green] Saved to [dim]{save_path}[/dim]")

    return mean_diffs

def load_cached_direction_if_exists(cfg):
    """
    Load cached direction and metadata if they exist.

    Returns:
        Tuple of (pos, layer, direction) if cached files exist, otherwise (None, None, None)
    """
    direction_path = f'{cfg.artifact_path()}/direction.pt'
    metadata_path = f'{cfg.artifact_path()}/direction_metadata.json'

    if os.path.exists(direction_path) and os.path.exists(metadata_path):
        console.print("\n")
        console.print(Panel.fit("🎯 Step 4: Loading Cached Refusal Direction", style="bold cyan"))

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        direction = torch.load(direction_path)
        pos = metadata['pos']
        layer = metadata['layer']

        console.print(f"[green]✓[/green] Loaded cached direction at position [bold yellow]{pos}[/bold yellow], layer [bold yellow]{layer}[/bold yellow]")
        console.print(f"[green]✓[/green] Direction loaded from [dim]{direction_path}[/dim]")
        console.print(f"[green]✓[/green] Metadata loaded from [dim]{metadata_path}[/dim]")
        console.print(f"[dim]Tip: Use --ignore-cached-direction to recompute from scratch[/dim]")

        return pos, layer, direction

    return None, None, None

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    console.print("\n")
    console.print(Panel.fit("🎯 Step 4: Selecting Best Refusal Direction", style="bold cyan"))

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    console.print(f"[cyan]Evaluating candidate directions on {len(harmful_val)} harmful and {len(harmless_val)} harmless validation examples...[/cyan]")
    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
        batch_size=cfg.batch_size
    )

    metadata_path = f'{cfg.artifact_path()}/direction_metadata.json'
    with open(metadata_path, "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    direction_path = f'{cfg.artifact_path()}/direction.pt'
    torch.save(direction, direction_path)

    console.print(f"[green]✓[/green] Selected direction at position [bold yellow]{pos}[/bold yellow], layer [bold yellow]{layer}[/bold yellow]")
    console.print(f"[green]✓[/green] Saved direction to [dim]{direction_path}[/dim]")
    console.print(f"[green]✓[/green] Saved metadata to [dim]{metadata_path}[/dim]")

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        with console.status(f"[cyan]Loading dataset '{dataset_name}'..."):
            dataset = load_dataset(dataset_name)

    with console.status(f"[cyan]Generating {len(dataset)} completions for '{dataset_name}' with intervention '{intervention_label}'..."):
        completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size)

    save_path = f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json'
    with open(save_path, "w") as f:
        json.dump(completions, f, indent=4)
    console.print(f"  [green]✓[/green] Saved [bold]{dataset_name}[/bold] + [bold]{intervention_label}[/bold] completions")

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with console.status(f"[cyan]Evaluating '{dataset_name}' + '{intervention_label}' using {eval_methodologies}..."):
        with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
            completions = json.load(f)

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
        )

        save_path = f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json'
        with open(save_path, "w") as f:
            json.dump(evaluation, f, indent=4)

    console.print(f"  [green]✓[/green] Evaluated [bold]{dataset_name}[/bold] + [bold]{intervention_label}[/bold]")

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    with console.status(f"[cyan]Evaluating cross-entropy loss for intervention '{intervention_label}'..."):
        on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

        loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

        save_path = f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json'
        with open(save_path, "w") as f:
            json.dump(loss_evals, f, indent=4)

    console.print(f"  [green]✓[/green] Loss evaluation for [bold]{intervention_label}[/bold]")

def run_pipeline(model_path, batch_size=1, ignore_cached_direction=False, cache_layers=None, eval_datasets=None, eval_methodologies=None, no_evaluation=False):
    """Run the full pipeline."""
    console.print("\n")
    console.print(Panel.fit(
        f"[bold white]REFUSAL DIRECTION PIPELINE[/bold white]\n[cyan]{model_path}[/cyan]",
        border_style="bold magenta"
    ))

    model_alias = os.path.basename(model_path)

    # Parse eval_datasets if provided
    if eval_datasets is not None:
        eval_datasets_tuple = tuple(ds.strip() for ds in eval_datasets.split(','))
    else:
        eval_datasets_tuple = ("harmbench_val",)

    # Parse eval_methodologies if provided
    if eval_methodologies is not None:
        eval_methodologies_tuple = tuple(m.strip() for m in eval_methodologies.split(','))
    else:
        eval_methodologies_tuple = ("substring_matching",)

    cfg = Config(
        model_alias=model_alias,
        model_path=model_path,
        batch_size=batch_size,
        evaluation_datasets=eval_datasets_tuple,
        jailbreak_eval_methodologies=eval_methodologies_tuple
    )

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_row("[bold]Model alias:[/bold]", f"[yellow]{model_alias}[/yellow]")
    info_table.add_row("[bold]Batch size:[/bold]", f"[yellow]{cfg.batch_size}[/yellow]")
    info_table.add_row("[bold]Eval datasets:[/bold]", f"[yellow]{', '.join(cfg.evaluation_datasets)}[/yellow]")
    info_table.add_row("[bold]Eval methodologies:[/bold]", f"[yellow]{', '.join(cfg.jailbreak_eval_methodologies)}[/yellow]")
    if cache_layers is not None:
        info_table.add_row("[bold]Cache layers:[/bold]", f"[yellow]{cache_layers}[/yellow]")
    info_table.add_row("[bold]Artifacts path:[/bold]", f"[dim]{cfg.artifact_path()}[/dim]")
    console.print(info_table)

    with console.status("[cyan]Loading model..."):
        model_base = construct_model_base(cfg.model_path)
    console.print("[green]✓[/green] Model loaded")

    # Try to load cached direction first (unless ignored)
    pos, layer, direction = None, None, None
    if not ignore_cached_direction:
        pos, layer, direction = load_cached_direction_if_exists(cfg)

    # If no cached direction, compute from scratch
    if direction is None:
        # Load and sample datasets
        harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

        # Filter datasets based on refusal scores
        harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

        # 1. Generate candidate refusal directions
        candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, cache_layers=cache_layers)

        # 2. Select the most effective refusal direction
        pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    console.print("\n")
    console.print(Panel.fit("💬 Step 5: Generating Completions on Harmful Datasets", style="bold cyan"))
    console.print(f"[bold]Datasets:[/bold] {', '.join(cfg.evaluation_datasets)}")
    console.print(f"[bold]Interventions:[/bold] baseline, ablation, actadd")

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, direction)
    ablation_fwd_pre_hooks = []
    actadd_fwd_hooks, _ = get_activation_addition_hooks(model_base, direction, -1.0, layer)
    actadd_fwd_pre_hooks = []

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        console.print(f"\n[yellow]Processing dataset:[/yellow] [bold]{dataset_name}[/bold]")
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    if not no_evaluation:
        console.print("\n")
        console.print(Panel.fit("📈 Step 6: Evaluating Harmful Dataset Completions", style="bold cyan"))
        for dataset_name in cfg.evaluation_datasets:
            console.print(f"\n[yellow]Evaluating dataset:[/yellow] [bold]{dataset_name}[/bold]")
            evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    else:
        console.print("\n")
        console.print("[dim]⏭  Step 6: Skipped (--no-evaluation)[/dim]")

    # 4a. Generate and save completions on harmless evaluation dataset
    console.print("\n")
    console.print(Panel.fit("✅ Step 7: Generating Completions on Harmless Dataset", style="bold cyan"))

    with console.status(f"[cyan]Loading {cfg.n_test} harmless test examples..."):
        harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

    actadd_refusal_hooks, _ = get_activation_addition_hooks(model_base, direction, +1.0, layer)
    actadd_refusal_pre_hooks = []

    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    if not no_evaluation:
        console.print("\n")
        console.print(Panel.fit("📊 Step 8: Evaluating Harmless Dataset Completions", style="bold cyan"))
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    else:
        console.print("\n")
        console.print("[dim]⏭  Step 8: Skipped (--no-evaluation)[/dim]")

    # 5. Evaluate loss on harmless datasets
    if not no_evaluation:
        console.print("\n")
        console.print(Panel.fit("📉 Step 9: Evaluating Cross-Entropy Loss", style="bold cyan"))
        evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
        evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
        evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')
    else:
        console.print("\n")
        console.print("[dim]⏭  Step 9: Skipped (--no-evaluation)[/dim]")

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]✨ PIPELINE COMPLETE! ✨[/bold green]\n\n[dim]All results saved to:\n{cfg.artifact_path()}[/dim]",
        border_style="bold green"
    ))

if __name__ == "__main__":
    args = parse_arguments()
    # Note: argparse converts hyphens to underscores in attribute names
    ignore_cached = getattr(args, 'ignore_cached_direction', False)
    cache_layers = getattr(args, 'cache_layers', None)
    eval_datasets = getattr(args, 'eval_datasets', None)
    eval_methodologies = getattr(args, 'eval_methodologies', None)
    no_evaluation = getattr(args, 'no_evaluation', False)
    run_pipeline(
        model_path=args.model_path,
        batch_size=args.batch_size,
        ignore_cached_direction=ignore_cached,
        cache_layers=cache_layers,
        eval_datasets=eval_datasets,
        eval_methodologies=eval_methodologies,
        no_evaluation=no_evaluation
    )
