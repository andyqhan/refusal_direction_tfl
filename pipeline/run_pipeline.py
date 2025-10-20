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

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing (default: 1)')
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

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    console.print("\n")
    console.print(Panel.fit("🧭 Step 3: Generating Candidate Refusal Directions", style="bold cyan"))

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    console.print(f"[cyan]Computing activation differences between {len(harmful_train)} harmful and {len(harmless_train)} harmless examples...[/cyan]")
    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        batch_size=cfg.batch_size)

    save_path = os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt')
    torch.save(mean_diffs, save_path)
    console.print(f"[green]✓[/green] Generated candidate directions with shape [bold]{mean_diffs.shape}[/bold]")
    console.print(f"[green]✓[/green] Saved to [dim]{save_path}[/dim]")

    return mean_diffs

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

def run_pipeline(model_path, batch_size=1):
    """Run the full pipeline."""
    console.print("\n")
    console.print(Panel.fit(
        f"[bold white]REFUSAL DIRECTION PIPELINE[/bold white]\n[cyan]{model_path}[/cyan]",
        border_style="bold magenta"
    ))

    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, batch_size=batch_size)

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_row("[bold]Model alias:[/bold]", f"[yellow]{model_alias}[/yellow]")
    info_table.add_row("[bold]Batch size:[/bold]", f"[yellow]{cfg.batch_size}[/yellow]")
    info_table.add_row("[bold]Artifacts path:[/bold]", f"[dim]{cfg.artifact_path()}[/dim]")
    console.print(info_table)

    with console.status("[cyan]Loading model..."):
        model_base = construct_model_base(cfg.model_path)
    console.print("[green]✓[/green] Model loaded")

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)

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
    console.print("\n")
    console.print(Panel.fit("📈 Step 6: Evaluating Harmful Dataset Completions", style="bold cyan"))
    for dataset_name in cfg.evaluation_datasets:
        console.print(f"\n[yellow]Evaluating dataset:[/yellow] [bold]{dataset_name}[/bold]")
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

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
    console.print("\n")
    console.print(Panel.fit("📊 Step 8: Evaluating Harmless Dataset Completions", style="bold cyan"))
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # 5. Evaluate loss on harmless datasets
    console.print("\n")
    console.print(Panel.fit("📉 Step 9: Evaluating Cross-Entropy Loss", style="bold cyan"))
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]✨ PIPELINE COMPLETE! ✨[/bold green]\n\n[dim]All results saved to:\n{cfg.artifact_path()}[/dim]",
        border_style="bold green"
    ))

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, batch_size=args.batch_size)
