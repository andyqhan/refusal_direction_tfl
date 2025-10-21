"""Resume pipeline after harmful completions have been generated."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel

from dataset.load_dataset import load_dataset_split
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import (
    console,
    evaluate_completions_and_save_results_for_dataset,
    evaluate_loss_for_datasets,
    generate_and_save_completions_for_dataset,
    load_cached_direction_if_exists,
)
from pipeline.utils.transformerlens_hook_utils import (
    get_activation_addition_hooks,
    get_all_direction_ablation_hooks,
)

LOCAL_CONSOLE = console if console is not None else Console()


def _missing_completions(cfg: Config) -> Dict[str, List[str]]:
    """Return mapping of dataset -> missing interventions."""
    missing: Dict[str, List[str]] = {}
    base = Path(cfg.artifact_path()) / "completions"
    for dataset_name in cfg.evaluation_datasets:
        missing_interventions: List[str] = []
        for intervention in ("baseline", "ablation", "actadd"):
            path = base / f"{dataset_name}_{intervention}_completions.json"
            if not path.exists():
                missing_interventions.append(intervention)
        if missing_interventions:
            missing[dataset_name] = missing_interventions
    return missing


def resume_pipeline(
    model_path: str,
    batch_size: int = 4,
    cache_layers: Optional[str] = None,
) -> None:
    """Resume pipeline after harmful completions have already been generated."""
    LOCAL_CONSOLE.print("\n")
    LOCAL_CONSOLE.print(Panel.fit(
        f"[bold white]REFUSAL DIRECTION PIPELINE (resume)[/bold white]\n[cyan]{model_path}[/cyan]",
        border_style="bold yellow"
    ))

    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, batch_size=batch_size)

    # Load direction and metadata that should already exist.
    pos, layer, direction = load_cached_direction_if_exists(cfg)
    if direction is None:
        raise RuntimeError(
            "No cached direction found. Run the full pipeline once before using the resume helper."
        )

    LOCAL_CONSOLE.print("\n[yellow]Skipping Step 5: using existing harmful completions.[/yellow]")

    missing = _missing_completions(cfg)
    if missing:
        LOCAL_CONSOLE.print("\n")
        LOCAL_CONSOLE.print(Panel.fit("⚠️ Missing harmful completions detected", style="bold yellow"))
        for dataset_name, interventions in missing.items():
            interventions_str = ", ".join(interventions)
            LOCAL_CONSOLE.print(f"[yellow]- {dataset_name}[/yellow]: regenerate {interventions_str}")
    else:
        LOCAL_CONSOLE.print("[green]✓ All harmful completions already present[/green]")

    # Load the model once (needed for generation/evaluation).
    LOCAL_CONSOLE.print("\n[cyan]Loading model for remaining steps...[/cyan]")
    model_base = construct_model_base(cfg.model_path)
    LOCAL_CONSOLE.print("[green]✓ Model loaded[/green]")

    ablation_fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, direction)
    ablation_fwd_pre_hooks: list = []
    actadd_fwd_hooks, _ = get_activation_addition_hooks(model_base, direction, -1.0, layer)
    actadd_fwd_pre_hooks: list = []

    # Generate any missing harmful completions before evaluation.
    if missing:
        LOCAL_CONSOLE.print("\n")
        LOCAL_CONSOLE.print(Panel.fit("⏩ Regenerating missing harmful completions", style="bold cyan"))
        for dataset_name, interventions in missing.items():
            LOCAL_CONSOLE.print(f"[cyan]Dataset:[/cyan] [bold]{dataset_name}[/bold]")
            if "baseline" in interventions:
                generate_and_save_completions_for_dataset(
                    cfg,
                    model_base,
                    [],
                    [],
                    "baseline",
                    dataset_name,
                )
            if "ablation" in interventions:
                generate_and_save_completions_for_dataset(
                    cfg,
                    model_base,
                    ablation_fwd_pre_hooks,
                    ablation_fwd_hooks,
                    "ablation",
                    dataset_name,
                )
            if "actadd" in interventions:
                generate_and_save_completions_for_dataset(
                    cfg,
                    model_base,
                    actadd_fwd_pre_hooks,
                    actadd_fwd_hooks,
                    "actadd",
                    dataset_name,
                )

    # Step 6: Evaluate harmful dataset completions.
    LOCAL_CONSOLE.print("\n")
    LOCAL_CONSOLE.print(Panel.fit("📈 Step 6 (resume): Evaluating Harmful Dataset Completions", style="bold cyan"))
    for dataset_name in cfg.evaluation_datasets:
        LOCAL_CONSOLE.print(f"\n[yellow]Evaluating dataset:[/yellow] [bold]{dataset_name}[/bold]")
        evaluate_completions_and_save_results_for_dataset(
            cfg,
            "baseline",
            dataset_name,
            eval_methodologies=cfg.jailbreak_eval_methodologies,
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg,
            "ablation",
            dataset_name,
            eval_methodologies=cfg.jailbreak_eval_methodologies,
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg,
            "actadd",
            dataset_name,
            eval_methodologies=cfg.jailbreak_eval_methodologies,
        )

    # Step 7: Generate completions for harmless dataset (baseline + induced refusal).
    LOCAL_CONSOLE.print("\n")
    LOCAL_CONSOLE.print(Panel.fit("💬 Step 7: Generating Completions on Harmless Dataset", style="bold cyan"))
    random.seed(42)
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        [],
        [],
        "baseline",
        "harmless",
        dataset=harmless_test,
    )

    actadd_refusal_hooks, _ = get_activation_addition_hooks(model_base, direction, +1.0, layer)
    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        [],
        actadd_refusal_hooks,
        "actadd",
        "harmless",
        dataset=harmless_test,
    )

    # Step 8: Evaluate harmless dataset completions.
    LOCAL_CONSOLE.print("\n")
    LOCAL_CONSOLE.print(Panel.fit("📊 Step 8: Evaluating Harmless Dataset Completions", style="bold cyan"))
    evaluate_completions_and_save_results_for_dataset(
        cfg,
        "baseline",
        "harmless",
        eval_methodologies=cfg.refusal_eval_methodologies,
    )
    evaluate_completions_and_save_results_for_dataset(
        cfg,
        "actadd",
        "harmless",
        eval_methodologies=cfg.refusal_eval_methodologies,
    )

    # Step 9: Evaluate cross-entropy loss.
    LOCAL_CONSOLE.print("\n")
    LOCAL_CONSOLE.print(Panel.fit("📉 Step 9: Evaluating Cross-Entropy Loss", style="bold cyan"))

    ablation_hooks, _ = get_all_direction_ablation_hooks(model_base, direction)
    actadd_hooks, _ = get_activation_addition_hooks(model_base, direction, -1.0, layer)

    evaluate_loss_for_datasets(cfg, model_base, [], [], "baseline")
    evaluate_loss_for_datasets(cfg, model_base, [], ablation_hooks, "ablation")
    evaluate_loss_for_datasets(cfg, model_base, [], actadd_hooks, "actadd")

    LOCAL_CONSOLE.print("\n[bold green]✨ Resume pipeline complete! ✨[/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume pipeline using existing harmful completions.")
    parser.add_argument("--model-path", required=True, help="Hugging Face model path/identifier.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for remaining steps.")
    parser.add_argument("--cache-layers", type=str, default=None, help="Unused placeholder for symmetry with run_pipeline.")

    args = parser.parse_args()
    resume_pipeline(model_path=args.model_path, batch_size=args.batch_size, cache_layers=args.cache_layers)
