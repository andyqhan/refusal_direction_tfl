"""
Modal wrapper for running the refusal direction pipeline on a GPU-backed worker.

Usage:
    modal run modal_pipeline.py::main --model-path google/gemma-2b-it --batch-size 16
    # If Step 5 already succeeded and you only need the tail end:
    modal run modal_pipeline.py::resume --model-path google/gemma-2b-it --batch-size 16

Before running:
    1) Authenticate: `modal token new`
    2) Ensure a secret named `huggingface-token` exists
    3) Optional: create a Volume `refusal-direction-runs` to persist artifacts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "refusal-direction-pipeline"
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

# Secrets (optional). Provide at least HuggingFace for gated checkpoints.
HF_SECRET = modal.Secret.from_name("huggingface-secret")
# Together secret intentionally omitted unless you recreate it and update _prepare_environment.

# Optional persistent storage for pipeline artifacts.
RUNS_VOLUME = modal.Volume.from_name("refusal-direction-runs", create_if_missing=True)

# Image with your code baked in and on PYTHONPATH.
# NOTE: copy=True guarantees /root/app exists at runtime (no reliance on runtime mounts).
pipeline_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements(str(REQUIREMENTS_PATH))
    .pip_install(
        "torch==2.4.1+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_dir(  # bake your source code into the image
        local_path=str(PROJECT_ROOT),
        remote_path="/root/app",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "**/.git/**",
            "**/.venv/**",
            "**/node_modules/**",
            "pipeline/runs",     # exclude the volume mountpoint
            "pipeline/runs/**",
        ],
    )
    .env({"PYTHONPATH": "/root/app"})  # make /root/app importable everywhere
)

# Modern App API
app = modal.App(
    name=APP_NAME,
    image=pipeline_image,
    secrets=[s for s in (HF_SECRET,) if s],
    volumes={"/root/app/pipeline/runs": RUNS_VOLUME},
)

def _mirror_secret_env_var(candidates: tuple[str, ...], targets: tuple[str, ...]) -> None:
    """Copy the first populated env var from candidates into each of the target names."""
    value = None
    for key in candidates:
        if key in os.environ and os.environ[key]:
            value = os.environ[key]
            break
    if value:
        for target in targets:
            os.environ.setdefault(target, value)

def _prepare_environment() -> None:
    """Populate env vars expected by the pipeline from Modal-provided secrets."""
    _mirror_secret_env_var(
        ("HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"),
        ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"),
    )
    # NOTE: Together is intentionally disabled per user request.

@app.function(gpu="A10G", timeout=60 * 60 * 3)
def run_pipeline(
    model_path: str,
    batch_size: int = 4,
    ignore_cached_direction: bool = False,
    cache_layers: Optional[str] = None,
) -> None:
    """
    Entry point executed on Modal. Parameters map 1:1 to the CLI flags in pipeline/run_pipeline.py.
    """
    # Work from the baked-in project root.
    os.chdir("/root/app")
    _prepare_environment()

    # If you ever need to sanity-check packaging:
    # import os; print("CWD contents:", os.listdir()); print("pipeline:", os.listdir("pipeline"))

    from pipeline.run_pipeline import run_pipeline as _run_pipeline

    _run_pipeline(
        model_path=model_path,
        batch_size=batch_size,
        ignore_cached_direction=ignore_cached_direction,
        cache_layers=cache_layers,
    )

    # Optional: force a commit of any new artifacts
    try:
        RUNS_VOLUME.commit()
    except Exception:
        pass

@app.local_entrypoint()
def main(
    model_path: str,
    batch_size: int = 4,
    ignore_cached_direction: bool = False,
    cache_layers: Optional[str] = None,
) -> None:
    """Invoke the remote pipeline from the local CLI."""
    run_pipeline.remote(
        model_path=model_path,
        batch_size=batch_size,
        ignore_cached_direction=ignore_cached_direction,
        cache_layers=cache_layers,
    )


@app.function(gpu="A10G", timeout=60 * 60 * 2)
def resume_pipeline(
    model_path: str,
    batch_size: int = 4,
    cache_layers: Optional[str] = None,
) -> None:
    """Resume the tail of the pipeline using cached harmful completions."""
    os.chdir("/root/app")
    _prepare_environment()

    from resume_pipeline import resume_pipeline as _resume_pipeline

    _resume_pipeline(
        model_path=model_path,
        batch_size=batch_size,
        cache_layers=cache_layers,
    )

    try:
        RUNS_VOLUME.commit()
    except Exception:
        pass


@app.local_entrypoint()
def resume(
    model_path: str,
    batch_size: int = 4,
    cache_layers: Optional[str] = None,
) -> None:
    """Invoke the resume helper after harmful completions already exist."""
    resume_pipeline.remote(
        model_path=model_path,
        batch_size=batch_size,
        cache_layers=cache_layers,
    )
