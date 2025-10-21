import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.transformerlens_hook_utils import (
    get_all_direction_ablation_hooks,
    get_activation_addition_hooks
)

def refusal_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    refusal_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    # MPS doesn't support float64, use float32 instead
    # Use float64 on CPU/CUDA for better numerical precision
    if logits.device.type == 'mps':
        logits = logits.to(torch.float32)
    else:
        logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1):
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)
    device = model.cfg.device
    refusal_scores = torch.zeros(len(instructions), device=device)

    # Calculate number of examples per batch for progress bar
    n_batches = (len(instructions) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(instructions), batch_size),
                  desc=f"Computing refusal scores ({len(instructions)} examples)",
                  unit="batch",
                  total=n_batches):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        # Combine all hooks for TransformerLens
        all_hooks = fwd_pre_hooks + fwd_hooks

        if len(all_hooks) > 0:
            # Use run_with_hooks for TransformerLens
            logits = model.run_with_hooks(
                tokenized_instructions.input_ids.to(device),
                fwd_hooks=all_hooks,
                prepend_bos=False,
            )
        else:
            # No hooks, just run the model
            with torch.no_grad():
                logits = model(
                    tokenized_instructions.input_ids.to(device),
                    prepend_bos=False,
                )

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores

def get_last_position_logits(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1) -> Float[Tensor, "n_instructions d_vocab"]:
    last_position_logits = None
    device = model.cfg.device

    n_batches = (len(instructions) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(instructions), batch_size),
                  desc=f"Computing logits ({len(instructions)} examples)",
                  unit="batch",
                  total=n_batches):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        # Combine all hooks for TransformerLens
        all_hooks = fwd_pre_hooks + fwd_hooks

        if len(all_hooks) > 0:
            # Use run_with_hooks for TransformerLens
            logits = model.run_with_hooks(
                tokenized_instructions.input_ids.to(device),
                fwd_hooks=all_hooks,
                prepend_bos=False,
            )
        else:
            # No hooks, just run the model
            with torch.no_grad():
                logits = model(
                    tokenized_instructions.input_ids.to(device),
                    prepend_bos=False,
                )

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits

def plot_refusal_scores(
    refusal_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = refusal_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            refusal_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_refusal_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_refusal_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Refusal score')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False

def select_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=1
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    # MPS doesn't support float64, use float32 instead
    precision_dtype = torch.float32 if model_base.model.cfg.device == 'mps' else torch.float64
    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.cfg.device, dtype=precision_dtype)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.cfg.device, dtype=precision_dtype)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.cfg.device, dtype=precision_dtype)

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, ablation_dir)

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=[],
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_hooks, _ = get_all_direction_ablation_hooks(model_base, ablation_dir)

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=[], fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = 1.0
            fwd_hooks, _ = get_activation_addition_hooks(model_base, refusal_vector, coeff, source_layer)

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=[], fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    # MPS doesn't support float64, use float32 instead
    if logits_a.device.type == 'mps':
        logits_a = logits_a.to(torch.float32)
        logits_b = logits_b.to(torch.float32)
    else:
        logits_a = logits_a.to(torch.float64)
        logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)