
import torch
import functools
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

logger = logging.getLogger(__name__)

# Qwen3 chat templates use ChatML format (same as Qwen 1.x)
# - Based on official Qwen documentation
# - Uses <|im_start|> and <|im_end|> tokens
# Note: Qwen3 has a "thinking mode" that must be disabled for refusal analysis

QWEN3_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN3_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Refusal tokens for Qwen3 - these should be verified on actual Qwen3 model
# Expected to be similar to Qwen 1.x: ['I', 'As']
QWEN3_REFUSAL_TOKS = [40, 2121]

def format_instruction_qwen3_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN3_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen3_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen3_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_qwen3_weights(model, direction: Float[Tensor, "d_model"]):
    # Qwen3 uses LLaMA-style architecture (model.model.layers)
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        # Qwen3 uses self_attn.o_proj (not attn.c_proj like Qwen1)
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        # Qwen3 uses mlp.down_proj (not mlp.c_proj like Qwen1)
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_qwen3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Qwen3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Qwen3 requires transformers >= 4.51.0 and trust_remote_code=True
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        # Log device information
        device = model.device
        logger.info(f"Qwen3 model loaded on device: {device}")

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        tokenizer.padding_side = "left"
        # Qwen3 uses standard eos_token (LLaMA-style), not eod_id (Qwen1-style)
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return QWEN3_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Qwen3 uses model.model.layers (LLaMA-style), not model.transformer.h (Qwen1/GPT-2 style)
        return self.model.model.layers

    def _get_attn_modules(self):
        # Qwen3 uses self_attn (LLaMA-style), not attn (Qwen1-style)
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen3_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen3_weights, direction=direction, coeff=coeff, layer=layer)
