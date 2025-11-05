
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

# Gemma 3 chat template is based on Gemma 2 format
# - Official Gemma 3 documentation: https://huggingface.co/google/gemma-3-1b-it
# - Uses same <start_of_turn> and <end_of_turn> tokens as Gemma 1/2
# - However, Gemma 3 uses a NEW tokenizer (Gemini 2.0) with 262K entries

GEMMA3_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

# TODO: CRITICAL - These refusal token IDs need to be verified with actual Gemma 3 tokenizer!
# Gemma 3 uses the new Gemini 2.0 tokenizer with 262K entries, so token IDs are different from Gemma 1/2
# The old Gemma (1/2) used token ID 235285 for 'I'
# These are placeholder values that should be verified by:
#   tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
#   print(tokenizer.encode("I", add_special_tokens=False))
#   print(tokenizer.encode("As", add_special_tokens=False))
GEMMA3_REFUSAL_TOKS = [235285]  # Placeholder - needs verification!

def format_instruction_gemma3_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")

    formatted_instruction = GEMMA3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_gemma3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    """
    Tokenize instructions using chat templates.

    Args:
        instructions: Can be either:
            - List of strings (legacy format) - will use old template formatting
            - List of chat dicts [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    """
    # Check if instructions are already in chat format
    if instructions and isinstance(instructions[0], list):
        # New chat format: use apply_chat_template
        prompts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in instructions
        ]
    else:
        # Legacy format: use old template formatting
        if outputs is not None:
            prompts = [
                format_instruction_gemma3_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
                for instruction, output in zip(instructions, outputs)
            ]
        else:
            prompts = [
                format_instruction_gemma3_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
                for instruction in instructions
            ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_gemma3_weights(model: AutoTokenizer, direction: Float[Tensor, "d_model"]):
    # Gemma 3 uses same architecture as Gemma 2 (model.model.layers)
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        # Gemma 3 uses self_attn.o_proj (same as Gemma 2)
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        # Gemma 3 uses mlp.down_proj (same as Gemma 2)
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_gemma3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Gemma3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Gemma 3 uses standard transformers loading, no trust_remote_code needed
        # Recommended dtype is bfloat16 per official documentation
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        # Log device information
        device = model.device
        logger.info(f"Gemma 3 model loaded on device: {device}")

        return model

    def _load_tokenizer(self, model_path):
        # Gemma 3 uses the new Gemini 2.0 tokenizer with 262K entries
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(GEMMA3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return GEMMA3_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Gemma 3 uses model.model.layers (same as Gemma 2 and LLaMA-style)
        return self.model.model.layers

    def _get_attn_modules(self):
        # Gemma 3 uses self_attn (same as Gemma 2)
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma3_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_gemma3_weights, direction=direction, coeff=coeff, layer=layer)
