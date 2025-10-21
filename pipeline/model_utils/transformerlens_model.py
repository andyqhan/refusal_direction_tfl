
import torch
import functools

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.model_utils.model_base import ModelBase

# Chat templates for different model families
# These follow the same format as the original model-specific files

# Llama 2 Chat Template
LLAMA2_CHAT_TEMPLATE = """[INST] <<SYS>>
{system_prompt}
<</SYS>>

{instruction} [/INST] """

LLAMA2_CHAT_TEMPLATE_NO_SYSTEM = """[INST] {instruction} [/INST] """

# Llama 3 Chat Template
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Gemma Chat Template
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

# Qwen Chat Template
QWEN_CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Yi Chat Template
YI_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Refusal tokens for different models
LLAMA2_REFUSAL_TOKS = [306]  # ' I'
LLAMA3_REFUSAL_TOKS = [40]   # 'I'
GEMMA_REFUSAL_TOKS = [235285]  # 'I'
QWEN_REFUSAL_TOKS = [40]  # 'I'
YI_REFUSAL_TOKS = [40]  # 'I'


def format_instruction(instruction: str, output: str = None, system: str = None,
                       chat_template: str = None, include_trailing_whitespace: bool = True):
    """Format an instruction using the appropriate chat template."""
    if system is not None and '{system_prompt}' in chat_template:
        formatted_instruction = chat_template.format(instruction=instruction, system_prompt=system)
    elif system is None and '{system_prompt}' not in chat_template:
        formatted_instruction = chat_template.format(instruction=instruction)
    elif system is not None and '{system_prompt}' not in chat_template:
        # Model doesn't support system prompt, just use instruction
        formatted_instruction = chat_template.format(instruction=instruction)
    else:
        # system is None but template expects it - use empty system
        formatted_instruction = chat_template.format(instruction=instruction, system_prompt="")

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_generic(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    chat_template: str = None,
    include_trailing_whitespace: bool = True
):
    """Generic tokenization function for all model types."""
    if outputs is not None:
        prompts = [
            format_instruction(
                instruction=instruction,
                output=output,
                system=system,
                chat_template=chat_template,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction(
                instruction=instruction,
                system=system,
                chat_template=chat_template,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class TransformerLensModel(ModelBase):
    """
    TransformerLens-based model wrapper that implements the ModelBase interface.
    This provides a unified interface for all supported models using TransformerLens's HookedTransformer.
    """

    def __init__(self, model_name_or_path: str, dtype=torch.bfloat16):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype

        # Detect model family from path
        self.model_family = self._detect_model_family(model_name_or_path)

        # Set up chat template and refusal tokens based on model family
        self._setup_model_specifics()

        # Call parent init which will call our _load_model and _load_tokenizer
        super().__init__(model_name_or_path)

    def _detect_model_family(self, model_path: str) -> str:
        """Detect the model family from the model path."""
        model_path_lower = model_path.lower()

        if 'llama-3' in model_path_lower or 'llama3' in model_path_lower:
            return 'llama3'
        elif 'llama' in model_path_lower:
            return 'llama2'
        elif 'gemma' in model_path_lower:
            return 'gemma'
        elif 'qwen' in model_path_lower:
            return 'qwen'
        elif 'yi' in model_path_lower:
            return 'yi'
        else:
            raise ValueError(f"Unknown model family for path: {model_path}")

    def _setup_model_specifics(self):
        """Set up model-specific chat templates and refusal tokens."""
        if self.model_family == 'llama2':
            self.chat_template = LLAMA2_CHAT_TEMPLATE_NO_SYSTEM
            self.chat_template_with_system = LLAMA2_CHAT_TEMPLATE
            self._refusal_toks = LLAMA2_REFUSAL_TOKS
        elif self.model_family == 'llama3':
            self.chat_template = LLAMA3_CHAT_TEMPLATE
            self.chat_template_with_system = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM
            self._refusal_toks = LLAMA3_REFUSAL_TOKS
        elif self.model_family == 'gemma':
            self.chat_template = GEMMA_CHAT_TEMPLATE
            self.chat_template_with_system = GEMMA_CHAT_TEMPLATE  # Gemma doesn't have system prompt
            self._refusal_toks = GEMMA_REFUSAL_TOKS
        elif self.model_family == 'qwen':
            self.chat_template = QWEN_CHAT_TEMPLATE
            self.chat_template_with_system = QWEN_CHAT_TEMPLATE
            self._refusal_toks = QWEN_REFUSAL_TOKS
        elif self.model_family == 'yi':
            self.chat_template = YI_CHAT_TEMPLATE
            self.chat_template_with_system = YI_CHAT_TEMPLATE
            self._refusal_toks = YI_REFUSAL_TOKS

    def _load_model(self, model_name_or_path: str) -> HookedTransformer:
        """Load model using TransformerLens's HookedTransformer."""
        # Detect best available device: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = HookedTransformer.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            device=device,
            fold_ln=False,  # Keep LayerNorm separate for better interpretability
            center_writing_weights=False,  # Don't center weights
            center_unembed=False,  # Don't center unembed
        )

        # Enable hook_result hook on attention layers (must be set after loading)
        model.set_use_attn_result(True)

        model.eval()
        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """Load tokenizer using HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        """Return the tokenization function for this model."""
        # Determine which template to use based on whether system prompt is provided
        return functools.partial(
            tokenize_instructions_generic,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            system=None,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        """Get end-of-instruction tokens."""
        # Extract the template suffix after {instruction}
        template_parts = self.chat_template.split("{instruction}")
        if len(template_parts) > 1:
            suffix = template_parts[-1]
            # For Llama 2, remove the trailing whitespace before tokenizing
            if not suffix.rstrip():
                # If suffix is only whitespace, try the system template
                template_parts = self.chat_template_with_system.split("{instruction}")
                suffix = template_parts[-1] if len(template_parts) > 1 else ""

            # Tokenize the suffix
            if suffix:
                # Remove leading/trailing whitespace for tokenization
                suffix_cleaned = suffix.lstrip()
                eoi_toks = self.tokenizer.encode(suffix_cleaned, add_special_tokens=False)
                return eoi_toks

        # Fallback: return empty list if can't determine
        return []

    def _get_refusal_toks(self):
        """Get refusal tokens."""
        return self._refusal_toks

    def _get_model_block_modules(self):
        """Get the transformer blocks from the model."""
        return self.model.blocks

    def _get_attn_modules(self):
        """Get attention modules from each block."""
        return torch.nn.ModuleList([block.attn for block in self.model.blocks])

    def _get_mlp_modules(self):
        """Get MLP modules from each block."""
        return torch.nn.ModuleList([block.mlp for block in self.model.blocks])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        """
        Get function to orthogonalize model weights.
        Note: This may not be directly needed with TransformerLens hooks.
        """
        # This is kept for compatibility but may not be used with hook-based interventions
        raise NotImplementedError("Orthogonalization not yet implemented for TransformerLens")

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        """
        Get function to add activations to model.
        Note: This may not be directly needed with TransformerLens hooks.
        """
        # This is kept for compatibility but may not be used with hook-based interventions
        raise NotImplementedError("Activation addition not yet implemented for TransformerLens")

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1, max_new_tokens=64):
        """
        Generate completions using TransformerLens with hooks.

        Args:
            dataset: List of dicts with 'instruction' and 'category' keys
            fwd_pre_hooks: List of (hook_name, hook_fn) tuples (TransformerLens format)
            fwd_hooks: List of (hook_name, hook_fn) tuples (TransformerLens format)
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            List of completion dicts with 'category', 'prompt', 'response' keys
        """
        from transformers import GenerationConfig
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
        import contextlib

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        # Combine hooks
        all_hooks = fwd_pre_hooks + fwd_hooks

        # Calculate total batches
        n_batches = (len(dataset) + batch_size - 1) // batch_size

        # Create nested progress bars with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # Batch-level progress
            batch_task = progress.add_task("[cyan]Generating completions", total=n_batches)
            # Token-level progress - use indeterminate progress since we can't track real-time
            token_task = progress.add_task("[dim]  └─ Generating tokens...", total=None, visible=False)

            for batch_idx in range(0, len(dataset), batch_size):
                current_batch_num = batch_idx // batch_size + 1
                current_batch_size = min(batch_size, len(dataset) - batch_idx)

                # Update batch description with current batch info
                progress.update(batch_task, description=f"[cyan]Batch {current_batch_num}/{n_batches} ({current_batch_size} prompts)")

                # Show token progress for this batch
                progress.update(token_task, visible=True, description=f"[dim]  └─ Generating tokens (max {max_new_tokens})...")

                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[batch_idx:batch_idx + batch_size])

                @contextlib.contextmanager
                def temporary_hooks(hook_list):
                    """Temporarily register hooks on the model."""
                    handles = []
                    try:
                        for hook_name, hook_fn in hook_list:
                            handle = self.model.add_hook(hook_name, hook_fn)
                            handles.append(handle)
                        yield
                    finally:
                        for handle in handles:
                            if handle is not None:
                                handle.remove()

                if len(all_hooks) > 0:
                    input_ids = tokenized_instructions.input_ids.to(self.model.cfg.device)

                    with temporary_hooks(all_hooks):
                        generation_toks = self.model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            eos_token_id=generation_config.eos_token_id,
                            stop_at_eos=True,
                            prepend_bos=False,
                            verbose=False,
                        )
                else:
                    input_ids = tokenized_instructions.input_ids.to(self.model.cfg.device)

                    generation_toks = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=generation_config.eos_token_id,
                        stop_at_eos=True,
                        prepend_bos=False,
                        verbose=False,
                    )

                # Hide token progress after generation completes
                progress.update(token_task, visible=False)

                # Remove the input tokens from generation
                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                # Decode generations
                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[batch_idx + generation_idx],
                        'prompt': instructions[batch_idx + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

                # Update batch progress
                progress.update(batch_task, advance=1)

        return completions
