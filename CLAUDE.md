# Repository Analysis: Refusal Direction Extraction

## 1. How It Uses Transformers

The repository uses the HuggingFace Transformers library in several key ways:

### Model Loading (model_base.py:12, llama3_model.py:96-107)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map="auto",
).eval()
```

* **AutoModelForCausalLM**: This is Transformers' auto-class that automatically loads the correct model architecture based on the model path.
* **device\_map="auto"**: Automatically distributes model layers across available GPUs.
* **.eval()**: Sets model to evaluation mode (disables dropout, etc.).
* **requires\_grad\_(False)**: Freezes all parameters - they're never training, only doing inference.

### Tokenizer Usage (llama3\_model.py:109-115)

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
```

  * **AutoTokenizer**: Auto-loads the correct tokenizer for the model.
  * **Left padding**: Required for causal LMs during batched generation so attention masks work correctly.
  * **Pad token**: Some models don't define a pad token, so they reuse the EOS token.

### Forward Hooks (hook\_utils.py:11-39)

This is the most important Transformers feature they use:

  * `module.register_forward_pre_hook(hook_fn)`: Registers a function that runs *before* a module's `forward` pass.
  * `module.register_forward_hook(hook_fn)`: Registers a function that runs *after* a module's `forward` pass.

These hooks allow them to intercept and modify activations as they flow through the model without changing the model's code.

-----

## 2\. Model Architecture Knowledge Baked In

The codebase has moderate architecture-specific knowledge, organized in a clean abstraction:

### Architecture-Agnostic Base Class (model\_base.py:9-66)

The `ModelBase` abstract class defines what any model implementation must provide:

  * `tokenize_instructions_fn`: How to format prompts for this model.
  * `eoi_toks`: "End of instruction" tokens (where to extract activations).
  * `refusal_toks`: Tokens that indicate refusal (e.g., "I" in "I can't help with that").
  * `model_block_modules`: List of transformer blocks/layers.
  * `model_attn_modules`: List of attention sublayers.
  * `model_mlp_modules`: List of MLP sublayers.

### Model-Specific Knowledge (llama3\_model.py:126-133)

Each model implementation needs to know:

1.  **Module Access Patterns** - This is the main architecture dependency:

    ```python
    def _get_model_block_modules(self):
        return self.model.model.layers  # Llama-3 specific path

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])
    ```

    Different architectures have different module hierarchies:

      * **Llama**: `model.model.layers[i].self_attn` / `.mlp`
      * **Qwen/Gemma/Yi**: Likely similar but with different attribute names

2.  **Chat Templates** - Model-specific prompt formatting:

    ```python
    LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

    ```

{instruction}\<|eot\_id|\>\<|start\_header\_id|\>assistant\<|end\_header\_id|\>

"""
\`\`\`

3.  **Refusal Tokens** - Which token IDs indicate refusal:

    ```python
    LLAMA3_REFUSAL_TOKS = [40]  # Token ID for 'I'
    ```

### Portability to Other Architectures

Easy to port if:

  * The architecture follows standard transformer patterns (blocks → attention + MLP).
  * You can identify the module paths (inspect with `print(model)`).
  * You know the chat template format.

What you'd need to implement:

1.  New subclass of `ModelBase`.
2.  Override the 7 abstract methods (\~30-50 lines of code).
3.  Add entry to the factory (`model_factory.py:5-20`).

What stays the same:

  * All the core extraction logic (hooks, activation averaging, direction computation).
  * The selection/evaluation pipeline.

-----

## 3\. Steering Vector Extraction Logic - Deep Dive

Now for the main event - how they actually extract the steering vectors.

### High-Level Overview

The extraction happens in two phases:

1.  **Generate candidate directions** (`generate_directions.py`): Extract activation differences at multiple layers and token positions.
2.  **Select best direction** (`select_direction.py`): Evaluate which candidate works best.

Let me explain the first phase in detail since that's what you're most interested in.

### Phase 1: Generating Candidate Directions

#### The Core Idea (generate\_directions.py:42-48)

```python
def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, ...):
    mean_activations_harmful = get_mean_activations(..., harmful_instructions, ...)
    mean_activations_harmless = get_mean_activations(..., harmless_instructions, ...)

    mean_diff = mean_activations_harmful - mean_activations_harmless
    return mean_diff
```

The hypothesis: The difference between activations on harmful vs harmless prompts reveals the "refusal direction".

### Step-by-Step Extraction Process

#### 1\. Setup: Allocate Storage (generate\_directions.py:18-27)

```python
n_positions = len(positions)  # e.g., [-1] for last token, or [-4, -3, -2, -1] for last 4 tokens
n_layers = model.config.num_hidden_layers  # e.g., 32 for Llama-3-8B
d_model = model.config.hidden_size  # e.g., 4096 for Llama-3-8B

# Create accumulator tensor (using float64 for numerical precision)
mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64)
```

This creates a tensor to accumulate the mean activations across all samples.

#### 2\. Create Hooks for Each Layer (generate\_directions.py:29)

```python
fwd_pre_hooks = [
    (block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, ...))
    for layer in range(n_layers)
]
```

This creates a list of `(module, hook_function)` pairs - one for each transformer block.

#### 3\. The Hook Function (generate\_directions.py:12-16)

This is where the magic happens:

```python
def get_mean_activations_pre_hook(layer, cache, n_samples, positions):
    def hook_fn(module, input):
        # input[0] is the activation tensor entering this layer
        activation = input[0].clone().to(cache)  # Shape: [batch_size, seq_len, d_model]

        # Extract activations at specific token positions and accumulate
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn
```

What this does:

  * `input[0]`: In Transformers, layer inputs are tuples. The first element is the hidden states tensor.
  * `.clone()`: Creates a copy so we don't affect the original.
  * `activation[:, positions, :]`: Extracts specific token positions
      * `:` = all samples in batch
      * `positions` = e.g., `[-1]` for last token, or `[-4, -3, -2, -1]` for last 4 tokens
      * `:` = all hidden dimensions
  * `(1.0 / n_samples) * ... .sum(dim=0)`: Computes running mean across batches.

Let me explain the indexing more clearly:

If `positions = [-1]` and we have a batch of 2 samples:

```
activation shape: [2, 20, 4096]  # 2 samples, 20 tokens each, 4096 dims
activation[:, [-1], :] → [2, 1, 4096]  # Extract last token
.sum(dim=0) → [1, 4096]  # Sum across batch
cache[:, layer] += (1/128) * [1, 4096]  # Accumulate into cache
```

After processing all 128 training samples, `cache[0, layer]` contains the mean activation at position `-1` in layer `layer`.

#### 4\. Run Forward Passes (generate\_directions.py:31-38)

```python
for i in range(0, len(instructions), batch_size):
    inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
        model(
            input_ids=inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
        )
```

What happens here:

1.  Tokenize a batch of instructions (e.g., 32 prompts).
2.  Enter hook context manager (`add_hooks`):
      * Registers all the hooks we created
      * Runs the forward pass
      * Hooks fire automatically as each layer processes
      * Deregisters hooks when exiting context
3.  Run forward pass: Just calls `model()` normally.
      * We don't care about the output (logits).
      * We only care about the intermediate activations captured by hooks.
4.  Hooks accumulate means: Each hook adds to the `mean_activations` tensor.

#### 5\. The Hook Context Manager (hook\_utils.py:10-39)

```python
@contextlib.contextmanager
def add_hooks(module_forward_pre_hooks, module_forward_hooks, **kwargs):
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield  # Code inside 'with' block runs here
    finally:
        for h in handles:
            h.remove()  # Clean up hooks
```

This is a standard Python context manager pattern that:

  * Registers hooks *before* entering the `with` block.
  * Lets the model run.
  * Removes hooks *after* (even if there's an error).

#### Token Position Selection (generate\_directions.py:54)

```python
positions=list(range(-len(model_base.eoi_toks), 0))
```

This extracts activations at the end-of-instruction tokens. For Llama-3, `eoi_toks` is the token sequence *after* the instruction (`llama3_model.py:121`):

```python
def _get_eoi_toks(self):
    return self.tokenizer.encode(
        LLAMA3_CHAT_TEMPLATE.split("{instruction}")[-1],
        add_special_tokens=False
    )
```

This would be: `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`

So if this is 4 tokens, `positions = [-4, -3, -2, -1]`.

Why? They want to capture the model's "decision state" right before it starts generating the response.

#### Final Output (generate\_directions.py:54-60)

```python
mean_diffs = get_mean_diff(
    model, tokenizer,
    harmful_instructions,
    harmless_instructions,
    tokenize_instructions_fn,
    model_block_modules,
    positions=list(range(-len(model_base.eoi_toks), 0))
)
```

This runs the whole process twice (once for harmful, once for harmless) and computes:

`mean_diffs[pos, layer] = mean_activation_harmful[pos, layer] - mean_activation_harmless[pos, layer]`

Result shape: `[n_positions, n_layers, d_model]`

  * e.g., `[4, 32, 4096]` for Llama-3-8B with 4 end-of-instruction tokens

This gives them candidate refusal directions for every combination of:

  * **Token position** (which of the 4 EOI tokens)
  * **Layer** (which of the 32 layers)

### Phase 2: Selecting the Best Direction (Brief Overview)

Since you said to keep this vague, here's the high-level:

The selection process evaluates each candidate by:

1.  **Ablation test** (`select_direction.py:170-179`): Remove the direction from activations on *harmful* prompts → does it reduce refusal?
2.  **Steering test** (`select_direction.py:181-191`): Add the direction to activations on *harmless* prompts → does it increase refusal?
3.  **KL divergence test** (`select_direction.py:150-168`): Does removing the direction distort the output distribution too much on *harmless* prompts?

Selection criteria (`select_direction.py:278-289`):

  * Choose the direction that maximally reduces refusal when ablated (lowest refusal score).
  * But filter out directions with high KL divergence or poor steering ability.
  * Prune last 20% of layers (they don't work as well).

-----

## Key Insights About the Architecture

### 1\. Where Are Hooks Placed?

For **extraction** (`generate_directions.py:29`):

  * **Pre-hooks** on transformer blocks (`model.model.layers[i]`).
  * This captures the residual stream *entering* each layer.
  * Called `"resid_pre"` in the code comments.

For **intervention** (`hook_utils.py:84-88`):

  * **Pre-hooks** on transformer blocks (residual stream input).
  * **Post-hooks** on attention modules (`model.model.layers[i].self_attn`).
  * **Post-hooks** on MLP modules (`model.model.layers[i].mlp`).

This ensures they ablate/steer the direction both in the residual stream AND in the outputs of attention and MLP sublayers.

### 2\. Orthogonal Projection Math (hook\_utils.py:50-52)

```python
direction = direction / direction.norm()  # Normalize to unit vector
activation -= (activation @ direction).unsqueeze(-1) * direction
```

This is the standard orthogonal projection formula:

```
a_orthogonal = a - (a · d̂) * d̂
```

Where `d̂` is the unit direction vector. This removes the component of the activation in the direction of `d̂`.

### 3\. Why Float64? (generate\_directions.py:27)

```python
mean_activations = torch.zeros(..., dtype=torch.float64)
```

Accumulating means across 128+ samples can cause numerical errors with float32. Using float64 prevents drift.

-----

## Summary

### How hard to port to another architecture?

  * Very easy if it follows standard transformer patterns.
  * You just need to know the module paths and write \~40 lines of model-specific code.
  * The core extraction logic is completely architecture-agnostic.

### What's actually being done to the model?

1.  Run forward passes on harmful and harmless prompts.
2.  Use PyTorch hooks to intercept activations at each layer.
3.  Accumulate mean activations at end-of-instruction token positions.
4.  Subtract means to get candidate refusal directions.
5.  Evaluate candidates and select the best one.

The clever part: Using forward pre-hooks to capture activations without any model modifications. This is a general technique applicable to any PyTorch model.

```
```
