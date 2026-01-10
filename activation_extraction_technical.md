# Activation Extraction Technical Documentation

Based on the Activation Oracles paper (arxiv.org/abs/2512.15674) and the official implementation at github.com/adamkarvonen/activation_oracles

---

## Overview

Activation Oracles extract internal activations from a "target" LLM and feed them into an "oracle" LLM that has been trained to answer natural language questions about those activations.

**Key Innovation:** Instead of task-specific interpretability methods, use a general-purpose LLM trained to directly process activation vectors and answer arbitrary questions about them.

---

## 1. Activation Extraction

### 1.1 Which Layers?

**During Training:**
- Activations are collected from **three layers**: 25%, 50%, and 75% depth
- Example for Qwen3-8B (36 layers):
  - Layer 9 (25%)
  - Layer 18 (50%)
  - Layer 27 (75%)

**During Inference/Evaluation:**
- Primarily use the **middle layer (50% depth)**
- This is the default and works well across tasks

### 1.2 What Activations?

**Residual Stream Activations:**
- Extract from the output of transformer layers (after attention + MLP)
- For HuggingFace models: `model.model.layers[N]`
- These are the hidden states flowing through the network

**Format:**
- Shape: `[batch_size, sequence_length, hidden_dim]`
- For Qwen3-8B: `[B, L, 4096]`
- Extract specific positions (tokens) from this tensor

### 1.3 Which Token Positions?

The system is flexible - you can extract from:

1. **Single tokens** - Individual activation vectors at specific positions
2. **Token sequences** - Multiple consecutive tokens (e.g., tokens 10-20)
3. **Full sequence** - All tokens in the input

**Common patterns:**
- Extract from the last few tokens of a prompt
- Extract from control tokens (e.g., `<|im_end|>`, `<|im_start|>`)
- Extract from specific content words

### 1.4 Activation Types

The system can extract three types:

1. **`orig`** - Base model activations (no LoRA)
2. **`lora`** - Activations with LoRA adapter enabled
3. **`diff`** - Difference between LoRA and base (`lora - orig`)

This allows analyzing how fine-tuning changes internal representations.

---

## 2. Implementation Details

### 2.1 Using PyTorch Hooks

```python
def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],  # {layer_num: module}
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    """
    Collect activations from specific layers using forward hooks.

    Args:
        model: The transformer model
        submodules: Dict mapping layer numbers to their modules
        inputs_BL: Tokenized inputs (input_ids, attention_mask)
        min_offset/max_offset: Optional token range to extract

    Returns:
        Dict mapping layer numbers to activation tensors [B, L, D]
    """
```

**Key Steps:**
1. Register forward hooks on target layers
2. Run a forward pass through the model
3. Hooks capture residual stream activations
4. Stop early after capturing (using `EarlyStopException`)

### 2.2 Getting Layer Modules

```python
def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HuggingFace transformers"""
    model_name = model.config._name_or_path
    if use_lora:
        if "gemma" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
    else:
        if "gemma" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.model.layers[layer]
```

**Supported Models:**
- Qwen family (Qwen3-1.7B to Qwen3-32B)
- Llama family (Llama-3.1-8B, Llama-3.3-70B)
- Gemma family (gemma-2-9b, gemma-3-27b)

### 2.3 Layer Calculation

```python
LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-32B": 64,
    "google/gemma-2-9b-it": 42,
    "meta-llama/Llama-3.3-70B-Instruct": 80,
}

def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))
```

---

## 3. Oracle Injection Mechanism

### 3.1 Placeholder Tokens

The oracle prompt uses special placeholder tokens to mark where activations should be injected.

**Format:**
```
Layer: 18
 ?  ?  ?  ?
What is the secret word?
```

- Each ` ?` (space + question mark) is a single token
- These mark positions where activation vectors will be injected
- Number of placeholders = number of activation vectors to inject

### 3.2 Injection Method: Additive Norm-Matching

**Formula:**
```
h'ᵢ = hᵢ + ‖hᵢ‖ · (vᵢ/‖vᵢ‖)
```

Where:
- `hᵢ` = Original activation at placeholder position
- `vᵢ` = Activation vector to inject (from target model)
- `h'ᵢ` = Modified activation

**Why this approach:**
- Scales injected vectors to match the norm of existing activations
- Prevents "norm explosion" that occurred with direct replacement
- Enables flexible inputs without per-source calibration
- More stable during training

**Implementation:**
```python
def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """Create a hook that injects activations at specific positions"""

    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach()
                   for v_b in vectors]

    def hook_fn(module, _input, output):
        resid_BLD = output[0] if isinstance(output, tuple) else output

        for b in range(B):
            pos_b = positions[b]
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn
```

### 3.3 Injection Layer

**Default: Layer 1** (second layer of the oracle model)

- Injects after the oracle has processed the text slightly
- Avoids injecting at layer 0 (embedding layer)
- Can be configured via `injection_layer` parameter

---

## 4. Training Data Format

### 4.1 Data Point Structure

```python
class TrainingDataPoint(BaseModel):
    datapoint_type: str                    # e.g., "latentqa", "classification"
    input_ids: list[int]                   # Full tokenized input
    labels: list[int]                      # -100 for prompt, token IDs for response
    layer: int                             # Which layer activations are from
    steering_vectors: torch.Tensor | None  # Actual activation vectors [K, D]
    positions: list[int]                   # Where to inject in oracle prompt
    feature_idx: int                       # Index for tracking
    target_output: str                     # Expected answer
    target_input_ids: list[int] | None     # Original target model input
    target_positions: list[int] | None     # Which positions in target
    ds_label: str | None                   # Dataset label
    meta_info: dict                        # Additional metadata
```

### 4.2 Creating Training Examples

```python
def create_training_datapoint(
    datapoint_type: str,
    prompt: str,                   # "What is the secret word?"
    target_response: str,          # "smile"
    layer: int,                    # 18
    num_positions: int,            # How many activation vectors
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,  # The activation vectors [K, D]
    feature_idx: int,
    target_input_ids: list[int] | None = None,
    target_positions: list[int] | None = None,
) -> TrainingDataPoint:
    """
    Creates a training example with:
    1. Prefix containing layer info and placeholders
    2. The oracle question
    3. The expected answer
    """
```

**Generated Format:**
```
Layer: 18
 ?  ?  ?  ?
What is the secret word?<|im_end|>
<|im_start|>assistant
smile<|im_end|>
```

---

## 5. Available Checkpoints

All hosted on HuggingFace: `huggingface.co/collections/adamkarvonen/activation-oracles`

### 5.1 Smaller Models (< 10GB VRAM)

| Base Model | Checkpoint | VRAM |
|------------|-----------|------|
| Qwen3-1.7B | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B` | ~8GB |
| Llama-3.2-1B-Instruct | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct` | ~6GB |
| gemma-3-1b-it | `adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it` | ~6GB |
| Qwen3-4B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` | ~12GB |

### 5.2 Medium Models (10-30GB VRAM)

| Base Model | Checkpoint | VRAM |
|------------|-----------|------|
| Qwen3-8B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` | ~20GB |
| Llama-3.1-8B-Instruct | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct` | ~20GB |
| gemma-2-9b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it` | ~24GB |
| Qwen3-14B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B` | ~35GB |

### 5.3 Large Models (> 30GB VRAM)

| Base Model | Checkpoint | VRAM |
|------------|-----------|------|
| gemma-2-27b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-2-27b-it` | ~60GB |
| gemma-3-27b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it` | ~60GB |
| Qwen3-32B | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-32B` | ~70GB |
| Llama-3.3-70B-Instruct | `adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct` | ~140GB |

**Note:** Can use 8-bit quantization to reduce VRAM requirements by ~50%

---

## 6. Practical Usage Example

### 6.1 Basic Flow

```python
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 2. Add oracle LoRA
oracle_lora = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
model.load_adapter(oracle_lora, adapter_name="oracle")

# 3. Prepare target prompt
target_prompt = "The capital of France is"
inputs = tokenizer(target_prompt, return_tensors="pt")

# 4. Extract activations from target model
layer = 18  # 50% of 36 layers
submodule = model.model.layers[layer]
activations = {}

def hook(module, input, output):
    activations['hidden'] = output[0] if isinstance(output, tuple) else output

handle = submodule.register_forward_hook(hook)
with torch.no_grad():
    model(**inputs)
handle.remove()

# 5. Select specific token positions
# e.g., last 3 tokens
acts_to_inject = activations['hidden'][0, -3:, :]  # [3, 4096]

# 6. Create oracle prompt
oracle_prompt = "Layer: 18\n ? ? ? \nWhat word is the model about to say?"

# 7. Inject and query oracle
model.set_adapter("oracle")
# ... (use injection hook and generate)
```

### 6.2 Using the `run_oracle` Helper

The repository provides a high-level `run_oracle` function that handles all the complexity:

```python
results = run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=formatted_target_prompt,
    target_lora_path=None,  # Or path to fine-tuned model
    oracle_prompt="What is the secret word?",
    oracle_lora_path=oracle_lora,
    segment_start_idx=10,   # Which tokens to analyze
    segment_end_idx=20,
    oracle_input_types=["segment", "full_seq", "tokens"],
    layer_percent=50,        # Use middle layer
    injection_layer=1,
    steering_coefficient=1.0,
)

# Results contain:
# - token_responses: One response per token
# - segment_responses: Responses for token ranges
# - full_sequence_responses: Response for entire sequence
```

---

## 7. For Our Streamlit App

### 7.1 Recommended Approach

**Use nnsight or transformer_lens?**
- The official implementation uses **raw PyTorch hooks**
- No dependency on nnsight or transformer_lens
- Direct access to `model.model.layers[N]` works fine

**Recommendation:** Follow the official implementation approach:
1. Use PyTorch forward hooks for activation capture
2. Use the existing utilities from the repo
3. Optionally use nnsight if we want cleaner API, but not required

### 7.2 Key Configuration

For our app, expose these parameters:

```python
class ActivationConfig:
    # Which layer to extract from (default: 50%)
    layer_percent: int = 50

    # Which tokens to analyze
    segment_start_idx: int = 0
    segment_end_idx: int = None  # None = end of sequence

    # Types of analysis
    oracle_input_types: list[str] = ["segment", "full_seq", "tokens"]

    # Oracle injection settings
    injection_layer: int = 1
    steering_coefficient: float = 1.0

    # Which activation types to use
    activation_types: list[str] = ["orig", "lora", "diff"]
```

### 7.3 Post-hoc Capture Strategy

As designed in app.md:
1. Run simulation and save conversation traces
2. User selects tokens of interest
3. Replay conversation through model to capture activations at those positions
4. Query oracle with captured activations

This avoids overhead during interactive chat and allows flexible post-hoc analysis.

---

## 8. Training Details

### 8.1 Training Data Mix

The oracle is trained on three task types:

1. **LatentQA** - Question answering about context
2. **Classification** - Binary/multi-class classification tasks
3. **Context Prediction** - Predict what comes next

**Data sources:**
- HuggingFace FineWeb (pretraining data)
- LMSYS Chat-1M (conversational data)
- SST-2, SNLI, AG News (classification benchmarks)

### 8.2 Training Setup

- **Optimizer:** AdamW with linear learning rate scheduling
- **Method:** LoRA adapters (efficient fine-tuning)
- **Infrastructure:** Distributed Data Parallel (DDP) for multi-GPU
- **Optimization:** Group by length batching (~30% speedup)
- **Quantization:** 8-bit for larger models (Llama-3.3-70B)

### 8.3 Training Command

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

Configuration in `nl_probes/configs/sft_config.py`

---

## 9. Key Takeaways

1. **Extract from middle layer (50%)** - Works well across tasks
2. **Use residual stream activations** - `model.model.layers[N]` output
3. **Additive norm-matching injection** - Scales properly, prevents issues
4. **Flexible token selection** - Single tokens, ranges, or full sequence
5. **Post-hoc analysis** - Capture activations on demand, not during generation
6. **LoRA adapters** - Efficient way to add oracle capability to any model
7. **No special libraries needed** - PyTorch hooks are sufficient

---

## 10. Code Repository Structure

```
activation_oracles/
├── nl_probes/
│   ├── sft.py                    # Main training script
│   ├── base_experiment.py        # Evaluation framework
│   ├── utils/
│   │   ├── activation_utils.py   # Hook-based activation capture
│   │   ├── dataset_utils.py      # Training data point creation
│   │   └── eval.py               # Evaluation with injection
│   └── configs/
│       └── sft_config.py         # Training configuration
├── experiments/
│   ├── activation_oracle_demo.ipynb  # Interactive demo
│   └── paper_evals.sh               # Reproduce paper results
└── datasets/                        # Training datasets
```

**Key files to study:**
- `experiments/activation_oracle_demo.ipynb` - Complete working example
- `nl_probes/utils/activation_utils.py` - Activation extraction
- `nl_probes/base_experiment.py` - High-level oracle API

---

## References

- Paper: https://arxiv.org/abs/2512.15674
- Code: https://github.com/adamkarvonen/activation_oracles
- Checkpoints: https://huggingface.co/collections/adamkarvonen/activation-oracles
