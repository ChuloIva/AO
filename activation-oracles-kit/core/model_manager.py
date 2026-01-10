"""
Model Manager - Handles model loading with Mac/CUDA support and 12GB VRAM optimization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, List, Dict
import sys
import os

# Add activation_oracles to path if not already present
activation_oracles_path = os.path.join(os.path.dirname(__file__), '../../activation_oracles')
if os.path.exists(activation_oracles_path) and activation_oracles_path not in sys.path:
    sys.path.insert(0, activation_oracles_path)

from nl_probes.utils.common import load_model as ao_load_model, load_tokenizer as ao_load_tokenizer


# Supported models optimized for 12GB VRAM
SUPPORTED_MODELS = {
    "Qwen/Qwen3-1.7B": {
        "vram_8bit": 4,
        "vram_fp16": 8,
        "oracle": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
        "layers": 28
    },
    "Qwen/Qwen3-4B": {
        "vram_8bit": 6,
        "vram_fp16": 12,
        "oracle": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B",
        "layers": 36
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "vram_8bit": 3,
        "vram_fp16": 6,
        "oracle": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct",
        "layers": 16
    },
    "google/gemma-3-1b-it": {
        "vram_8bit": 3,
        "vram_fp16": 6,
        "oracle": "adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it",
        "layers": 26
    }
}


def get_device() -> torch.device:
    """
    Get appropriate device for current platform.
    Prioritizes: MPS (Mac) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name() -> str:
    """Get human-readable device name"""
    device = get_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple MPS (Metal)"
    return "CPU"


def get_vram_usage() -> float:
    """Get current VRAM usage in GB"""
    device = get_device()
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    elif device.type == "mps":
        # MPS doesn't have direct memory query, return 0
        return 0.0
    return 0.0


def clear_gpu_cache(device: torch.device = None):
    """Clear GPU cache for device"""
    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def configure_attention(model_name: str, device: torch.device) -> Dict:
    """
    Configure attention mechanism based on device.
    MPS doesn't support flash attention, CUDA can use it.
    """
    if device.type == "mps":
        # MPS doesn't support flash attention
        return {"attn_implementation": "eager"}
    elif device.type == "cuda":
        # Try flash attention, fallback to eager if not available
        return {"attn_implementation": "flash_attention_2"}
    return {}


def load_model_and_tokenizer(
    model_name: str,
    use_8bit: bool = True,
    device: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load model and tokenizer with platform-specific optimizations.

    Args:
        model_name: HuggingFace model identifier
        use_8bit: Whether to use 8-bit quantization (only works on CUDA)
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine device
    if device == "auto":
        device_obj = get_device()
    else:
        device_obj = torch.device(device)

    # Configure quantization (only works on CUDA)
    quantization_config = None
    if use_8bit and device_obj.type == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

    # Configure attention
    attention_config = configure_attention(model_name, device_obj)

    # Load model using activation_oracles utilities
    try:
        model = ao_load_model(
            model_name,
            torch.bfloat16,
            quantization_config=quantization_config,
            **attention_config
        )
    except Exception as e:
        # Fallback to eager attention if flash attention fails
        if "flash" in str(e).lower():
            print(f"Flash attention failed, falling back to eager: {e}")
            model = ao_load_model(
                model_name,
                torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation="eager"
            )
        else:
            raise

    # Load tokenizer
    tokenizer = ao_load_tokenizer(model_name)

    # Move model to device if not using quantization
    if quantization_config is None:
        model = model.to(device_obj)

    return model, tokenizer, device_obj


def load_oracle_adapter(model: AutoModelForCausalLM, oracle_path: str) -> str:
    """
    Load oracle LoRA adapter.

    Args:
        model: The base model (must be a PeftModel)
        oracle_path: HuggingFace path to oracle checkpoint

    Returns:
        Name of the loaded adapter (sanitized)
    """
    # Sanitize adapter name (replace dots with underscores)
    adapter_name = oracle_path.replace("/", "_").replace(".", "_")

    # Load adapter if not already loaded
    if not hasattr(model, 'peft_config') or adapter_name not in model.peft_config:
        model.load_adapter(oracle_path, adapter_name=adapter_name)

    return adapter_name


def get_available_models() -> List[str]:
    """Get list of models optimized for 12GB VRAM"""
    return list(SUPPORTED_MODELS.keys())


def get_oracle_checkpoint(model_name: str) -> str:
    """Get oracle checkpoint path for a model"""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]["oracle"]
    raise ValueError(f"No oracle checkpoint found for model: {model_name}")


def get_model_info(model_name: str) -> Dict:
    """Get model information (VRAM usage, oracle path, etc.)"""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    return {}


def estimate_vram_usage(model_name: str, use_8bit: bool) -> float:
    """Estimate VRAM usage in GB"""
    info = get_model_info(model_name)
    if not info:
        return 0.0

    return info["vram_8bit"] if use_8bit else info["vram_fp16"]
