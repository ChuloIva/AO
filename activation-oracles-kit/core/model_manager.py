"""
Model Manager - Handles model loading with Mac/CUDA/ROCm support and 12GB VRAM optimization
"""

import sys
import os

# Setup ROCm environment variables before importing torch
def setup_rocm_environment():
    """
    Detect and configure ROCm environment for AMD GPUs.
    Must be called before importing torch.

    Reads configuration from config.yaml and applies ROCm environment
    variables from rocm.md for optimal performance on AMD GPUs.
    """
    # Skip if already configured
    if 'HSA_OVERRIDE_GFX_VERSION' in os.environ:
        return False

    # Load config to check if auto-configure is enabled
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = {}
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
    except Exception:
        pass  # Continue with defaults if config load fails

    rocm_config = config.get('rocm', {})
    auto_configure = rocm_config.get('auto_configure', True)

    if not auto_configure:
        return False

    # Get configuration values
    gpu_arch = rocm_config.get('gpu_arch', 'gfx1100')
    gfx_version = rocm_config.get('gfx_version', '11.0.0')
    configured_rocm_home = rocm_config.get('rocm_home', '')

    # Check for ROCm installation paths
    rocm_paths = [
        configured_rocm_home,
        os.environ.get('ROCm_HOME', ''),
        '/opt/rocm',
        '/opt/rocm-6.0.0',
        '/opt/rocm-6.0',
    ]

    rocm_path = None
    for path in rocm_paths:
        if path and os.path.exists(path):
            rocm_path = path
            break

    # Set environment variables for ROCm
    # These improve performance even if full ROCm toolkit isn't installed
    print("Configuring ROCm environment for AMD GPU...")

    # Core ROCm environment variables from rocm.md
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = gfx_version
    os.environ['PYTORCH_ROCM_ARCH'] = gpu_arch

    # Set ROCm home if we found an installation
    if rocm_path:
        os.environ['ROCm_HOME'] = rocm_path
        os.environ['ROCM_VERSION'] = '6.0'
        print(f"  ROCm installation found at: {rocm_path}")

        # ROCtracer library path
        roctracer_lib = f'{rocm_path}/lib/libroctracer64.so'
        if os.path.exists(roctracer_lib):
            os.environ['HSA_TOOLS_LIB'] = roctracer_lib
    else:
        print("  ROCm toolkit not found, using PyTorch ROCm backend")

    # PyTorch ROCm configuration (works with or without full toolkit)
    os.environ['TORCH_BACKEND'] = 'rocm'
    os.environ['HIP_VISIBLE_DEVICES'] = '0'

    # Performance optimizations
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"  GPU Architecture: {gpu_arch}")
    print(f"  GFX Version: {gfx_version}")
    print("ROCm environment configured successfully")
    return True

# Setup ROCm before importing torch
_rocm_configured = setup_rocm_environment()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict

# Try to import BitsAndBytesConfig, but make it optional
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None
    if _rocm_configured:
        print(f"Note: bitsandbytes not available for ROCm - quantization disabled")
        print(f"  (This is expected if rocminfo is not installed)")
    else:
        print(f"Note: bitsandbytes not available - quantization disabled")

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


def is_rocm() -> bool:
    """Check if running on ROCm (AMD GPU)"""
    if not torch.cuda.is_available():
        return False
    # ROCm builds of PyTorch have torch.version.hip attribute
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def get_device() -> torch.device:
    """
    Get appropriate device for current platform.
    Prioritizes: MPS (Mac) > CUDA/ROCm (NVIDIA/AMD) > CPU
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
        gpu_name = torch.cuda.get_device_name(0)
        if is_rocm():
            return f"ROCm/AMD ({gpu_name})"
        else:
            return f"CUDA/NVIDIA ({gpu_name})"
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
    Automatically detects and configures ROCm for AMD GPUs.

    Args:
        model_name: HuggingFace model identifier
        use_8bit: Whether to use 8-bit quantization (works on CUDA/ROCm)
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine device
    if device == "auto":
        device_obj = get_device()
    else:
        device_obj = torch.device(device)

    # Configure quantization (works on CUDA/ROCm, but only if bitsandbytes is available)
    quantization_config = None
    if use_8bit and device_obj.type == "cuda":
        if BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            print("Warning: 8-bit quantization requested but bitsandbytes not available")
            print("  Loading model in full precision instead")

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


def load_oracle_adapter(model: AutoModelForCausalLM, oracle_path: str) -> tuple[AutoModelForCausalLM, str]:
    """
    Load oracle LoRA adapter.

    Args:
        model: The base model
        oracle_path: HuggingFace path to oracle checkpoint

    Returns:
        Tuple of (wrapped_model, adapter_name)
    """
    from peft import PeftModel

    # Sanitize adapter name (replace dots with underscores)
    adapter_name = oracle_path.replace("/", "_").replace(".", "_")

    # Check if model is already a PEFT model
    if hasattr(model, 'peft_config') and adapter_name in model.peft_config:
        return model, adapter_name

    # Wrap model with PEFT adapter
    # Use PeftModel.from_pretrained to get the correct model structure
    if not hasattr(model, 'peft_config'):
        # First time loading adapter - wrap the model
        wrapped_model = PeftModel.from_pretrained(
            model,
            oracle_path,
            adapter_name=adapter_name,
            is_trainable=False
        )
        return wrapped_model, adapter_name
    else:
        # Model already has adapters, add another one
        model.load_adapter(oracle_path, adapter_name=adapter_name, is_trainable=False)
        return model, adapter_name


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
