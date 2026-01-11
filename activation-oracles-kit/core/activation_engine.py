"""
Activation Engine - Post-hoc activation capture using PyTorch hooks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from pathlib import Path
import logging

from core.conversation_trace import ConversationTrace
from core.activation_cache import ActivationCache, CapturedActivation
from core.generation import prepare_for_replay, decode_token_by_token

# Import from activation_oracles
from nl_probes.utils.common import layer_percent_to_layer

logger = logging.getLogger(__name__)


def get_layer_module(
    model: AutoModelForCausalLM,
    layer_num: int,
    use_lora: bool = False
) -> torch.nn.Module:
    """
    Get the residual stream module for a specific layer.

    Based on activation_oracles/nl_probes/utils/activation_utils.py:get_hf_submodule

    Supports: Qwen, Llama, Gemma architectures

    Args:
        model: The model
        layer_num: Layer index (0-indexed)
        use_lora: Whether model has LoRA adapters loaded

    Returns:
        The layer module to hook

    Raises:
        ValueError: If model architecture is unsupported
    """
    model_name = model.config._name_or_path

    if use_lora:
        # PEFT/LoRA models
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma-3" in model_name:
            return model.base_model.language_model.layers[layer_num]
        elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer_num]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")
    else:
        # Base models (no PEFT)
        if "pythia" in model_name:
            return model.gpt_neox.layers[layer_num]
        elif "gemma-3" in model_name:
            return model.language_model.layers[layer_num]
        elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.model.layers[layer_num]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")


def calculate_layer_from_percent(
    model_name: str,
    layer_percent: int,
    total_layers: int
) -> int:
    """
    Calculate layer index from percentage.

    Args:
        model_name: HuggingFace model identifier
        layer_percent: Percentage (0-100)
        total_layers: Total number of layers in the model

    Returns:
        Layer index (0-indexed)
    """
    # Use activation_oracles utility if available
    try:
        return layer_percent_to_layer(model_name, layer_percent)
    except Exception as e:
        # Fallback: simple percentage calculation
        logger.warning(f"Using fallback layer calculation: {e}")
        return int(total_layers * (layer_percent / 100))


class ActivationEngine:
    """Main engine for capturing activations post-hoc"""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        cache_dir: Path = None
    ):
        """
        Initialize activation engine.

        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to use
            cache_dir: Directory for activation cache (default: data/activations)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if cache_dir is None:
            cache_dir = Path("data/activations")

        self.cache = ActivationCache(cache_dir)

        # Detect if model has LoRA adapters
        self.use_lora = self._detect_lora()

    def _detect_lora(self) -> bool:
        """
        Detect if model has LoRA adapters loaded.

        Returns:
            True if model has LoRA adapters, False otherwise
        """
        # Check if model is a PEFT model
        return hasattr(self.model, 'peft_config') and self.model.peft_config is not None

    def capture_activations(
        self,
        trace: ConversationTrace,
        token_positions: List[int],
        layer: int,
        use_cache: bool = True
    ) -> Dict[int, CapturedActivation]:
        """
        Capture activations at specified token positions by replaying conversation.

        Args:
            trace: The conversation trace to replay
            token_positions: List of token positions to capture (absolute indices)
            layer: Which layer to extract from
            use_cache: Whether to use cached activations if available

        Returns:
            Dict mapping token position -> CapturedActivation

        Raises:
            ValueError: If token positions are invalid
        """
        if not token_positions:
            raise ValueError("No token positions specified")

        # Validate token positions
        max_position = len(trace.token_ids) - 1
        invalid_positions = [p for p in token_positions if p < 0 or p > max_position]
        if invalid_positions:
            raise ValueError(
                f"Invalid token positions: {invalid_positions}. "
                f"Valid range: 0-{max_position}"
            )

        # Check cache first
        if use_cache and self.cache.has_activations(trace.trace_id, token_positions, layer):
            logger.info(f"Loading {len(token_positions)} activations from cache")
            return self.cache.load_activations(trace.trace_id, layer)

        logger.info(f"Capturing {len(token_positions)} activations from layer {layer}")

        # Check adapter status
        if hasattr(self.model, 'active_adapter'):
            logger.info(f"Model has active adapter: {self.model.active_adapter}")
        if hasattr(self.model, 'peft_config'):
            logger.info(f"Model has PEFT config with adapters: {list(self.model.peft_config.keys())}")

        # Prepare inputs by replaying conversation
        # CRITICAL: Pass token_ids to replay the FULL sequence (prompt + generated tokens)
        inputs = prepare_for_replay(
            self.tokenizer,
            trace.formatted_prompt,
            self.device,
            token_ids=trace.token_ids  # Use exact token IDs from generation
        )
        logger.info(f"Prepared inputs: input_ids shape={inputs['input_ids'].shape}")
        logger.info(f"  Token count: {len(trace.token_ids)}, Sequence length: {inputs['input_ids'].shape[1]}")

        # Get layer module to hook
        try:
            layer_module = get_layer_module(self.model, layer, use_lora=self.use_lora)
        except ValueError as e:
            logger.error(f"Failed to get layer module: {e}")
            raise

        # Set up hook to capture activations
        captured_acts = {}
        hook_called = [False]  # Use list to modify in closure

        def capture_hook(module, input, output):
            """Hook to capture residual stream activations"""
            hook_called[0] = True
            logger.info(f"Hook called for layer {layer}")

            # Output shape: [batch, seq_len, hidden_dim]
            if isinstance(output, tuple):
                activations = output[0]
                logger.info(f"  Output is tuple, using first element")
            else:
                activations = output
                logger.info(f"  Output is tensor")

            logger.info(f"  Activations shape: {activations.shape}")
            logger.info(f"  Requested positions: {token_positions}")
            logger.info(f"  Max position requested: {max(token_positions) if token_positions else 'N/A'}")

            # Extract specific positions
            captured_count = 0
            for pos in token_positions:
                if pos < activations.shape[1]:
                    captured_acts[pos] = activations[0, pos, :].detach().cpu()
                    captured_count += 1
                else:
                    logger.warning(f"  Position {pos} >= sequence length {activations.shape[1]}, skipping")

            logger.info(f"  Captured {captured_count} activations")

        # Register hook and run forward pass
        logger.info(f"Registering hook on layer module: {type(layer_module).__name__}")
        handle = layer_module.register_forward_hook(capture_hook)

        try:
            logger.info(f"Running forward pass with inputs shape: {inputs['input_ids'].shape}")
            with torch.no_grad():
                _ = self.model(**inputs)
            logger.info(f"Forward pass completed. Hook called: {hook_called[0]}")
        finally:
            handle.remove()

        # Convert to CapturedActivation objects with metadata
        logger.info(f"After forward pass, captured_acts has {len(captured_acts)} items")
        token_texts = decode_token_by_token(self.tokenizer, trace.token_ids)

        results = {}
        for pos, act_tensor in captured_acts.items():
            # Find which message this token belongs to
            msg_idx = self._find_message_for_position(trace, pos)

            # Convert to float32 before numpy conversion for MPS/bfloat16 compatibility
            act_numpy = act_tensor.float().numpy() if act_tensor.dtype == torch.bfloat16 else act_tensor.numpy()

            results[pos] = CapturedActivation(
                layer=layer,
                token_position=pos,
                activation=act_numpy,
                token_text=token_texts[pos] if pos < len(token_texts) else "",
                message_idx=msg_idx
            )

        # Save to cache
        self.cache.save_activations(trace.trace_id, results, layer)

        # Update trace metadata
        trace.activations_captured = True
        trace.activation_layer = layer
        trace.activation_positions = list(token_positions)
        trace.activation_file_path = str(self.cache._get_cache_path(trace.trace_id, layer))

        logger.info(f"Successfully captured {len(results)} activations")
        return results

    def _find_message_for_position(
        self,
        trace: ConversationTrace,
        pos: int
    ) -> int:
        """
        Find which message a token position belongs to.

        Args:
            trace: Conversation trace
            pos: Token position

        Returns:
            Message index, or -1 if not found
        """
        for msg_idx, token_range in trace.message_to_tokens.items():
            start, end = token_range
            if start <= pos < end:
                return msg_idx

        return -1

    def capture_activations_multiple_layers(
        self,
        trace: ConversationTrace,
        token_positions: List[int],
        layers: List[int]
    ) -> Dict[int, Dict[int, CapturedActivation]]:
        """
        Capture activations from multiple layers in a single forward pass.

        Args:
            trace: Conversation trace to replay
            token_positions: List of token positions to capture
            layers: List of layer indices

        Returns:
            Dict[layer -> Dict[position -> CapturedActivation]]
        """
        if not token_positions:
            raise ValueError("No token positions specified")

        if not layers:
            raise ValueError("No layers specified")

        logger.info(f"Capturing {len(token_positions)} activations from {len(layers)} layers")

        # Prepare inputs - use exact token IDs from generation
        inputs = prepare_for_replay(
            self.tokenizer,
            trace.formatted_prompt,
            self.device,
            token_ids=trace.token_ids
        )

        # Set up hooks for all layers
        captured_acts = {layer: {} for layer in layers}

        def make_capture_hook(layer_num):
            """Create a capture hook for a specific layer"""
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output

                for pos in token_positions:
                    if pos < activations.shape[1]:
                        captured_acts[layer_num][pos] = activations[0, pos, :].detach().cpu()

            return hook

        # Register hooks for all layers
        handles = []
        for layer in layers:
            try:
                layer_module = get_layer_module(self.model, layer, use_lora=self.use_lora)
                handle = layer_module.register_forward_hook(make_capture_hook(layer))
                handles.append(handle)
            except ValueError as e:
                logger.error(f"Failed to get layer {layer}: {e}")

        # Run forward pass once
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()

        # Convert to CapturedActivation objects
        token_texts = decode_token_by_token(self.tokenizer, trace.token_ids)

        results = {}
        for layer in layers:
            results[layer] = {}
            for pos, act_tensor in captured_acts[layer].items():
                msg_idx = self._find_message_for_position(trace, pos)

                # Convert to float32 before numpy conversion for MPS/bfloat16 compatibility
                act_numpy = act_tensor.float().numpy() if act_tensor.dtype == torch.bfloat16 else act_tensor.numpy()

                results[layer][pos] = CapturedActivation(
                    layer=layer,
                    token_position=pos,
                    activation=act_numpy,
                    token_text=token_texts[pos] if pos < len(token_texts) else "",
                    message_idx=msg_idx
                )

            # Save each layer to cache
            if results[layer]:
                self.cache.save_activations(trace.trace_id, results[layer], layer)

        logger.info(f"Successfully captured activations from {len(layers)} layers")
        return results
