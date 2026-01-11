"""
Oracle Interface - Query oracle with captured activations using injection
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

from core.activation_cache import CapturedActivation
from core.activation_engine import get_layer_module

# Import from activation_oracles
from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

logger = logging.getLogger(__name__)


# Preset oracle questions
PRESET_QUESTIONS = {
    "what_next": "What word or concept is the model about to say?",
    "confidence": "How confident is the model in its current decision?",
    "reasoning": "What is the model's reasoning at this point?",
    "alternatives": "What alternative options is the model considering?",
    "emotion": "What emotional state or bias is present in the model's processing?",
    "belief": "What does the model believe about the situation?",
    "knowledge": "What knowledge is the model recalling or using?",
    "why_action": "Why did the model choose this specific action?",
    "would_change": "What would make the model change its decision?",
    "rational": "Is the model's decision rational based on the available information?",
}


def get_preset_question(key: str) -> str:
    """
    Get a preset oracle question.

    Args:
        key: Question key (e.g., "what_next", "confidence")

    Returns:
        Question text
    """
    return PRESET_QUESTIONS.get(key, "")


def format_oracle_prompt(
    question: str,
    layer: int,
    num_tokens: int,
    context: str = ""
) -> str:
    """
    Format oracle prompt with placeholder tokens.

    Based on activation_oracles format:
    Layer: {layer}
     ? ? ? ... (num_tokens placeholders)
    {context}
    {question}

    Args:
        question: Question to ask the oracle
        layer: Layer number activations are from
        num_tokens: Number of activation vectors to inject
        context: Optional context string

    Returns:
        Formatted prompt string
    """
    # Create placeholders: " ?" repeated num_tokens times
    placeholders = " ".join([" ?"] * num_tokens)

    prompt_parts = [f"Layer: {layer}"]
    prompt_parts.append(placeholders)

    if context:
        prompt_parts.append(f"Context: {context}")

    prompt_parts.append(question)

    return "\n".join(prompt_parts)


class OracleInterface:
    """Interface for querying oracle with activations"""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        oracle_adapter_name: str,
        device: torch.device,
        injection_layer: int = 1,
        steering_coefficient: float = 1.0
    ):
        """
        Initialize oracle interface.

        Args:
            model: The base model (must have oracle adapter loaded)
            tokenizer: The tokenizer
            oracle_adapter_name: Name of the oracle adapter
            device: Device to use
            injection_layer: Layer to inject activations (default: 1)
            steering_coefficient: Scaling coefficient for injection (default: 1.0)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.oracle_adapter_name = oracle_adapter_name
        self.device = device
        self.injection_layer = injection_layer
        self.steering_coefficient = steering_coefficient

    def query_oracle(
        self,
        activations: List[CapturedActivation],
        question: str,
        context: str = "",
        generation_kwargs: Optional[Dict] = None
    ) -> str:
        """
        Query oracle with captured activations.

        Args:
            activations: List of captured activations to inject
            question: Question to ask
            context: Optional context string
            generation_kwargs: Generation parameters

        Returns:
            Oracle's response

        Raises:
            ValueError: If activations list is empty or from different layers
        """
        if not activations:
            raise ValueError("No activations provided")

        # Verify all activations are from the same layer
        layers = set(act.layer for act in activations)
        if len(layers) > 1:
            raise ValueError(f"Activations from multiple layers: {layers}")

        layer = activations[0].layer

        # Switch to oracle adapter
        try:
            self.model.set_adapter(self.oracle_adapter_name)
        except Exception as e:
            logger.error(f"Failed to set oracle adapter: {e}")
            raise

        # Format oracle prompt
        oracle_prompt = format_oracle_prompt(
            question=question,
            layer=layer,
            num_tokens=len(activations),
            context=context
        )

        # Tokenize oracle prompt
        messages = [{"role": "user", "content": oracle_prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback to simple formatting
            logger.warning(f"Chat template failed, using simple format: {e}")
            formatted = f"User: {oracle_prompt}\n\nAssistant:"

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        # Find placeholder token positions
        placeholder_positions = self._find_placeholder_positions(inputs["input_ids"][0])

        if len(placeholder_positions) != len(activations):
            logger.warning(
                f"Placeholder count mismatch: found {len(placeholder_positions)}, "
                f"expected {len(activations)}"
            )

        # Prepare activation vectors for injection
        activation_tensors = [
            torch.from_numpy(act.activation).to(self.device)
            for act in activations
        ]

        # Pad if necessary
        while len(activation_tensors) < len(placeholder_positions):
            activation_tensors.append(activation_tensors[-1])

        # Truncate if necessary
        activation_tensors = activation_tensors[:len(placeholder_positions)]

        # Stack into 2D tensor: (num_activations, d_model)
        # The hook expects vectors to be a list of 2D tensors (one per batch item)
        activation_tensor_2D = torch.stack(activation_tensors)  # Shape: (K, d_model)

        logger.info(f"Prepared {len(activation_tensors)} activations for injection")
        logger.info(f"  Individual activation shape: {activation_tensors[0].shape}")
        logger.info(f"  Stacked tensor shape: {activation_tensor_2D.shape}")
        logger.info(f"  Placeholder positions: {placeholder_positions}")

        # Create injection hook
        try:
            injection_module = get_layer_module(
                self.model,
                self.injection_layer,
                use_lora=True
            )
        except ValueError as e:
            logger.error(f"Failed to get injection layer: {e}")
            raise

        # CRITICAL: vectors must be a list of 2D tensors, not a list of lists
        # For batch size 1, we pass a single 2D tensor wrapped in a list
        hook_fn = get_hf_activation_steering_hook(
            vectors=[activation_tensor_2D],  # List containing one 2D tensor (K, d_model)
            positions=[placeholder_positions],  # List containing one list of positions
            steering_coefficient=self.steering_coefficient,
            device=self.device,
            dtype=self.model.dtype
        )

        # Default generation kwargs
        default_gen_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if generation_kwargs:
            default_gen_kwargs.update(generation_kwargs)

        # Generate with hook
        with add_hook(injection_module, hook_fn):
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **default_gen_kwargs)

        # Decode response
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response.strip()

    def _find_placeholder_positions(self, input_ids: torch.Tensor) -> List[int]:
        """
        Find positions of placeholder tokens (' ?') in input.

        Args:
            input_ids: Tokenized input tensor

        Returns:
            List of token positions where activations should be injected
        """
        # Tokenize " ?" to find what token ID it maps to
        placeholder_token_ids = self.tokenizer.encode(" ?", add_special_tokens=False)

        if not placeholder_token_ids:
            logger.error("Failed to tokenize placeholder ' ?'")
            return []

        # Use the first token if multiple
        placeholder_token = placeholder_token_ids[0]

        # Find all positions where this token appears
        positions = (input_ids == placeholder_token).nonzero(as_tuple=True)[0].tolist()

        logger.debug(f"Found {len(positions)} placeholder positions: {positions}")
        return positions

    def query_oracle_batch(
        self,
        activations_list: List[List[CapturedActivation]],
        questions: List[str],
        contexts: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict] = None
    ) -> List[str]:
        """
        Query oracle with multiple sets of activations (batch processing).

        Args:
            activations_list: List of activation lists
            questions: List of questions (same length as activations_list)
            contexts: Optional list of contexts
            generation_kwargs: Generation parameters

        Returns:
            List of oracle responses
        """
        if len(activations_list) != len(questions):
            raise ValueError("activations_list and questions must have same length")

        if contexts is None:
            contexts = [""] * len(questions)

        responses = []
        for activations, question, context in zip(activations_list, questions, contexts):
            try:
                response = self.query_oracle(
                    activations=activations,
                    question=question,
                    context=context,
                    generation_kwargs=generation_kwargs
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Oracle query failed: {e}")
                responses.append(f"Error: {str(e)}")

        return responses
