"""
Generation Module - Handles model generation with proper tokenization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    device: torch.device = None,
    generation_kwargs: Optional[Dict] = None
) -> Tuple[str, List[int], str]:
    """
    Generate response from model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
        system_prompt: System prompt to prepend
        device: Device to use for generation
        generation_kwargs: Additional generation parameters

    Returns:
        Tuple of (response_text, full_token_ids, formatted_prompt)
    """
    if device is None:
        device = next(model.parameters()).device

    # Default generation kwargs
    default_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if generation_kwargs:
        default_kwargs.update(generation_kwargs)

    # Prepare messages with system prompt
    formatted_messages = messages.copy()
    if system_prompt:
        # Insert system message at the beginning
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})

    # Apply chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback if chat template not available
        print(f"Chat template not available, using simple formatting: {e}")
        formatted_prompt = format_messages_simple(formatted_messages)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **default_kwargs
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response_only = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Get full token IDs
    full_token_ids = outputs[0].tolist()

    return response_only.strip(), full_token_ids, formatted_prompt


def format_messages_simple(messages: List[Dict[str, str]]) -> str:
    """
    Simple message formatting fallback when chat template is not available.

    Args:
        messages: List of message dicts

    Returns:
        Formatted string
    """
    formatted = ""
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        formatted += f"[{role}] {content}\n\n"

    # Add prompt for assistant response
    formatted += "[ASSISTANT] "
    return formatted


def prepare_for_replay(
    tokenizer: AutoTokenizer,
    formatted_prompt: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepare formatted prompt for replay (activation capture).

    Args:
        tokenizer: The tokenizer
        formatted_prompt: Previously generated formatted prompt
        device: Device to use

    Returns:
        Tokenized inputs ready for model
    """
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    return inputs


def extract_assistant_response_from_output(
    output_text: str,
    input_text: str
) -> str:
    """
    Extract just the assistant's response from full output.

    Args:
        output_text: Full model output
        input_text: Input prompt text

    Returns:
        Assistant's response only
    """
    # Remove input prompt from output
    if output_text.startswith(input_text):
        response = output_text[len(input_text):]
    else:
        response = output_text

    # Clean up
    return response.strip()


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Count number of tokens in text"""
    return len(tokenizer.encode(text))


def decode_tokens(tokenizer: AutoTokenizer, token_ids: List[int], skip_special: bool = True) -> str:
    """Decode token IDs to text"""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special)


def decode_token_by_token(tokenizer: AutoTokenizer, token_ids: List[int]) -> List[str]:
    """Decode each token individually"""
    return [tokenizer.decode([tid]) for tid in token_ids]
