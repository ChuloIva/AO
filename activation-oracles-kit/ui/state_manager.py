"""
State Manager - Centralized Streamlit session state management
"""

import streamlit as st
from typing import Any, Dict


def init_session_state():
    """
    Initialize all session state variables with default values.
    Call this at the start of app.py.
    """
    defaults = {
        # Model state
        "model": None,
        "tokenizer": None,
        "device": None,
        "oracle_adapter_name": None,
        "model_name": "",
        "oracle_checkpoint": "",

        # OpenRouter / World LLM state
        "openrouter_api_key": "",
        "openrouter_model": "anthropic/claude-3.5-sonnet",
        "world_llm": None,

        # Scenario state
        "current_scenario": None,
        "available_scenarios": [],

        # Conversation state
        "current_trace": None,
        "trace_history": [],
        "scenario_messages": [],  # Messages for current scenario

        # Free chat state
        "free_chat_messages": [],

        # Analysis state (Phase 2)
        "selected_tokens": [],
        "selected_token_positions": set(),
        "current_activations": {},
        "current_layer": None,
        "cached_activations": {},
        "oracle_chat_history": [],
        "trigger_capture": False,

        # Prompt editing state
        "current_prompt": "",
        "selected_preset": "baseline",

        # UI state
        "setup_complete": False,
        "scenario_running": False,
        "active_tab": "Setup",

        # Error handling
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """
    Get session state value.

    Args:
        key: State key
        default: Default value if key doesn't exist

    Returns:
        State value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """
    Set session state value.

    Args:
        key: State key
        value: Value to set
    """
    st.session_state[key] = value


def update_state(updates: Dict[str, Any]):
    """
    Update multiple state values at once.

    Args:
        updates: Dict of key-value pairs to update
    """
    for key, value in updates.items():
        st.session_state[key] = value


def clear_state(key: str):
    """
    Clear/reset a state value to None.

    Args:
        key: State key to clear
    """
    if key in st.session_state:
        st.session_state[key] = None


def is_model_loaded() -> bool:
    """Check if model is loaded"""
    return st.session_state.get("model") is not None


def is_world_llm_configured() -> bool:
    """Check if World LLM is configured"""
    return (
        st.session_state.get("openrouter_api_key", "") != "" and
        st.session_state.get("world_llm") is not None
    )


def is_setup_complete() -> bool:
    """Check if initial setup is complete"""
    return (
        is_model_loaded() and
        is_world_llm_configured() and
        st.session_state.get("setup_complete", False)
    )


def reset_scenario_state():
    """Reset scenario-related state"""
    update_state({
        "current_scenario": None,
        "scenario_messages": [],
        "current_trace": None,
        "scenario_running": False
    })


def reset_all_state():
    """Reset all state (for debugging/starting over)"""
    keys_to_keep = []  # Could preserve some keys if needed

    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    # Reinitialize
    init_session_state()
