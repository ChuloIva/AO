"""
Setup Tab - Model loading, Oracle configuration, and World LLM setup
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_manager import (
    load_model_and_tokenizer,
    load_oracle_adapter,
    get_available_models,
    get_oracle_checkpoint,
    get_model_info,
    get_device_name,
    get_vram_usage,
    estimate_vram_usage
)
from scenarios.world_llm import WorldLLM, get_recommended_models
from ui.state_manager import update_state, is_model_loaded, is_world_llm_configured


def render():
    """Render the Setup tab"""
    st.header("⚙️ Setup")

    st.markdown("""
    **Configure your research environment:**
    1. Select and load a language model (subject LLM)
    2. Configure OpenRouter API for World LLM simulation
    3. Once setup is complete, proceed to the Scenarios or Free Chat tab
    """)

    st.divider()

    # Device info
    col1, col2 = st.columns(2)
    with col1:
        device_name = get_device_name()
        st.info(f"🖥️ **Device:** {device_name}")

    with col2:
        if is_model_loaded():
            vram = get_vram_usage()
            if vram > 0:
                st.info(f"💾 **VRAM Used:** {vram:.2f} GB")
            else:
                st.info("💾 **VRAM:** Not available (MPS/CPU)")

    st.divider()

    # Model Selection
    st.subheader("1. Subject LLM (Model to Analyze)")

    available_models = get_available_models()

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "Base Model",
            available_models,
            index=0,  # Default to first model (Qwen3-1.7B)
            help="Select a model optimized for 12GB VRAM"
        )

    with col2:
        use_8bit = st.checkbox(
            "Use 8-bit quantization",
            value=True,
            help="Reduces VRAM usage by ~50% (CUDA only)"
        )

    # Show model info
    model_info = get_model_info(model_name)
    if model_info:
        estimated_vram = estimate_vram_usage(model_name, use_8bit)
        oracle_checkpoint = get_oracle_checkpoint(model_name)

        st.markdown(f"""
        **Model Info:**
        - Estimated VRAM: ~{estimated_vram}GB {'(8-bit)' if use_8bit else '(FP16)'}
        - Layers: {model_info.get('layers', 'Unknown')}
        - Oracle: `{oracle_checkpoint}`
        """)

    # Load button
    if st.button("🚀 Load Model", type="primary", use_container_width=True):
        with st.spinner(f"Loading {model_name}... This may take a minute."):
            try:
                # Load model and tokenizer
                model, tokenizer, device = load_model_and_tokenizer(
                    model_name=model_name,
                    use_8bit=use_8bit
                )

                # Load oracle adapter
                oracle_checkpoint = get_oracle_checkpoint(model_name)
                adapter_name = load_oracle_adapter(model, oracle_checkpoint)

                # Update state
                update_state({
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": device,
                    "model_name": model_name,
                    "oracle_adapter_name": adapter_name,
                    "oracle_checkpoint": oracle_checkpoint
                })

                st.success(f"✅ Model loaded successfully!")
                st.success(f"✅ Oracle adapter loaded: {adapter_name}")

            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # OpenRouter Configuration
    st.subheader("2. World LLM (OpenRouter API)")

    st.markdown("""
    The World LLM simulates the environment (patient, opponent, etc.) in scenarios.
    Get your API key at [openrouter.ai](https://openrouter.ai)
    """)

    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.get("openrouter_api_key", ""),
            help="Your OpenRouter API key (starts with 'sk-or-...')"
        )

    with col2:
        # Get recommended models
        recommended = get_recommended_models()
        model_options = {
            f"{info['id']} ({info['cost']})": info['id']
            for name, info in recommended.items()
        }

        selected_display = st.selectbox(
            "World LLM Model",
            list(model_options.keys()),
            index=0,
            help="Select model for environment simulation"
        )
        openrouter_model = model_options[selected_display]

    # Show model description
    for name, info in recommended.items():
        if info['id'] == openrouter_model:
            st.caption(f"ℹ️ {info['description']}")
            break

    # Test connection button
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔌 Test Connection", use_container_width=True):
            if not api_key:
                st.warning("Please enter an API key first")
            else:
                with st.spinner("Testing connection..."):
                    try:
                        world_llm = WorldLLM(api_key=api_key, model=openrouter_model)
                        success = world_llm.test_connection()

                        if success:
                            st.success("✅ Connection successful!")
                            # Save to state
                            update_state({
                                "openrouter_api_key": api_key,
                                "openrouter_model": openrouter_model,
                                "world_llm": world_llm
                            })
                        else:
                            st.error("❌ Connection failed. Check your API key.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            if not api_key:
                st.warning("Please enter an API key first")
            else:
                world_llm = WorldLLM(api_key=api_key, model=openrouter_model)
                update_state({
                    "openrouter_api_key": api_key,
                    "openrouter_model": openrouter_model,
                    "world_llm": world_llm,
                    "setup_complete": True
                })
                st.success("✅ Configuration saved!")

    st.divider()

    # Setup Status
    st.subheader("Setup Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if is_model_loaded():
            st.success("✅ Model loaded")
        else:
            st.warning("⏳ Model not loaded")

    with col2:
        if is_world_llm_configured():
            st.success("✅ World LLM configured")
        else:
            st.warning("⏳ World LLM not configured")

    with col3:
        if is_model_loaded() and is_world_llm_configured():
            st.success("✅ Ready to start!")
        else:
            st.info("ℹ️ Complete setup above")

    # Debug info (collapsible)
    with st.expander("🔧 Debug Info"):
        st.json({
            "model_loaded": is_model_loaded(),
            "model_name": st.session_state.get("model_name", ""),
            "device": str(st.session_state.get("device", "")),
            "oracle_adapter": st.session_state.get("oracle_adapter_name", ""),
            "world_llm_configured": is_world_llm_configured(),
            "world_llm_model": st.session_state.get("openrouter_model", ""),
        })
