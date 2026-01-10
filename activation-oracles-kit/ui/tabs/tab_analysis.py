"""
Analysis Tab - Post-hoc activation capture and oracle querying
"""

import streamlit as st
from pathlib import Path
import json

from ui.state_manager import get_state, set_state
from core.activation_engine import ActivationEngine
from core.oracle_interface import OracleInterface
from ui.components.token_selector import TokenSelector
from ui.components.oracle_chat import OracleChat
from core.model_manager import clear_gpu_cache


def render():
    """Render the Analysis tab"""
    st.header("🔬 Activation Analysis")

    # Check prerequisites
    if not get_state("model"):
        st.warning("⚠️ Please load a model in the **Setup** tab first.")
        return

    # Check if there are any traces
    trace_history = get_state("trace_history", [])

    if not trace_history:
        st.info("""
        📝 **No conversation traces available.**

        Run a scenario in the **Scenarios** tab first, then return here to analyze the activations.
        """)
        return

    # Section 1: Trace Selection
    st.subheader("1. Select Conversation Trace")
    trace = _select_trace(trace_history)

    if not trace:
        return

    # Display trace metadata
    _display_trace_info(trace)

    st.divider()

    # Section 2: Token Selection
    st.subheader("2. Select Tokens for Analysis")

    token_selector = TokenSelector(trace, get_state("tokenizer"))
    selected_positions, layer_num = token_selector.render()

    # Update state
    set_state("current_layer", layer_num)

    # Handle activation capture trigger
    if st.session_state.get("trigger_capture", False):
        _capture_activations(trace, selected_positions, layer_num)
        st.session_state["trigger_capture"] = False
        st.rerun()

    st.divider()

    # Section 3: Oracle Query Interface
    current_activations = get_state("current_activations")

    if current_activations:
        st.subheader("3. Query Oracle")
        _render_oracle_interface()

        st.divider()

    # Section 4: Export
    st.subheader("4. Export Results")
    _render_export_section(trace)


def _select_trace(trace_history):
    """
    Render trace selection UI.

    Args:
        trace_history: List of ConversationTrace objects

    Returns:
        Selected trace or None
    """
    # Create dropdown options with metadata
    options = []
    for i, trace in enumerate(trace_history):
        scenario_name = trace.scenario_id or "Free Chat"
        timestamp = trace.timestamp.strftime('%Y-%m-%d %H:%M')
        score_info = f"Score: {trace.final_score:.1f}" if trace.final_score is not None else ""
        option = f"{i+1}. {scenario_name} - {trace.persona} - {timestamp} {score_info}"
        options.append(option)

    selected_idx = st.selectbox(
        "Select a trace to analyze:",
        range(len(options)),
        format_func=lambda i: options[i],
        key="trace_selection"
    )

    return trace_history[selected_idx]


def _display_trace_info(trace):
    """
    Display trace metadata.

    Args:
        trace: ConversationTrace object
    """
    with st.expander("📊 Trace Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Scenario", trace.scenario_id or "Free Chat")
            st.metric("Persona", trace.persona)

        with col2:
            st.metric("Messages", len(trace.messages))
            st.metric("Tokens", len(trace.token_ids))

        with col3:
            model_display = trace.model_name.split("/")[-1] if "/" in trace.model_name else trace.model_name
            st.metric("Model", model_display)
            oracle_display = "Loaded" if trace.oracle_name else "None"
            st.metric("Oracle", oracle_display)

        with col4:
            if trace.final_score is not None:
                st.metric("Score", f"{trace.final_score:.1f}")

            if trace.activations_captured:
                st.metric("Activations", f"✅ Layer {trace.activation_layer}")
            else:
                st.metric("Activations", "Not captured")


def _capture_activations(trace, selected_positions, layer_num):
    """
    Capture activations for selected tokens.

    Args:
        trace: ConversationTrace object
        selected_positions: List of token positions
        layer_num: Layer to capture from
    """
    if not selected_positions:
        st.warning("⚠️ No tokens selected! Please select tokens first.")
        return

    with st.spinner(f"Capturing {len(selected_positions)} activations from layer {layer_num}..."):
        try:
            # Initialize activation engine
            engine = ActivationEngine(
                model=get_state("model"),
                tokenizer=get_state("tokenizer"),
                device=get_state("device")
            )

            # Capture activations
            activations = engine.capture_activations(
                trace=trace,
                token_positions=selected_positions,
                layer=layer_num,
                use_cache=True
            )

            # Store in session state
            set_state("current_activations", activations)
            set_state("current_layer", layer_num)

            # Clear GPU cache to free memory
            clear_gpu_cache(get_state("device"))

            st.success(f"✅ Captured {len(activations)} activations from layer {layer_num}!")

        except Exception as e:
            st.error(f"❌ Error capturing activations: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def _render_oracle_interface():
    """Render oracle Q&A interface"""
    activations = get_state("current_activations")

    if not activations:
        st.info("No activations captured yet.")
        return

    try:
        # Initialize oracle interface
        oracle = OracleInterface(
            model=get_state("model"),
            tokenizer=get_state("tokenizer"),
            oracle_adapter_name=get_state("oracle_adapter_name"),
            device=get_state("device"),
            injection_layer=1,
            steering_coefficient=1.0
        )

        # Render oracle chat component
        chat = OracleChat(oracle, activations)
        chat.render()

    except Exception as e:
        st.error(f"❌ Error initializing oracle: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def _render_export_section(trace):
    """
    Render export options.

    Args:
        trace: ConversationTrace object
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export Trace (JSON)", use_container_width=True):
            _export_trace(trace)

    with col2:
        if st.button("💾 Export Activations", use_container_width=True):
            _export_activations(trace)

    with col3:
        if st.button("💬 Export Oracle Chat", use_container_width=True):
            _export_oracle_chat()


def _export_trace(trace):
    """Export conversation trace to JSON"""
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / f"{trace.trace_id}.json"

    try:
        trace.save(export_path)
        st.success(f"✅ Trace exported to `{export_path}`")

        # Show download button
        with open(export_path, 'r') as f:
            json_str = f.read()

        st.download_button(
            label="Download Trace JSON",
            data=json_str,
            file_name=f"trace_{trace.trace_id}.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Export failed: {e}")


def _export_activations(trace):
    """Export activations to NPZ"""
    if not trace.activations_captured or not trace.activation_file_path:
        st.warning("⚠️ No activations captured for this trace.")
        return

    try:
        activation_path = Path(trace.activation_file_path)

        if not activation_path.exists():
            st.error(f"❌ Activation file not found: {activation_path}")
            return

        st.success(f"✅ Activations available at: `{activation_path}`")

        # Show file info
        file_size = activation_path.stat().st_size / (1024 * 1024)  # MB
        st.info(f"File size: {file_size:.2f} MB")

        # Read file for download
        with open(activation_path, 'rb') as f:
            npz_data = f.read()

        st.download_button(
            label="Download Activations NPZ",
            data=npz_data,
            file_name=f"activations_{trace.trace_id}_layer{trace.activation_layer}.npz",
            mime="application/octet-stream"
        )

    except Exception as e:
        st.error(f"❌ Export failed: {e}")


def _export_oracle_chat():
    """Export oracle chat history to JSON"""
    history = st.session_state.get("oracle_chat_history", [])

    if not history:
        st.warning("⚠️ No oracle chat history to export.")
        return

    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / "oracle_chat.json"

    try:
        with open(export_path, "w") as f:
            json.dump(history, f, indent=2)

        st.success(f"✅ Oracle chat exported to `{export_path}`")

        # Show download button
        json_str = json.dumps(history, indent=2)

        st.download_button(
            label="Download Oracle Chat JSON",
            data=json_str,
            file_name="oracle_chat.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Export failed: {e}")
