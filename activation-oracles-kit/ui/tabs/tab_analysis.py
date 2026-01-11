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
    selected_positions, layer_nums = token_selector.render()

    # Update state
    set_state("current_layers", layer_nums)

    # Handle activation capture trigger
    if st.session_state.get("trigger_capture", False):
        capture_layers = st.session_state.get("capture_layers", layer_nums)
        _capture_activations(trace, selected_positions, capture_layers)
        st.session_state["trigger_capture"] = False
        # Don't rerun here - let the user see the success/error message
        # The oracle interface will appear automatically after the next interaction

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

        # Check if manually stopped
        manually_stopped = trace.metadata.get("manually_stopped", False)
        status = "⏹️ Stopped" if manually_stopped else "✅ Completed"

        option = f"{i+1}. {scenario_name} - {trace.persona} - {timestamp} {score_info} [{status}]"
        options.append(option)

    # Use the trace display strings as options directly for better selection
    selected_option = st.selectbox(
        "Select a trace to analyze:",
        options,
        key="trace_selection"
    )

    # Find the index of the selected option
    selected_idx = options.index(selected_option)

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

            # Show completion status
            manually_stopped = trace.metadata.get("manually_stopped", False)
            if manually_stopped:
                stop_round = trace.metadata.get("stop_round", "?")
                st.metric("Status", f"⏹️ Stopped (R{stop_round})")
            else:
                st.metric("Status", "✅ Completed")

        # Show activations info separately
        if trace.activations_captured:
            if trace.activation_layers:
                layers_str = ", ".join(map(str, trace.activation_layers))
                st.info(f"✅ Activations captured from {len(trace.activation_layers)} layer(s): {layers_str}")
            elif trace.activation_layer is not None:
                st.info(f"✅ Activations captured from layer {trace.activation_layer}")
        else:
            st.caption("ℹ️ No activations captured yet")


def _capture_activations(trace, selected_positions, layer_nums):
    """
    Capture activations for selected tokens from multiple layers.

    Args:
        trace: ConversationTrace object
        selected_positions: List of token positions
        layer_nums: List of layer numbers to capture from
    """
    if not selected_positions:
        st.warning("⚠️ No tokens selected! Please select tokens first.")
        return

    if not layer_nums:
        st.warning("⚠️ No layers selected! Please select at least one layer.")
        return

    # Validate prerequisites
    model = get_state("model")
    tokenizer = get_state("tokenizer")
    device = get_state("device")

    if model is None:
        st.error("❌ No model loaded! Please load a model in the Setup tab first.")
        return

    if not trace.formatted_prompt:
        st.error("❌ No formatted prompt in trace! This trace may be incomplete.")
        return

    layer_text = f"{len(layer_nums)} layer(s): {layer_nums}" if len(layer_nums) > 1 else f"layer {layer_nums[0]}"
    with st.spinner(f"Capturing {len(selected_positions)} activations from {layer_text}..."):
        try:
            # Initialize activation engine
            engine = ActivationEngine(
                model=model,
                tokenizer=tokenizer,
                device=device
            )

            # Capture activations from multiple layers
            activations_by_layer = engine.capture_activations_multiple_layers(
                trace=trace,
                token_positions=selected_positions,
                layers=layer_nums
            )

            # Store in session state
            set_state("current_activations_multilayer", activations_by_layer)
            set_state("current_layers", layer_nums)

            # For backward compatibility, also set default single layer
            # Use middle layer as default
            default_layer = layer_nums[len(layer_nums) // 2]
            set_state("current_activations", activations_by_layer.get(default_layer, {}))
            set_state("current_layer", default_layer)

            # Clear GPU cache to free memory
            clear_gpu_cache(device)

            st.success(f"✅ Captured {len(selected_positions)} activations from {len(layer_nums)} layer(s)!")

            # Show breakdown by layer
            with st.expander("📊 Capture Details"):
                for layer in sorted(layer_nums):
                    count = len(activations_by_layer.get(layer, {}))
                    st.write(f"- Layer {layer}: {count} activations")

            st.info(f"💡 The Oracle Query interface is now available below. Scroll down to see it!")

        except Exception as e:
            st.error(f"❌ Error capturing activations: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def _render_oracle_interface():
    """Render oracle Q&A interface with layer selection"""
    multilayer_activations = get_state("current_activations_multilayer")

    if not multilayer_activations:
        # Fallback to old single-layer format for backward compatibility
        activations = get_state("current_activations")
        if not activations:
            st.info("No activations captured yet.")
            return
        # Use current_layer for backward compat
        current_layer = get_state("current_layer")
        multilayer_activations = {current_layer: activations} if current_layer is not None else {}

    if not multilayer_activations:
        st.info("No activations captured yet.")
        return

    available_layers = sorted(multilayer_activations.keys())

    # Layer selector for oracle queries
    st.write("**Select Layer for Oracle Query:**")
    selected_layer = st.selectbox(
        "Layer",
        options=available_layers,
        format_func=lambda x: f"Layer {x} ({len(multilayer_activations[x])} activations)",
        help="Choose which layer's activations to query"
    )

    activations = multilayer_activations[selected_layer]

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

        # Show currently selected layer
        st.caption(f"🔍 Querying activations from Layer {selected_layer}")

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
