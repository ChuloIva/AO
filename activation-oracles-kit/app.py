"""
Activation Oracles Research Kit - Main Streamlit Application

A research tool for exploring LLM internals through Activation Oracles.
Core features:
- Run psychological/psychiatric scenarios with different personas
- Capture model activations post-hoc from conversation traces
- Query oracle LLM to understand *why* the model behaved as it did
- Fully customizable prompts for experimentation
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import UI components
from ui.state_manager import init_session_state, is_setup_complete
from ui.tabs import tab_setup, tab_scenario, tab_analysis

# Page configuration
st.set_page_config(
    page_title="Activation Oracles Research Kit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
init_session_state()

# Header
st.title("🧠 Activation Oracles Research Kit")
st.markdown("""
**Explore LLM internals through Activation Oracles**

This tool enables you to:
- 🎭 Run scenarios with different persona prompts
- 🔬 Capture model activations from conversations
- 🔮 Query an oracle LLM to understand model behavior
- ✏️ Fully customize prompts for experimentation
""")

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Setup",
    "🎭 Scenarios",
    "🔬 Analysis",
    "💬 Free Chat"
])

# Render tabs
with tab1:
    tab_setup.render()

with tab2:
    tab_scenario.render()

with tab3:
    tab_analysis.render()

with tab4:
    st.header("💬 Free Chat")
    st.info("📝 Free chat mode coming in Phase 3")
    st.markdown("""
    **Planned features:**
    - Chat with model using custom prompts
    - Persona presets
    - Prompt editor
    - Trace recording for later analysis
    """)

# Footer
st.divider()
with st.expander("ℹ️ About"):
    st.markdown("""
    ## Activation Oracles Research Kit

    **Version:** 0.1.0 (Phase 1 MVP)

    **Based on:**
    - Paper: [Activation Oracles](https://arxiv.org/abs/2512.15674)
    - Code: [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)

    **Current Phase:** Phase 2 - Activation Capture & Oracle Querying
    - ✅ Model loading (12GB VRAM optimized)
    - ✅ Mac support (PyTorch + MPS)
    - ✅ OpenRouter World LLM integration
    - ✅ Epistemic Doctor scenario
    - ✅ Post-hoc activation capture
    - ✅ Oracle queries with activation injection
    - ⏳ Free chat (Phase 3)

    **Documentation:**
    - [GitHub Repository](#)
    - [User Guide](#)
    - [API Documentation](#)
    """)

# Debug sidebar (only show if enabled)
if st.sidebar.checkbox("🔧 Show Debug Info", value=False):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({
        "setup_complete": is_setup_complete(),
        "model_loaded": st.session_state.get("model") is not None,
        "model_name": st.session_state.get("model_name", ""),
        "device": str(st.session_state.get("device", "")),
        "world_llm_configured": st.session_state.get("world_llm") is not None,
        "scenario_running": st.session_state.get("scenario_running", False),
        "current_scenario": st.session_state.get("current_scenario", {}).name if st.session_state.get("current_scenario") else None,
        "trace_history_count": len(st.session_state.get("trace_history", [])),
    })

    if st.sidebar.button("🗑️ Reset All State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.sidebar.success("State reset!")
        st.rerun()
