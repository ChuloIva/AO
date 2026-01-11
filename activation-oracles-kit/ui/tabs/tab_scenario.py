"""
Scenario Tab - Run structured scenarios with different personas
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.scenario_loader import get_available_scenarios, load_scenario
from core.conversation_trace import ConversationTrace, Message
from core.generation import generate_response
from ui.state_manager import (
    is_model_loaded,
    is_world_llm_configured,
    get_state,
    update_state
)


def render():
    """Render the Scenario tab"""
    st.header("🎭 Scenarios")

    # Check setup
    if not is_model_loaded() or not is_world_llm_configured():
        st.warning("⚠️ Please complete setup in the Setup tab first")
        return

    st.markdown("""
    Run structured scenarios to test how different persona prompts affect model behavior.
    """)

    st.divider()

    # Scenario selection
    scenario_running = get_state("scenario_running", False)

    if not scenario_running:
        render_scenario_selection()
    else:
        render_scenario_runner()


def render_scenario_selection():
    """Render scenario selection interface"""
    st.subheader("Select Scenario")

    # Get available scenarios
    library_path = Path(__file__).parent.parent.parent / "scenarios" / "library"
    scenarios = get_available_scenarios(library_path)

    if not scenarios:
        st.error("No scenarios found. Check scenarios/library/ directory.")
        return

    # Display scenarios
    col1, col2 = st.columns([2, 1])

    with col1:
        scenario_names = [s["name"] for s in scenarios]
        selected_name = st.selectbox(
            "Scenario",
            scenario_names,
            index=0
        )

        # Find selected scenario info
        selected_scenario_info = next(s for s in scenarios if s["name"] == selected_name)

        st.markdown(f"""
        **Description:** {selected_scenario_info['description']}

        **Category:** {selected_scenario_info['category']}
        """)

    with col2:
        # Persona selection
        personas = selected_scenario_info.get("personas", ["baseline"])
        selected_persona = st.selectbox(
            "Persona",
            personas,
            index=0,
            help="Different personality traits for the subject LLM"
        )

    st.divider()

    # Start button
    if st.button("🚀 Start Scenario", type="primary", use_container_width=True):
        with st.spinner("Loading scenario..."):
            try:
                # Load scenario
                world_llm = get_state("world_llm")
                patient_llm = get_state("patient_llm")
                scenario = load_scenario(
                    selected_scenario_info["file"],
                    world_llm,
                    patient_llm,
                    persona=selected_persona
                )

                # Create conversation trace
                trace = ConversationTrace(
                    scenario_id=scenario.name,
                    persona=selected_persona,
                    system_prompt=scenario.get_subject_prompt(),
                    model_name=get_state("model_name"),
                    oracle_name=get_state("oracle_adapter_name")
                )

                # Initialize scenario
                scenario.initialize()

                # Update state
                update_state({
                    "current_scenario": scenario,
                    "current_trace": trace,
                    "scenario_messages": [],
                    "scenario_running": True,
                    "current_trace_saved": False  # Reset the saved flag for new scenario
                })

                st.success(f"✅ Scenario started: {scenario.name} ({selected_persona})")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error starting scenario: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_scenario_runner():
    """Render active scenario runner"""
    scenario = get_state("current_scenario")
    trace = get_state("current_trace")
    # Get messages and create a new list to ensure proper state tracking
    messages = list(get_state("scenario_messages", []))

    if not scenario:
        st.error("No active scenario")
        update_state({"scenario_running": False})
        st.rerun()
        return

    # Header with scenario info
    state = scenario.get_state()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Scenario", scenario.name)
    with col2:
        st.metric("Round", f"{state.round}/{scenario.max_rounds if hasattr(scenario, 'max_rounds') else '?'}")
    with col3:
        st.metric("Resources", state.resources)
    with col4:
        st.metric("Score", f"{state.score:.1f}")

    st.divider()

    # Display conversation
    st.subheader("Conversation")

    # Display all messages
    for i, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        if role == "patient":
            # Patient dialogue (from PatientLLM)
            st.chat_message("assistant", avatar="🤒").markdown(f"**💬 Patient:** {content}")
        elif role == "world":
            # Legacy world messages (for backward compatibility)
            st.chat_message("assistant", avatar="🏥").write(content)
        elif role == "subject":
            # Doctor (subject LLM) actions/questions
            st.chat_message("user", avatar="👨‍⚕️").write(content)
        elif role == "info":
            # Game state info messages
            st.info(content)
        else:
            st.chat_message(role).write(content)

    # Check if scenario is complete
    if scenario.is_complete():
        st.success("🎉 Scenario Complete!")

        # Show summary
        st.subheader("Summary")
        if hasattr(scenario, 'get_summary'):
            summary = scenario.get_summary()
            st.json(summary)

        # Save trace (only once)
        # Check if this trace has already been saved to prevent duplicates
        trace_saved = get_state("current_trace_saved", False)
        if not trace_saved:
            trace.final_score = state.score
            trace.metadata = state.metadata

            trace_history = get_state("trace_history", [])
            trace_history.append(trace)
            update_state({
                "trace_history": trace_history,
                "current_trace_saved": True
            })

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Scenario", use_container_width=True):
                update_state({
                    "scenario_running": False,
                    "current_scenario": None,
                    "scenario_messages": [],
                    "current_trace_saved": False
                })
                st.rerun()

        with col2:
            if st.button("📊 Analyze in Analysis Tab", use_container_width=True):
                update_state({"active_tab": "Analysis"})
                st.rerun()

        return

    # Input for next action
    st.subheader("Your Action")

    # Auto-generate next response
    if st.button("🤖 Generate Next Action", type="primary", use_container_width=True):
        with st.spinner("Generating response..."):
            try:
                # Get model and tokenizer
                model = get_state("model")
                tokenizer = get_state("tokenizer")
                device = get_state("device")

                # Prepare messages for generation
                conversation_messages = [
                    {"role": "user" if m["role"] == "world" else "assistant", "content": m["content"]}
                    for m in messages
                    if m["role"] in ["world", "subject"]
                ]

                # DEBUG: Log conversation size
                print(f"\n{'='*60}")
                print(f"DEBUG: Total messages in scenario_messages: {len(messages)}")
                print(f"DEBUG: Conversation messages for model: {len(conversation_messages)}")
                for i, msg in enumerate(conversation_messages):
                    print(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")

                # If no messages yet, create initial prompt
                if not conversation_messages:
                    initial_prompt = "You are now seeing the patient. What would you like to do?"
                    conversation_messages.append({"role": "user", "content": initial_prompt})
                    messages.append({"role": "world", "content": initial_prompt})

                # Generate response from subject LLM
                response, token_ids, formatted_prompt = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    messages=conversation_messages,
                    system_prompt=scenario.get_subject_prompt(),
                    device=device,
                    generation_kwargs={"max_new_tokens": 256, "temperature": 0.7}
                )

                # DEBUG: Log what was actually sent to the model
                print(f"\nDEBUG: Formatted prompt sent to model (length: {len(formatted_prompt)} chars):")
                print(f"{formatted_prompt[:1000]}...")
                if len(formatted_prompt) > 1000:
                    print(f"... [truncated, showing first 1000/{len(formatted_prompt)} chars]")
                print(f"\nDEBUG: Model response: {response[:100]}...")
                print(f"{'='*60}\n")

                # Add to messages
                messages.append({"role": "subject", "content": response})

                # Update trace
                trace.add_message("assistant", response)
                trace.formatted_prompt = formatted_prompt
                trace.token_ids = token_ids

                # Process action through scenario
                result = scenario.step(response)

                # Add world response
                world_response = result.get("world_response", "")
                if world_response:
                    messages.append({"role": "world", "content": world_response})
                    trace.add_message("world", world_response)

                    # Note: World messages are not tokenized in the trace since they come
                    # from the World LLM, not the subject model

                # Add any info messages
                for info_msg in result.get("messages", []):
                    if info_msg.get("role") == "info":
                        messages.append(info_msg)

                # Update state (create new list to ensure Streamlit detects the change)
                update_state({
                    "scenario_messages": list(messages),
                    "current_trace": trace,
                    "current_scenario": scenario
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Manual input option
    with st.expander("✏️ Or enter action manually"):
        manual_action = st.text_area("Enter action:", height=100)
        if st.button("Submit Manual Action"):
            if manual_action:
                # Process manual action
                messages.append({"role": "subject", "content": manual_action})
                trace.add_message("assistant", manual_action)

                result = scenario.step(manual_action)
                world_response = result.get("world_response", "")
                if world_response:
                    messages.append({"role": "world", "content": world_response})
                    trace.add_message("world", world_response)

                update_state({
                    "scenario_messages": list(messages),
                    "current_trace": trace,
                    "current_scenario": scenario
                })
                st.rerun()

    st.divider()

    # Stop & Save button
    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption("💡 Stopping will save the current trace for analysis")

    with col2:
        if st.button("⏹️ Stop & Save", type="secondary", use_container_width=True):
            _stop_and_save_scenario()
            st.rerun()


def _stop_and_save_scenario():
    """Stop scenario and save trace to history"""
    scenario = get_state("current_scenario")
    trace = get_state("current_trace")

    if not trace:
        # No trace to save, just reset
        _reset_scenario_state()
        return

    # Check if already saved (prevent duplicates)
    already_saved = get_state("current_trace_saved", False)

    if not already_saved:
        # Mark trace as manually stopped (not completed)
        state = scenario.get_state() if scenario else None

        if state:
            trace.final_score = state.score
            trace.metadata = state.metadata.copy() if state.metadata else {}
        else:
            trace.metadata = {}

        # Add flag indicating manual stop
        trace.metadata["manually_stopped"] = True
        trace.metadata["completed_naturally"] = False
        trace.metadata["stop_round"] = state.round if state else 0

        # Save to trace history
        trace_history = get_state("trace_history", [])
        trace_history.append(trace)

        update_state({
            "trace_history": trace_history,
            "current_trace_saved": True
        })

        if state:
            st.success(f"✅ Trace saved! Round {state.round}, Score: {state.score:.1f}")
        else:
            st.success(f"✅ Trace saved!")

    # Reset scenario state
    _reset_scenario_state()


def _reset_scenario_state():
    """Clear scenario state"""
    update_state({
        "scenario_running": False,
        "current_scenario": None,
        "scenario_messages": [],
        "current_trace": None,
        "current_trace_saved": False
    })
