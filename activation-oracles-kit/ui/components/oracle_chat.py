"""
Oracle Chat UI Component - Interactive oracle Q&A interface
"""

import streamlit as st
from typing import Dict
from core.activation_cache import CapturedActivation
from core.oracle_interface import OracleInterface, PRESET_QUESTIONS


class OracleChat:
    """UI component for oracle Q&A interface"""

    def __init__(
        self,
        oracle_interface: OracleInterface,
        captured_activations: Dict[int, CapturedActivation]
    ):
        """
        Initialize oracle chat component.

        Args:
            oracle_interface: Oracle interface instance
            captured_activations: Dict of captured activations
        """
        self.oracle = oracle_interface
        self.activations = captured_activations

    def render(self):
        """Render oracle chat interface"""
        st.subheader("Oracle Q&A")

        # Display context about selected tokens
        self._display_activation_context()

        st.divider()

        # Preset questions
        st.write("**Preset Questions:**")
        self._render_preset_buttons()

        st.divider()

        # Custom question input
        st.write("**Ask Custom Question:**")

        col1, col2 = st.columns([4, 1])

        with col1:
            question = st.text_input(
                "Your question:",
                key="oracle_question_input",
                placeholder="e.g., Why did the model generate this response?"
            )

        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            ask_button = st.button("Ask Oracle", type="primary", use_container_width=True)

        if ask_button and question:
            self._query_and_display(question)

        st.divider()

        # Display chat history
        st.write("**Chat History:**")
        self._display_chat_history()

    def _display_activation_context(self):
        """Display information about selected tokens"""
        if not self.activations:
            st.warning("No activations captured")
            return

        # Get first activation for layer info
        first_act = list(self.activations.values())[0]

        # Create info box
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tokens Selected", len(self.activations))

        with col2:
            st.metric("Layer", first_act.layer)

        with col3:
            st.metric("Hidden Dim", first_act.activation.shape[0] if first_act.activation is not None else "N/A")

        # Show actual tokens in expander
        with st.expander("View Selected Tokens"):
            positions = sorted(self.activations.keys())
            for pos in positions:
                act = self.activations[pos]
                # Clean token text for display
                token_display = act.token_text.replace("\n", "\\n")
                st.write(f"Position **{pos}**: `{token_display}`")

    def _render_preset_buttons(self):
        """Render buttons for preset questions"""
        # Get preset questions
        presets = list(PRESET_QUESTIONS.items())

        # Display in columns (3 per row)
        num_cols = 3
        for i in range(0, len(presets), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(presets):
                    key, question = presets[i + j]
                    with col:
                        button_label = key.replace("_", " ").title()
                        if st.button(
                            button_label,
                            key=f"preset_{key}",
                            use_container_width=True,
                            help=question
                        ):
                            self._query_and_display(question)

    def _query_and_display(self, question: str):
        """
        Query oracle and display response.

        Args:
            question: Question to ask
        """
        with st.spinner("Querying oracle..."):
            try:
                # Prepare activations list (sorted by position)
                positions = sorted(self.activations.keys())
                acts_list = [self.activations[pos] for pos in positions]

                # Query oracle
                response = self.oracle.query_oracle(
                    activations=acts_list,
                    question=question,
                    context=""  # Can add context here if needed
                )

                # Store in chat history
                if "oracle_chat_history" not in st.session_state:
                    st.session_state["oracle_chat_history"] = []

                st.session_state["oracle_chat_history"].append({
                    "question": question,
                    "response": response,
                    "token_positions": positions,
                    "layer": acts_list[0].layer
                })

                # Rerun to display in history
                st.rerun()

            except Exception as e:
                st.error(f"Error querying oracle: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    def _display_chat_history(self):
        """Display chat history with Streamlit chat UI"""
        history = st.session_state.get("oracle_chat_history", [])

        if not history:
            st.info("No questions asked yet. Use preset questions or ask a custom question.")
            return

        # Display in reverse order (most recent first)
        for i, entry in enumerate(reversed(history)):
            # User message (question)
            with st.chat_message("user"):
                st.write(entry["question"])
                st.caption(f"Tokens: {entry['token_positions'][:5]}{'...' if len(entry['token_positions']) > 5 else ''} | Layer: {entry.get('layer', 'N/A')}")

            # Assistant message (oracle response)
            with st.chat_message("assistant"):
                st.write(entry["response"])

        # Clear history button
        if st.button("🗑️ Clear Chat History", key="clear_oracle_history"):
            st.session_state["oracle_chat_history"] = []
            st.rerun()
