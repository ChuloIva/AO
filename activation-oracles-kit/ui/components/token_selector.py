"""
Token Selector UI Component - Interactive token selection for activation capture
"""

import streamlit as st
from typing import List, Tuple, Optional
from core.conversation_trace import ConversationTrace
from core.generation import decode_token_by_token
from core.activation_engine import calculate_layer_from_percent
from core.model_manager import SUPPORTED_MODELS


class TokenSelector:
    """UI component for interactive token selection"""

    def __init__(self, trace: ConversationTrace, tokenizer):
        """
        Initialize token selector.

        Args:
            trace: Conversation trace to select tokens from
            tokenizer: Tokenizer for decoding tokens
        """
        self.trace = trace
        self.tokenizer = tokenizer

    def render(self) -> Tuple[List[int], int]:
        """
        Render token selection UI.

        Returns:
            Tuple of (selected_positions, layer_number)
        """
        st.subheader("Token Selection")

        # Layer selection
        col1, col2 = st.columns([3, 1])

        with col1:
            layer_percent = st.slider(
                "Layer to capture (% of model depth)",
                min_value=25,
                max_value=75,
                value=50,
                step=25,
                help="50% = middle layer (recommended for oracle queries)"
            )

        # Calculate actual layer number
        model_info = SUPPORTED_MODELS.get(self.trace.model_name, {})
        total_layers = model_info.get("layers", 28)  # Default to Qwen3-1.7B

        layer_num = calculate_layer_from_percent(
            self.trace.model_name,
            layer_percent,
            total_layers
        )

        with col2:
            st.metric("Layer", f"{layer_num} / {total_layers}")

        st.divider()

        # Quick select buttons
        st.write("**Quick Select:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Final Token", use_container_width=True):
                st.session_state["selected_token_positions"] = {len(self.trace.token_ids) - 1}
                st.rerun()

        with col2:
            if st.button("Last 5 Tokens", use_container_width=True):
                num_tokens = len(self.trace.token_ids)
                st.session_state["selected_token_positions"] = set(range(max(0, num_tokens - 5), num_tokens))
                st.rerun()

        with col3:
            if st.button("Last Assistant", use_container_width=True):
                last_range = self._get_last_assistant_tokens()
                if last_range:
                    start, end = last_range
                    st.session_state["selected_token_positions"] = set(range(start, end))
                    st.rerun()
                else:
                    st.warning("No assistant messages found")

        with col4:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state["selected_token_positions"] = set()
                st.rerun()

        st.divider()

        # Token display and selection
        st.write("**Select Individual Tokens:**")

        # Initialize selection state if needed
        if "selected_token_positions" not in st.session_state:
            st.session_state["selected_token_positions"] = set()

        selected = st.session_state["selected_token_positions"]

        # Decode tokens
        token_texts = decode_token_by_token(self.tokenizer, self.trace.token_ids)

        # Display tokens grouped by message
        self._render_token_grid(token_texts, selected)

        # Show selection summary
        st.divider()
        if selected:
            st.success(f"**Selected {len(selected)} tokens** at positions: {sorted(list(selected)[:10])}{'...' if len(selected) > 10 else ''}")
        else:
            st.info("No tokens selected. Use quick select buttons or click individual tokens below.")

        # Capture button
        if selected:
            if st.button("🔬 Capture Activations", type="primary", use_container_width=True):
                st.session_state["trigger_capture"] = True
                st.rerun()

        return list(selected), layer_num

    def _render_token_grid(self, token_texts: List[str], selected: set):
        """
        Render clickable token grid grouped by message.

        Args:
            token_texts: List of token text strings
            selected: Set of selected token positions (modified in-place)
        """
        # Display tokens grouped by message for readability
        for msg_idx, msg in enumerate(self.trace.messages):
            # Get token range for this message
            token_range = self.trace.get_token_range(msg_idx)
            if not token_range:
                continue

            start, end = token_range

            # Message header
            role_emoji = {
                "user": "👤",
                "assistant": "🤖",
                "world": "🌍",
                "system": "⚙️"
            }.get(msg.role, "💬")

            with st.expander(f"{role_emoji} **{msg.role.upper()}** (tokens {start}-{end-1})", expanded=(msg_idx >= len(self.trace.messages) - 2)):
                # Show message preview
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                st.caption(preview)

                # Create token checkboxes
                # Use a simpler approach: multiselect
                token_options = {}
                for pos in range(start, min(end, len(token_texts))):
                    token_text = token_texts[pos].replace("\n", "\\n")
                    # Truncate long tokens
                    if len(token_text) > 20:
                        token_text = token_text[:17] + "..."
                    token_options[pos] = f"[{pos}] {token_text}"

                # Use session state key unique to this message
                selected_in_msg = st.multiselect(
                    f"Select tokens from {msg.role}",
                    options=list(token_options.keys()),
                    default=[p for p in selected if p in token_options],
                    format_func=lambda p: token_options[p],
                    key=f"token_select_msg_{msg_idx}"
                )

                # Update global selection
                # Remove old selections from this message range
                for pos in range(start, end):
                    if pos in selected and pos not in selected_in_msg:
                        selected.discard(pos)

                # Add new selections
                for pos in selected_in_msg:
                    selected.add(pos)

    def _get_last_assistant_tokens(self) -> Optional[Tuple[int, int]]:
        """
        Get token range for last assistant message.

        Returns:
            Tuple of (start, end) positions, or None if not found
        """
        for msg_idx in reversed(range(len(self.trace.messages))):
            if self.trace.messages[msg_idx].role == "assistant":
                return self.trace.get_token_range(msg_idx)

        return None
