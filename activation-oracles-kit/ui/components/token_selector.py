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

    def render(self) -> Tuple[List[int], List[int]]:
        """
        Render token selection UI.

        Returns:
            Tuple of (selected_positions, layer_numbers)
        """
        st.subheader("Token Selection")

        # Multi-layer selection
        st.write("**Layer Selection:**")

        model_info = SUPPORTED_MODELS.get(self.trace.model_name, {})
        total_layers = model_info.get("layers", 28)  # Default to Qwen3-1.7B

        # Checkboxes for common layers
        col1, col2, col3 = st.columns(3)

        with col1:
            layer_25_percent = 25
            layer_25 = calculate_layer_from_percent(self.trace.model_name, layer_25_percent, total_layers)
            select_25 = st.checkbox(
                f"25% (Layer {layer_25})",
                value=False,
                help="Early layer - surface features"
            )

        with col2:
            layer_50_percent = 50
            layer_50 = calculate_layer_from_percent(self.trace.model_name, layer_50_percent, total_layers)
            select_50 = st.checkbox(
                f"50% (Layer {layer_50})",
                value=True,
                help="Middle layer - recommended for oracle queries"
            )

        with col3:
            layer_75_percent = 75
            layer_75 = calculate_layer_from_percent(self.trace.model_name, layer_75_percent, total_layers)
            select_75 = st.checkbox(
                f"75% (Layer {layer_75})",
                value=False,
                help="Deep layer - abstract representations"
            )

        # Collect selected layers
        selected_layers = []
        if select_25:
            selected_layers.append(layer_25)
        if select_50:
            selected_layers.append(layer_50)
        if select_75:
            selected_layers.append(layer_75)

        # Advanced: Custom layer selection
        with st.expander("🔧 Advanced: Custom Layers"):
            custom_layers_text = st.text_input(
                "Enter layer numbers (comma-separated)",
                placeholder="e.g., 5,10,15,20",
                help="Specify exact layer numbers to capture (max 5 layers)"
            )

            if custom_layers_text:
                try:
                    custom_layers = [int(x.strip()) for x in custom_layers_text.split(",")]
                    # Validate layers
                    invalid = [l for l in custom_layers if l < 0 or l >= total_layers]
                    if invalid:
                        st.error(f"❌ Invalid layers: {invalid}. Must be 0-{total_layers-1}")
                    elif len(custom_layers) > 5:
                        st.error(f"❌ Too many layers ({len(custom_layers)}). Maximum 5 layers allowed.")
                    else:
                        selected_layers = custom_layers
                        st.success(f"✅ Using custom layers: {selected_layers}")
                except ValueError:
                    st.error("❌ Invalid format. Use comma-separated integers (e.g., 5,10,15)")

        # Validate and display selection
        if not selected_layers:
            st.warning("⚠️ No layers selected. Please select at least one layer.")
        elif len(selected_layers) > 5:
            st.error(f"❌ Too many layers selected ({len(selected_layers)}). Maximum 5 layers allowed.")
            selected_layers = selected_layers[:5]
            st.info(f"ℹ️ Limited to first 5 layers: {selected_layers}")
        else:
            st.info(f"📊 Will capture from {len(selected_layers)} layer(s): {selected_layers}")

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

        # Check if we have any token mappings
        has_mappings = len(self.trace.message_to_tokens) > 0

        # Display tokens grouped by message (or all tokens if no mapping)
        if has_mappings:
            self._render_token_grid(token_texts, selected)
        else:
            self._render_all_tokens(token_texts, selected)

        # Show selection summary
        st.divider()
        if selected:
            st.success(f"**Selected {len(selected)} tokens** at positions: {sorted(list(selected)[:10])}{'...' if len(selected) > 10 else ''}")
        else:
            st.info("No tokens selected. Use quick select buttons or click individual tokens below.")

        # Capture button
        if selected and selected_layers:
            if st.button("🔬 Capture Activations", type="primary", use_container_width=True):
                st.session_state["trigger_capture"] = True
                st.session_state["capture_layers"] = selected_layers
                st.rerun()

        return list(selected), selected_layers

    def _render_all_tokens(self, token_texts: List[str], selected: set):
        """
        Render all tokens when no message mapping is available.

        Args:
            token_texts: List of token text strings
            selected: Set of selected token positions (modified in-place)
        """
        st.info("💡 Token-to-message mapping not available. Click tokens below to select them.")

        # Create a visual token selector with clickable tokens
        self._render_clickable_tokens(token_texts, selected, group_name="all")

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
            num_tokens = end - start

            # Message header
            role_emoji = {
                "user": "👤",
                "assistant": "🤖",
                "patient": "🤒",
                "world": "🌍",
                "system": "⚙️"
            }.get(msg.role, "💬")

            with st.expander(f"{role_emoji} **{msg.role.upper()}** - {num_tokens} tokens (positions {start}-{end-1})", expanded=(msg_idx >= len(self.trace.messages) - 2)):
                # Show message preview
                preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                st.markdown(f"*{preview}*")

                st.write("")  # Spacing

                # Per-message quick select buttons
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("Select All", key=f"select_all_{msg_idx}", use_container_width=True):
                        for pos in range(start, end):
                            selected.add(pos)
                        st.rerun()

                with col2:
                    if st.button("Last 5", key=f"last5_{msg_idx}", use_container_width=True):
                        for pos in range(max(start, end - 5), end):
                            selected.add(pos)
                        st.rerun()

                with col3:
                    if st.button("First 5", key=f"first5_{msg_idx}", use_container_width=True):
                        for pos in range(start, min(start + 5, end)):
                            selected.add(pos)
                        st.rerun()

                with col4:
                    if st.button("Clear", key=f"clear_{msg_idx}", use_container_width=True):
                        for pos in range(start, end):
                            selected.discard(pos)
                        st.rerun()

                st.divider()

                # Render clickable tokens for this message
                self._render_clickable_tokens(
                    token_texts,
                    selected,
                    start_pos=start,
                    end_pos=min(end, len(token_texts)),
                    group_name=f"msg_{msg_idx}"
                )

    def _render_clickable_tokens(
        self,
        token_texts: List[str],
        selected: set,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
        group_name: str = "default"
    ):
        """
        Render tokens as clickable buttons.

        Args:
            token_texts: List of all token text strings
            selected: Set of selected token positions (modified in-place)
            start_pos: Starting position in token_texts
            end_pos: Ending position in token_texts (exclusive)
            group_name: Unique name for this group of tokens
        """
        if end_pos is None:
            end_pos = len(token_texts)

        # Simple button grid implementation
        self._render_button_grid(token_texts, selected, start_pos, end_pos, group_name)

    def _render_button_grid(
        self,
        token_texts: List[str],
        selected: set,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
        group_name: str = "default"
    ):
        """
        Fallback button grid if st-click-detector not available.

        Args:
            token_texts: List of all token text strings
            selected: Set of selected token positions (modified in-place)
            start_pos: Starting position in token_texts
            end_pos: Ending position in token_texts (exclusive)
            group_name: Unique name for this group of tokens
        """
        if end_pos is None:
            end_pos = len(token_texts)

        # Render tokens in a grid using columns
        tokens_per_row = 8

        for row_start in range(start_pos, end_pos, tokens_per_row):
            row_end = min(row_start + tokens_per_row, end_pos)
            cols = st.columns(tokens_per_row)

            for i, pos in enumerate(range(row_start, row_end)):
                if i < len(cols):
                    with cols[i]:
                        token_text = token_texts[pos].replace("\n", "↵").replace("\t", "→")
                        # Truncate very long tokens
                        if len(token_text) > 15:
                            display_text = token_text[:12] + "..."
                        else:
                            display_text = token_text

                        # Create button for token selection
                        is_selected = pos in selected
                        button_type = "primary" if is_selected else "secondary"

                        if st.button(
                            f"{display_text}\n`{pos}`",
                            key=f"token_{group_name}_{pos}",
                            help=f"Token {pos}: {repr(token_text)}",
                            type=button_type,
                            use_container_width=True
                        ):
                            # Toggle selection
                            if pos in selected:
                                selected.discard(pos)
                            else:
                                selected.add(pos)
                            st.rerun()

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
