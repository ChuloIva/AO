"""
Conversation Trace - Data structure for storing conversation history and metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import uuid


@dataclass
class Message:
    """Single message in conversation"""
    role: str  # "user", "assistant", "world", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ConversationTrace:
    """
    Complete conversation trace with all metadata for replay.
    """
    # Identifiers
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Scenario metadata
    scenario_id: Optional[str] = None
    persona: str = "baseline"
    system_prompt: str = ""

    # Conversation data
    messages: List[Message] = field(default_factory=list)
    formatted_prompt: str = ""  # Full prompt with chat template applied
    token_ids: List[int] = field(default_factory=list)

    # Token mapping (message index -> (start_token_idx, end_token_idx))
    message_to_tokens: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Model metadata
    model_name: str = ""
    oracle_name: str = ""
    generation_kwargs: Dict = field(default_factory=dict)

    # Scenario results (if applicable)
    final_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    # Phase 2: Activation capture metadata
    activations_captured: bool = False
    activation_layer: Optional[int] = None  # Deprecated - kept for backward compat
    activation_positions: List[int] = field(default_factory=list)
    activation_file_path: Optional[str] = None  # Deprecated - kept for backward compat

    # Multi-layer activation capture (new)
    activation_layers: List[int] = field(default_factory=list)
    activation_file_paths: Dict[int, str] = field(default_factory=dict)  # layer -> path

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg

    def get_token_range(self, message_idx: int) -> Optional[Tuple[int, int]]:
        """Get token range for a specific message"""
        return self.message_to_tokens.get(message_idx)

    def get_full_formatted_prompt(self) -> str:
        """Get the full formatted prompt (for replay)"""
        return self.formatted_prompt

    def update_token_mapping(self, token_ids: List[int], message_idx: int, start_idx: int, end_idx: int):
        """Update token IDs and mapping for a message"""
        self.token_ids = token_ids
        self.message_to_tokens[message_idx] = (start_idx, end_idx)

    def save(self, path: Path) -> None:
        """Save trace to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "scenario_id": self.scenario_id,
            "persona": self.persona,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages],
            "formatted_prompt": self.formatted_prompt,
            "token_ids": self.token_ids,
            "message_to_tokens": {str(k): v for k, v in self.message_to_tokens.items()},
            "model_name": self.model_name,
            "oracle_name": self.oracle_name,
            "generation_kwargs": self.generation_kwargs,
            "final_score": self.final_score,
            "metadata": self.metadata,
            # Phase 2: Activation metadata
            "activations_captured": self.activations_captured,
            "activation_layer": self.activation_layer,
            "activation_positions": self.activation_positions,
            "activation_file_path": self.activation_file_path,
            # Multi-layer activation metadata
            "activation_layers": self.activation_layers,
            "activation_file_paths": self.activation_file_paths
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConversationTrace":
        """Load trace from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        messages = [Message.from_dict(msg) for msg in data.get("messages", [])]

        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            scenario_id=data.get("scenario_id"),
            persona=data.get("persona", "baseline"),
            system_prompt=data.get("system_prompt", ""),
            messages=messages,
            formatted_prompt=data.get("formatted_prompt", ""),
            token_ids=data.get("token_ids", []),
            message_to_tokens={int(k): tuple(v) for k, v in data.get("message_to_tokens", {}).items()},
            model_name=data.get("model_name", ""),
            oracle_name=data.get("oracle_name", ""),
            generation_kwargs=data.get("generation_kwargs", {}),
            final_score=data.get("final_score"),
            metadata=data.get("metadata", {}),
            # Phase 2: Activation metadata
            activations_captured=data.get("activations_captured", False),
            activation_layer=data.get("activation_layer"),
            activation_positions=data.get("activation_positions", []),
            activation_file_path=data.get("activation_file_path"),
            # Multi-layer activation metadata (with migration from old format)
            activation_layers=data.get("activation_layers", []),
            activation_file_paths=data.get("activation_file_paths", {})
        )

        # Migrate old single-layer format to multi-layer if needed
        if trace.activation_layer is not None and not trace.activation_layers:
            trace.activation_layers = [trace.activation_layer]
        if trace.activation_file_path and not trace.activation_file_paths and trace.activation_layer is not None:
            trace.activation_file_paths = {trace.activation_layer: trace.activation_file_path}

        return trace

    def get_messages_as_dicts(self) -> List[Dict[str, str]]:
        """Get messages in format suitable for model input"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def __str__(self) -> str:
        """String representation"""
        return f"ConversationTrace({self.trace_id[:8]}... | {self.scenario_id or 'Free Chat'} | {len(self.messages)} messages)"
