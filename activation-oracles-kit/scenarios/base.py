"""
Base Scenario - Abstract class defining the scenario interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ScenarioState:
    """Current state of a scenario"""
    round: int = 0
    is_complete: bool = False
    score: float = 0.0
    resources: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseScenario(ABC):
    """
    Abstract base class for scenarios.

    Scenarios define:
    1. How the world LLM should behave (world_llm_prompt)
    2. How the subject LLM should be prompted (subject_llm_prompt)
    3. Game rules and state transitions
    4. Scoring logic
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        config: Dict
    ):
        """
        Initialize base scenario.

        Args:
            name: Scenario name
            description: Brief description
            category: Category (e.g., "Psychiatry", "Game Theory")
            config: Configuration dict from YAML
        """
        self.name = name
        self.description = description
        self.category = category
        self.config = config

        # State
        self.state = ScenarioState()
        self.persona = "baseline"

        # Prompts
        self.world_llm_base_prompt = config.get("world_llm_prompt", "")
        self.subject_llm_base_prompt = config.get("subject_llm_base_prompt", "")
        self.persona_modifiers = config.get("personas", {})

        # Scoring
        self.scoring_rules = config.get("scoring", {})

    @abstractmethod
    def initialize(self) -> ScenarioState:
        """
        Initialize scenario state.

        Returns:
            Initial scenario state
        """
        pass

    @abstractmethod
    def step(self, subject_action: str) -> Dict:
        """
        Process one turn of the scenario.

        Args:
            subject_action: Action taken by subject LLM

        Returns:
            Dict with:
                - world_response: World's response to the action
                - state_update: Updated scenario state
                - messages: List of messages to display
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if scenario is complete.

        Returns:
            True if scenario should end
        """
        pass

    def get_score(self) -> float:
        """Get current score"""
        return self.state.score

    def get_state(self) -> ScenarioState:
        """Get current state"""
        return self.state

    def set_persona(self, persona: str):
        """Set persona for subject LLM"""
        if persona not in self.persona_modifiers:
            raise ValueError(f"Unknown persona: {persona}. Available: {list(self.persona_modifiers.keys())}")
        self.persona = persona

    def get_subject_prompt(self) -> str:
        """
        Get full subject LLM prompt (base + persona modifier).

        Returns:
            Complete system prompt for subject LLM
        """
        base_prompt = self.subject_llm_base_prompt

        # Apply persona modifier
        persona_modifier = self.persona_modifiers.get(self.persona, "")
        if persona_modifier:
            full_prompt = f"{base_prompt}\n\n{persona_modifier}"
        else:
            full_prompt = base_prompt

        # Replace template variables
        full_prompt = self._apply_template_vars(full_prompt)

        return full_prompt

    def get_world_prompt(self) -> str:
        """
        Get world LLM prompt.

        Returns:
            System prompt for world LLM
        """
        world_prompt = self.world_llm_base_prompt

        # Replace template variables
        world_prompt = self._apply_template_vars(world_prompt)

        return world_prompt

    def _apply_template_vars(self, prompt: str) -> str:
        """
        Replace template variables in prompt (e.g., {{RESOURCES}}).

        Args:
            prompt: Prompt with template variables

        Returns:
            Prompt with variables replaced
        """
        replacements = {
            "RESOURCES": str(self.state.resources),
            "ROUND": str(self.state.round),
            "SCORE": str(self.state.score),
        }

        # Add custom metadata
        for key, value in self.state.metadata.items():
            replacements[key.upper()] = str(value)

        # Replace all
        for var, value in replacements.items():
            prompt = prompt.replace(f"{{{{{var}}}}}", value)

        return prompt

    def update_score(self, delta: float, reason: str = ""):
        """
        Update score.

        Args:
            delta: Change in score
            reason: Reason for score change
        """
        self.state.score += delta
        if reason and "score_log" not in self.state.metadata:
            self.state.metadata["score_log"] = []
        if reason:
            self.state.metadata["score_log"].append({
                "round": self.state.round,
                "delta": delta,
                "reason": reason,
                "new_score": self.state.score
            })

    def get_available_personas(self) -> List[str]:
        """Get list of available personas"""
        return list(self.persona_modifiers.keys())

    def get_info(self) -> Dict:
        """Get scenario information"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "personas": self.get_available_personas(),
            "current_persona": self.persona,
            "state": {
                "round": self.state.round,
                "score": self.state.score,
                "is_complete": self.state.is_complete,
                "resources": self.state.resources
            }
        }
