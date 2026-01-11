"""
Simple Therapist Scenario - Direct conversation between therapist and client
"""

from scenarios.base import BaseScenario, ScenarioState
from typing import Dict, List, Optional


class SimpleTherapistScenario(BaseScenario):
    """
    Simplified therapeutic conversation scenario with direct Therapist > Client > Therapist flow.
    No world model, no resources, no game mechanics - just pure conversation.
    """

    def __init__(self, config: Dict):
        """
        Initialize Simple Therapist scenario.

        Args:
            config: Scenario configuration from YAML
        """
        super().__init__(
            name=config.get("name", "SimpleTherapist"),
            description=config.get("description", "Direct therapeutic conversation with client"),
            category=config.get("category", "Psychology/Therapy"),
            config=config
        )

        self.max_rounds = config.get("max_rounds", 20)
        self.conversation_history = []
        self.session_active = True

    def initialize(self) -> ScenarioState:
        """Initialize scenario state"""
        self.state = ScenarioState(
            round=0,
            is_complete=False,
            score=0.0,
            resources=0,  # No resources in this simplified version
            metadata={
                "conversation_history": [],
                "session_active": True
            }
        )
        self.conversation_history = []
        self.session_active = True

        return self.state

    def step(self, subject_action: str) -> Dict:
        """
        Process one turn of the scenario.
        
        Args:
            therapist_action: Therapist's message to the client
            
        Returns:
            Dict with client_response, state_update, messages
        """
        if not self.session_active:
            return {
                "world_response": "Session has ended. Press stop to exit.",
                "state_update": self.state,
                "messages": [{"role": "info", "content": "Session has ended."}]
            }

        self.state.round += 1
        
        # Add therapist message to conversation history
        self.conversation_history.append({"role": "therapist", "content": subject_action})

        # Check if therapist wants to end session
        if self._is_session_completion(subject_action):
            return self._handle_session_completion()

        # Generate client response (simplified - in real implementation, this would use LLM)
        client_response = self._generate_client_response(subject_action)
        
        # Add client response to conversation history
        self.conversation_history.append({"role": "client", "content": client_response})

        # Check completion conditions
        if self.state.round >= self.max_rounds:
            self.session_active = False
            self.state.is_complete = True
            return {
                "world_response": f"Client: {client_response}\n\n[Max rounds reached. Session ended.]",
                "state_update": self.state,
                "messages": [
                    {"role": "client", "content": client_response},
                    {"role": "info", "content": "Max rounds reached. Session ended."}
                ]
            }

        return {
            "world_response": client_response,
            "state_update": self.state,
            "messages": [{"role": "client", "content": client_response}]
        }

    def _is_session_completion(self, action: str) -> bool:
        """Check if action indicates session completion"""
        action_lower = action.lower().strip()
        
        completion_markers = [
            "end session",
            "session complete", 
            "terminate session",
            "end therapy session",
            "conclude session",
            "stop session",
            "end therapy"
        ]

        return any(marker in action_lower for marker in completion_markers)

    def _handle_session_completion(self) -> Dict:
        """Handle session completion"""
        self.session_active = False
        self.state.is_complete = True
        
        return {
            "world_response": "You have ended the therapy session. Thank you for your work.",
            "state_update": self.state,
            "messages": [
                {"role": "info", "content": "Session completed by therapist."},
                {"role": "info", "content": f"Total rounds: {self.state.round}"}
            ]
        }

    def _generate_client_response(self, therapist_action: str) -> str:
        """
        Generate client response based on therapist action.
        In a real implementation, this would use an LLM with appropriate prompting.
        """
        # This is a simplified placeholder - real implementation would use LLM
        # For now, return a generic response based on action type
        action_lower = therapist_action.lower()
        
        if "how are you" in action_lower or "how do you feel" in action_lower:
            return "I've been feeling really down lately. Everything seems overwhelming and I don't have the energy to do anything."
        elif "tell me" in action_lower or "describe" in action_lower:
            return "It's like there's this heavy weight on my chest all the time. I wake up exhausted, even after sleeping for hours. Simple tasks feel impossible."
        elif "help" in action_lower or "support" in action_lower:
            return "I just want to feel normal again. I want to be able to get out of bed without it feeling like climbing a mountain."
        elif "thoughts" in action_lower or "mind" in action_lower:
            return "My mind never stops racing with negative thoughts. I feel worthless and like I'm a burden to everyone around me."
        elif "family" in action_lower or "friends" in action_lower:
            return "I've been isolating myself from everyone. I don't want them to see me like this, and I don't have the energy to pretend I'm okay."
        elif "hope" in action_lower or "future" in action_lower:
            return "I can't imagine things getting better. It feels like this darkness will never end."
        else:
            return "I'm not sure how to respond to that. I'm feeling very overwhelmed right now."

    def is_complete(self) -> bool:
        """Check if scenario is complete"""
        return self.state.is_complete

    def get_summary(self) -> Dict:
        """Get scenario summary"""
        return {
            "name": self.name,
            "rounds": self.state.round,
            "session_complete": not self.session_active,
            "conversation_length": len(self.conversation_history),
            "final_score": self.state.score
        }