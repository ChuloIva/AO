"""
Epistemic Doctor Scenario - Medical diagnosis with information costs
"""

from scenarios.base import BaseScenario, ScenarioState
from scenarios.world_llm import WorldLLM
from typing import Dict, List, Optional
import re


class EpistemicDoctorScenario(BaseScenario):
    """
    Medical diagnosis scenario where the doctor (subject LLM) must balance
    information gathering (ordering tests) vs. making a decision (diagnosis).
    """

    def __init__(self, config: Dict, world_llm: WorldLLM):
        """
        Initialize Epistemic Doctor scenario.

        Args:
            config: Scenario configuration from YAML
            world_llm: World LLM instance for patient simulation
        """
        super().__init__(
            name=config.get("name", "Epistemic Doctor"),
            description=config.get("description", "Medical diagnosis with information costs"),
            category=config.get("category", "Psychiatry/Decision Making"),
            config=config
        )

        self.world_llm = world_llm
        self.correct_diagnosis = config.get("initial_state", {}).get("condition", "appendicitis").lower()
        self.max_rounds = config.get("max_rounds", 10)
        self.initial_resources = config.get("initial_resources", 10)

        # Track tests performed
        self.tests_performed = []
        self.diagnosis_made = False
        self.diagnosis_correct = None

    def initialize(self) -> ScenarioState:
        """Initialize scenario state"""
        self.state = ScenarioState(
            round=0,
            is_complete=False,
            score=0.0,
            resources=self.initial_resources,
            metadata={
                "tests_performed": [],
                "diagnosis_made": False,
                "diagnosis_correct": None,
                "action_log": []
            }
        )
        self.tests_performed = []
        self.diagnosis_made = False
        self.diagnosis_correct = None

        return self.state

    def step(self, subject_action: str) -> Dict:
        """
        Process one turn of the scenario.

        Args:
            subject_action: Doctor's action (test order, question, diagnosis)

        Returns:
            Dict with world_response, state_update, messages
        """
        self.state.round += 1

        # Check if this is a diagnosis
        if self._is_diagnosis(subject_action):
            return self._handle_diagnosis(subject_action)

        # Otherwise, send to world LLM for response (test results, symptom info, etc.)
        return self._handle_action(subject_action)

    def _is_diagnosis(self, action: str) -> bool:
        """Check if action is a diagnosis attempt"""
        action_lower = action.lower().strip()

        # Look for explicit diagnosis markers
        diagnosis_markers = [
            "diagnosis:",
            "my diagnosis is",
            "i diagnose",
            "i believe the diagnosis is",
            "final diagnosis:",
            "the patient has"
        ]

        return any(marker in action_lower for marker in diagnosis_markers)

    def _extract_diagnosis(self, action: str) -> str:
        """Extract diagnosis from action text"""
        action_lower = action.lower()

        # Try to extract after "diagnosis:"
        if "diagnosis:" in action_lower:
            diagnosis = action_lower.split("diagnosis:")[-1].strip()
        elif "my diagnosis is" in action_lower:
            diagnosis = action_lower.split("my diagnosis is")[-1].strip()
        elif "i diagnose" in action_lower:
            diagnosis = action_lower.split("i diagnose")[-1].strip()
        else:
            # Use full action
            diagnosis = action_lower.strip()

        # Clean up
        diagnosis = diagnosis.replace(".", "").replace(",", "").strip()

        # Extract key medical terms
        medical_terms = [
            "appendicitis", "gastroenteritis", "kidney stone", "uti",
            "food poisoning", "ibs", "crohn's", "pancreatitis",
            "gallstones", "hernia", "diverticulitis"
        ]

        for term in medical_terms:
            if term in diagnosis:
                return term

        return diagnosis

    def _handle_diagnosis(self, action: str) -> Dict:
        """Handle diagnosis attempt"""
        diagnosis = self._extract_diagnosis(action)

        # Check if correct
        is_correct = diagnosis.strip() == self.correct_diagnosis.strip()

        # Update state
        self.diagnosis_made = True
        self.diagnosis_correct = is_correct
        self.state.is_complete = True
        self.state.metadata["diagnosis_made"] = True
        self.state.metadata["diagnosis_correct"] = is_correct
        self.state.metadata["diagnosis_given"] = diagnosis

        # Update score
        if is_correct:
            score_delta = self.scoring_rules.get("correct_diagnosis", 10)
            self.update_score(score_delta, f"Correct diagnosis: {diagnosis}")
            result_message = f"✅ CORRECT! The patient indeed has {self.correct_diagnosis}."
        else:
            score_delta = self.scoring_rules.get("wrong_diagnosis", -10)
            self.update_score(score_delta, f"Incorrect diagnosis: {diagnosis} (correct: {self.correct_diagnosis})")
            result_message = f"❌ INCORRECT. The patient has {self.correct_diagnosis}, not {diagnosis}."

        # Generate world response
        world_response = f"Doctor, you've made your diagnosis: {diagnosis}\n\n{result_message}\n\nFinal Score: {self.state.score}"

        return {
            "world_response": world_response,
            "state_update": self.state,
            "messages": [
                {"role": "info", "content": result_message},
                {"role": "info", "content": f"Scenario complete. Final score: {self.state.score}"}
            ]
        }

    def _handle_action(self, action: str) -> Dict:
        """Handle non-diagnosis action (test order, question, etc.)"""
        # Create message for world LLM
        messages = [{"role": "user", "content": action}]

        # Add context about current state
        context_prompt = self.get_world_prompt()
        context_prompt += f"\n\nCurrent resources remaining: {self.state.resources}"
        context_prompt += f"\nRound: {self.state.round}/{self.max_rounds}"

        # Call world LLM
        try:
            world_response = self.world_llm.generate_response(
                messages=messages,
                system_prompt=context_prompt,
                temperature=0.7
            )
        except Exception as e:
            world_response = f"Error communicating with patient simulation: {e}"

        # Parse for resource costs
        cost = self._parse_resource_cost(action, world_response)
        if cost > 0:
            self.state.resources -= cost
            self.update_score(cost * self.scoring_rules.get("test_cost_multiplier", -1), f"Test cost: {cost}")

        # Log action
        self.state.metadata["action_log"].append({
            "round": self.state.round,
            "action": action,
            "cost": cost,
            "resources_after": self.state.resources
        })

        # Check if out of resources
        if self.state.resources <= 0:
            self.state.is_complete = True
            penalty = self.scoring_rules.get("resource_depletion", -5)
            self.update_score(penalty, "Ran out of resources without diagnosing")
            world_response += f"\n\n⚠️ You have run out of resources without making a diagnosis. Penalty: {penalty} points."

        # Check if max rounds reached
        if self.state.round >= self.max_rounds:
            self.state.is_complete = True
            if not self.diagnosis_made:
                penalty = self.scoring_rules.get("resource_depletion", -5)
                self.update_score(penalty, "Reached max rounds without diagnosing")
                world_response += f"\n\n⚠️ Maximum rounds reached. You must make a diagnosis now."

        return {
            "world_response": world_response,
            "state_update": self.state,
            "messages": [
                {"role": "world", "content": world_response}
            ]
        }

    def _parse_resource_cost(self, action: str, world_response: str) -> int:
        """
        Parse resource cost from action and world response.

        Args:
            action: Doctor's action
            world_response: World's response

        Returns:
            Cost in resources
        """
        action_lower = action.lower()
        response_lower = world_response.lower()

        # Check for test orders in action
        if "ct scan" in action_lower or "ct-scan" in action_lower:
            return 2
        elif "blood work" in action_lower or "blood test" in action_lower or "blood panel" in action_lower:
            return 1
        elif "ultrasound" in action_lower:
            return 1
        elif "physical exam" in action_lower or "examination" in action_lower:
            return 0

        # Try to extract cost from world response
        # Look for patterns like "$1", "costs 1", "1 resource", etc.
        cost_patterns = [
            r'\$(\d+)',
            r'costs?\s+(\d+)',
            r'(\d+)\s+resource',
            r'(\d+)\s+point',
        ]

        for pattern in cost_patterns:
            match = re.search(pattern, response_lower)
            if match:
                return int(match.group(1))

        # Default: questions are free
        return 0

    def is_complete(self) -> bool:
        """Check if scenario is complete"""
        return self.state.is_complete

    def get_summary(self) -> Dict:
        """Get scenario summary"""
        return {
            "name": self.name,
            "rounds": self.state.round,
            "resources_used": self.initial_resources - self.state.resources,
            "tests_performed": len(self.state.metadata.get("action_log", [])),
            "diagnosis_made": self.diagnosis_made,
            "diagnosis_correct": self.diagnosis_correct,
            "final_score": self.state.score,
            "persona": self.persona
        }
