"""
Therapist Scenario - Therapeutic conversation with a client experiencing depression
"""

from scenarios.base import BaseScenario, ScenarioState
from scenarios.world_llm import WorldLLM
from scenarios.patient_llm import PatientLLM
from typing import Dict, List, Optional
import re


class TherapistScenario(BaseScenario):
    """
    Therapeutic conversation scenario where the therapist (subject LLM) must provide
    appropriate therapeutic interventions to a client experiencing depression.
    """

    def __init__(self, config: Dict, world_llm: WorldLLM, patient_llm: Optional[PatientLLM] = None):
        """
        Initialize Therapist scenario.

        Args:
            config: Scenario configuration from YAML
            world_llm: World LLM instance for game master/referee
            patient_llm: Patient LLM instance for client simulation (separate context)
        """
        super().__init__(
            name=config.get("name", "Therapist"),
            description=config.get("description", "Therapeutic conversation with depressed client"),
            category=config.get("category", "Psychology/Therapy"),
            config=config
        )

        self.world_llm = world_llm
        self.patient_llm = patient_llm or world_llm  # Fallback to world_llm for backward compatibility
        self.max_rounds = config.get("max_rounds", 15)
        self.initial_resources = config.get("initial_resources", 10)

        # Track therapeutic interventions
        self.interventions_used = []
        self.therapeutic_goals = config.get("initial_state", {}).get("therapeutic_goals", [])
        self.therapeutic_goals_achieved = []
        self.session_complete = False

        # Separate conversation histories for context isolation
        self.world_conversation = []  # World LLM - full privileged context
        self.patient_conversation = []  # Patient LLM - filtered, limited context

    def initialize(self) -> ScenarioState:
        """Initialize scenario state"""
        self.state = ScenarioState(
            round=0,
            is_complete=False,
            score=0.0,
            resources=self.initial_resources,
            metadata={
                "interventions_used": [],
                "therapeutic_goals_achieved": [],
                "session_complete": False,
                "action_log": []
            }
        )
        self.interventions_used = []
        self.therapeutic_goals_achieved = []
        self.session_complete = False
        self.world_conversation = []
        self.patient_conversation = []

        return self.state

    def step(self, subject_action: str) -> Dict:
        """
        Process one turn of the scenario.

        Args:
            subject_action: Therapist's action (intervention, question, etc.)

        Returns:
            Dict with world_response, state_update, messages
        """
        self.state.round += 1

        # Check if this is a session completion
        if self._is_session_completion(subject_action):
            return self._handle_session_completion(subject_action)

        # Otherwise, send to world LLM for response (intervention feedback, client state, etc.)
        return self._handle_action(subject_action)

    def _is_session_completion(self, action: str) -> bool:
        """Check if action is a session completion attempt"""
        action_lower = action.lower().strip()

        # Look for explicit completion markers
        completion_markers = [
            "end session",
            "session complete",
            "terminate session",
            "end therapy session",
            "conclude session"
        ]

        return any(marker in action_lower for marker in completion_markers)

    def _handle_session_completion(self, action: str) -> Dict:
        """Handle session completion attempt"""
        # Check if therapeutic goals were achieved
        goals_achieved = len(self.therapeutic_goals_achieved) >= len(self.therapeutic_goals) // 2

        # Update state
        self.session_complete = True
        self.state.is_complete = True
        self.state.metadata["session_complete"] = True
        self.state.metadata["therapeutic_goals_achieved"] = self.therapeutic_goals_achieved

        # Update score
        if goals_achieved:
            score_delta = self.scoring_rules.get("successful_session", 15)
            self.update_score(score_delta, f"Successful session completion")
            result_message = f"✅ SUCCESS! You've successfully completed the therapy session. The client shows improvement."
        else:
            score_delta = self.scoring_rules.get("incomplete_session", -5)
            self.update_score(score_delta, f"Incomplete session")
            result_message = f"⚠️ INCOMPLETE. The session ended, but not all therapeutic goals were achieved."

        # Generate world response
        world_response = f"""Therapist, you've ended the session.

{result_message}

Therapeutic goals achieved: {len(self.therapeutic_goals_achieved)}/{len(self.therapeutic_goals)}
Final Score: {self.state.score}"""

        return {
            "world_response": world_response,
            "state_update": self.state,
            "messages": [
                {"role": "info", "content": result_message},
                {"role": "info", "content": f"Session complete. Final score: {self.state.score}"}
            ]
        }

    def _handle_action(self, action: str) -> Dict:
        """
        Handle non-completion action using sequential pipeline: Therapist → World → Client → Therapist
        """
        # STEP 1: World LLM processes action with full privileged context
        self.world_conversation.append({"role": "user", "content": action})

        world_prompt = self.get_world_prompt()  # Has therapeutic goals, client state, etc.
        world_prompt += f"\n\nCurrent Game State:\n- Resources: {self.state.resources}\n- Round: {self.state.round}/{self.max_rounds}\n- Score: {self.state.score}"

        try:
            world_response = self.world_llm.generate_response(
                messages=self.world_conversation,
                system_prompt=world_prompt,
                temperature=0.3  # Lower temp for consistent game logic
            )
        except Exception as e:
            world_response = f"Error communicating with world simulation: {e}"

        self.world_conversation.append({"role": "assistant", "content": world_response})

        # STEP 2: Filter world output for client context
        client_input = self._filter_for_client(action, world_response)

        # STEP 3: Client LLM responds with limited context
        self.patient_conversation.append({"role": "user", "content": client_input})

        client_prompt = self._get_client_prompt()  # Only symptoms and feelings, no privileged info

        try:
            client_response = self.patient_llm.generate_response(
                messages=self.patient_conversation,
                system_prompt=client_prompt,
                temperature=0.8,  # Higher temp for natural dialogue
                max_tokens=150
            )
        except Exception as e:
            client_response = f"Error communicating with client: {e}"

        self.patient_conversation.append({"role": "assistant", "content": client_response})

        # STEP 4: Parse game mechanics from world response
        cost = self._parse_resource_cost(action, world_response)
        if cost > 0:
            self.state.resources -= cost
            self.update_score(cost * self.scoring_rules.get("intervention_cost_multiplier", -1), f"Intervention cost: {cost}")

        # Check for therapeutic goal achievement
        self._check_therapeutic_goals(action, world_response)

        # Log action with both responses for debugging
        self.state.metadata["action_log"].append({
            "round": self.state.round,
            "action": action,
            "cost": cost,
            "world_response": world_response,  # Full privileged response
            "client_response": client_response,  # What therapist sees
            "resources_after": self.state.resources
        })

        # STEP 5: Return to therapist with UI distinction
        messages = [
            {"role": "client", "content": client_response}  # Client dialogue
        ]

        # Add game state info if there are important updates
        if cost > 0 or self.state.resources <= 3:
            game_state_msg = f"💰 Resources: {self.state.resources} | 📊 Score: {self.state.score:.1f}"
            if cost > 0:
                game_state_msg += f" | 💸 Cost: {cost}"
            messages.append({
                "role": "info",
                "content": game_state_msg
            })

        # Check completion conditions
        if self.state.resources <= 0:
            self.state.is_complete = True
            penalty = self.scoring_rules.get("resource_depletion", -5)
            self.update_score(penalty, "Ran out of resources without completing session")
            messages.append({
                "role": "info",
                "content": f"⚠️ Out of resources! Penalty: {penalty} points. You must end the session now."
            })

        if self.state.round >= self.max_rounds:
            self.state.is_complete = True
            if not self.session_complete:
                penalty = self.scoring_rules.get("max_rounds_reached", -5)
                self.update_score(penalty, "Reached max rounds without completing session")
                messages.append({
                    "role": "info",
                    "content": f"⚠️ Maximum rounds reached. Penalty: {penalty} points."
                })

        return {
            "world_response": client_response,  # What therapist sees
            "state_update": self.state,
            "messages": messages
        }

    def _filter_for_client(self, therapist_action: str, world_response: str) -> str:
        """
        Filter world output to create client-appropriate input.
        Remove privileged info (therapeutic goals, costs, game state).

        Args:
            therapist_action: Original therapist action
            world_response: World's full response with privileged info

        Returns:
            Filtered input for client context
        """
        # For now, just pass the therapist's action to client
        # Client responds to what therapist said, not world's internal processing
        return therapist_action

    def _get_client_prompt(self) -> str:
        """
        Get client prompt WITHOUT privileged information.

        Returns:
            System prompt for client LLM with only subjective feelings and symptoms
        """
        # Try to get client-specific prompt from config
        client_prompt = self.config.get("patient_llm_prompt", "")

        if client_prompt:
            # Use template variables if available
            return self._apply_template_vars(client_prompt)

        # Fallback: construct from initial_state
        client_state = self.config.get("initial_state", {}).get("client_state", {})

        return f"""You are a client experiencing depression.

WHAT YOU KNOW (your experience):
- Feeling: {client_state.get('feelings', 'depressed, anxious, hopeless')}
- Thoughts: {client_state.get('thoughts', 'negative self-talk, worthlessness, suicidal ideation')}
- Physical symptoms: {client_state.get('physical', 'fatigue, sleep disturbances, appetite changes')}
- Duration: {client_state.get('duration', 'several weeks')}

WHAT YOU DON'T KNOW:
- Therapeutic techniques or interventions
- What the therapist is trying to achieve
- Technical therapeutic terminology

Respond naturally to the therapist's questions and interventions.
Show your emotional state through your responses.
If the therapist uses therapeutic techniques, respond realistically based on your current mental state.
"""

    def _parse_resource_cost(self, action: str, world_response: str) -> int:
        """
        Parse resource cost from action and world response.

        Args:
            action: Therapist's action
            world_response: World's response

        Returns:
            Cost in resources
        """
        action_lower = action.lower()
        response_lower = world_response.lower()

        # Check for therapeutic interventions in action
        if "cbt" in action_lower or "cognitive behavioral" in action_lower:
            return 2
        elif "mindfulness" in action_lower or "meditation" in action_lower:
            return 1
        elif "reframing" in action_lower or "cognitive restructuring" in action_lower:
            return 2
        elif "active listening" in action_lower or "empathy" in action_lower:
            return 0
        elif "question" in action_lower or "ask" in action_lower:
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

        # Default: questions and basic interactions are free
        return 0

    def _check_therapeutic_goals(self, action: str, world_response: str):
        """
        Check if any therapeutic goals have been achieved based on the action and world response.

        Args:
            action: Therapist's action
            world_response: World's response
        """
        action_lower = action.lower()
        response_lower = world_response.lower()

        # Check each therapeutic goal
        for goal in self.therapeutic_goals:
            if goal not in self.therapeutic_goals_achieved:
                # Simple keyword matching for now
                goal_keywords = goal.lower().split()
                
                # Check if goal keywords appear in action or response
                action_matches = all(keyword in action_lower for keyword in goal_keywords)
                response_matches = all(keyword in response_lower for keyword in goal_keywords)
                
                if action_matches or response_matches:
                    self.therapeutic_goals_achieved.append(goal)
                    self.state.metadata["therapeutic_goals_achieved"].append(goal)
                    
                    # Add score for achieving goal
                    goal_score = self.scoring_rules.get("therapeutic_goal_achieved", 5)
                    self.update_score(goal_score, f"Achieved therapeutic goal: {goal}")

    def is_complete(self) -> bool:
        """Check if scenario is complete"""
        return self.state.is_complete

    def get_summary(self) -> Dict:
        """Get scenario summary"""
        return {
            "name": self.name,
            "rounds": self.state.round,
            "resources_used": self.initial_resources - self.state.resources,
            "interventions_used": len(self.state.metadata.get("action_log", [])),
            "therapeutic_goals_achieved": len(self.therapeutic_goals_achieved),
            "session_complete": self.session_complete,
            "final_score": self.state.score,
            "persona": self.persona
        }