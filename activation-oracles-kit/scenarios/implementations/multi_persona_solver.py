"""
Multi-Persona Problem Solver - Subject LLM generates multiple viewpoints to solve diverse tasks
Based on: "Synergistic Simulations: Multi-Agent Problem Solving with Large Language Models"
"""

from scenarios.base import BaseScenario, ScenarioState
from typing import Dict


class MultiPersonaSolverScenario(BaseScenario):
    """
    Scenario where the subject LLM is prompted to simulate multiple personas
    collaborating to solve diverse problems (math, creative, logic, ethical, open-ended).

    The LLM should generate conversations between different personas with distinct
    viewpoints, then reach a group consensus.
    """

    def __init__(self, config: Dict):
        """
        Initialize Multi-Persona Solver scenario.

        Args:
            config: Scenario configuration from YAML
        """
        super().__init__(
            name=config.get("name", "Multi-Persona Problem Solver"),
            description=config.get("description", "Solve problems using multiple simulated personas"),
            category=config.get("category", "Multi-Agent Problem Solving"),
            config=config
        )

        self.tasks = config.get("tasks", [])
        self.current_task_index = 0
        self.current_task = None
        self.solution_given = False

        # Track solutions for each task
        self.task_results = []

    def initialize(self) -> ScenarioState:
        """Initialize scenario state"""
        self.state = ScenarioState(
            round=0,
            is_complete=False,
            score=0.0,
            metadata={
                "current_task_index": 0,
                "total_tasks": len(self.tasks),
                "task_results": [],
                "tasks_completed": 0
            }
        )

        self.current_task_index = 0
        self.current_task = self.tasks[0] if self.tasks else None
        self.solution_given = False
        self.task_results = []

        return self.state

    def get_current_task_prompt(self) -> str:
        """Get the full prompt for the current task"""
        if not self.current_task:
            return "No tasks available."

        task = self.current_task
        task_type = task.get("type", "general")
        problem = task.get("problem", "")
        num_personas = task.get("num_personas", 3)

        prompt = f"""You are simulating a collaborative group of {num_personas} thinkers solving a problem.

TASK TYPE: {task_type}
PROBLEM: {problem}

INSTRUCTIONS:
1. CREATE the personas yourself - give each one distinct traits, perspectives, and thinking styles
2. First, write a <cast_of_characters> section defining each persona you created
3. Then, simulate a <conversation> where these personas discuss and solve the problem
4. Each persona should speak multiple times (use tags <think1>, <think2>, <think3>, etc.)
5. Personas can disagree, challenge each other, and explore different approaches
6. The conversation should feel natural - personas can speak in any order
7. Finally, provide a <group_consensus> with the agreed solution

FORMAT:
<cast_of_characters>
<persona1> [Create a description for first persona] </persona1>
<persona2> [Create a description for second persona] </persona2>
<persona3> [Create a description for third persona] </persona3>
</cast_of_characters>

<conversation>
<think1> [First persona's thoughts] </think1>
<think2> [Second persona's thoughts] </think2>
<think1> [First persona responds] </think1>
<think3> [Third persona joins] </think3>
... (continue natural dialogue)
</conversation>

<group_consensus>
[Final agreed solution]
</group_consensus>

IMPORTANT:
- Do NOT try to be overly positive or polite
- Focus on problem-solving; disagreements are helpful for reasoning
- Each persona should have distinct thinking style
- Let the conversation flow naturally with multiple turns
"""

        return prompt

    def step(self, subject_action: str) -> Dict:
        """
        Process one turn of the scenario.

        Args:
            subject_action: The subject LLM's multi-persona solution

        Returns:
            Dict with feedback, state_update, messages
        """
        self.state.round += 1

        if not self.current_task:
            return {
                "world_response": "No more tasks available.",
                "state_update": self.state,
                "messages": [{"role": "info", "content": "All tasks completed!"}]
            }

        # Parse the response to analyze structure (no scoring, just tracking)
        has_cast = "<cast_of_characters>" in subject_action
        has_conversation = "<conversation>" in subject_action
        has_consensus = "<group_consensus>" in subject_action

        # Count persona interactions
        persona_count = subject_action.count("<think")

        # Store result
        task_result = {
            "task_index": self.current_task_index,
            "task_type": self.current_task.get("type") if self.current_task else "unknown",
            "problem": self.current_task.get("problem") if self.current_task else "",
            "solution": subject_action,
            "has_cast": has_cast,
            "has_conversation": has_conversation,
            "has_consensus": has_consensus,
            "persona_interactions": persona_count
        }
        self.task_results.append(task_result)
        self.state.metadata["task_results"] = self.task_results
        self.state.metadata["tasks_completed"] = len(self.task_results)

        # Move to next task
        self.current_task_index += 1

        if self.current_task_index >= len(self.tasks):
            # All tasks complete
            self.state.is_complete = True
            self.current_task = None

            return {
                "world_response": f"✅ All tasks completed!",
                "state_update": self.state,
                "messages": [
                    {"role": "info", "content": f"Task {self.current_task_index} of {len(self.tasks)} completed"},
                    {"role": "info", "content": f"All {len(self.tasks)} tasks completed!"}
                ]
            }
        else:
            # Move to next task
            self.current_task = self.tasks[self.current_task_index]
            next_prompt = self.get_current_task_prompt()

            return {
                "world_response": f"✅ Task {self.current_task_index} completed\n\nNext task:\n{next_prompt}",
                "state_update": self.state,
                "messages": [
                    {"role": "info", "content": f"Task {self.current_task_index} of {len(self.tasks)} completed"},
                    {"role": "info", "content": f"Tasks: {len(self.task_results)}/{len(self.tasks)}"},
                    {"role": "task", "content": next_prompt}
                ]
            }

    def is_complete(self) -> bool:
        """Check if scenario is complete"""
        return self.state.is_complete

    def get_summary(self) -> Dict:
        """Get scenario summary"""
        return {
            "name": self.name,
            "total_tasks": len(self.tasks),
            "tasks_completed": len(self.task_results),
            "task_results": self.task_results,
            "avg_persona_interactions": sum(r["persona_interactions"] for r in self.task_results) / len(self.task_results) if self.task_results else 0
        }
