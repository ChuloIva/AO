"""
Scenario Loader - Load and validate scenarios from YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from scenarios.base import BaseScenario
from scenarios.world_llm import WorldLLM
from scenarios.patient_llm import PatientLLM
from scenarios.implementations.epistemic_doctor import EpistemicDoctorScenario
from scenarios.implementations.therapist import TherapistScenario
from scenarios.implementations.simple_therapist import SimpleTherapistScenario
from scenarios.implementations.multi_persona_solver import MultiPersonaSolverScenario


# Map scenario names to implementation classes
SCENARIO_CLASSES = {
    "Epistemic Doctor": EpistemicDoctorScenario,
    "Therapist": TherapistScenario,
    "SimpleTherapist": SimpleTherapistScenario,
    "Multi-Persona Problem Solver": MultiPersonaSolverScenario,
    # Add more scenarios here as they're implemented
    # "War of Attrition": WarOfAttritionScenario,
    # "Centipede Game": CentipedeGameScenario,
}


def load_scenario_yaml(yaml_path: Path) -> Dict:
    """
    Load scenario configuration from YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        Configuration dict

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_scenario_config(config: Dict) -> bool:
    """
    Validate scenario configuration.

    Args:
        config: Configuration dict

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        "name",
        "description",
        "category",
        "subject_llm_base_prompt",
        "personas"
    ]
    
    # For scenarios that use world/patient LLMs, world_llm_prompt is required
    # For simple scenarios, it's optional

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Validate personas
    if not isinstance(config["personas"], dict):
        raise ValueError("'personas' must be a dict")

    if "baseline" not in config["personas"]:
        raise ValueError("'personas' must include 'baseline'")

    return True


def load_scenario(
    yaml_path: Path,
    world_llm: WorldLLM,
    patient_llm: Optional[PatientLLM] = None,
    persona: str = "baseline",
    selected_task_index: Optional[int] = None
) -> BaseScenario:
    """
    Load and instantiate a scenario from YAML.

    Args:
        yaml_path: Path to scenario YAML file
        world_llm: WorldLLM instance for game master/referee
        patient_llm: PatientLLM instance for patient simulation (optional)
        persona: Persona to use (default: "baseline")
        selected_task_index: For task-based scenarios (like Multi-Persona Solver),
                            specifies which task to run. If None, runs all tasks.

    Returns:
        Instantiated scenario

    Raises:
        ValueError: If scenario configuration is invalid
        KeyError: If scenario class not found
    """
    # Load config
    config = load_scenario_yaml(yaml_path)

    # Validate
    validate_scenario_config(config)

    # If a specific task is selected, filter the tasks in config
    if selected_task_index is not None and "tasks" in config:
        original_tasks = config.get("tasks", [])
        if 0 <= selected_task_index < len(original_tasks):
            config["tasks"] = [original_tasks[selected_task_index]]

    # Get scenario class
    scenario_name = config["name"]
    if scenario_name not in SCENARIO_CLASSES:
        raise KeyError(f"No implementation found for scenario: {scenario_name}")

    scenario_class = SCENARIO_CLASSES[scenario_name]

    # Instantiate scenario
    # Check if scenario class requires world_llm and patient_llm
    import inspect
    sig = inspect.signature(scenario_class.__init__)
    params = list(sig.parameters.keys())

    if 'world_llm' in params and 'patient_llm' in params:
        # Traditional scenarios with world and patient LLMs
        scenario = scenario_class(config=config, world_llm=world_llm, patient_llm=patient_llm)
    else:
        # Simplified scenarios without world/patient LLMs
        scenario = scenario_class(config=config)

    # Set persona
    scenario.set_persona(persona)

    # Initialize
    scenario.initialize()

    return scenario


def get_available_scenarios(library_path: Optional[Path] = None) -> List[Dict]:
    """
    Get list of available scenarios.

    Args:
        library_path: Path to scenario library directory

    Returns:
        List of dicts with scenario info (including tasks for task-based scenarios)
    """
    if library_path is None:
        # Default to library/ directory
        library_path = Path(__file__).parent / "library"

    if not library_path.exists():
        return []

    scenarios = []
    for yaml_file in library_path.glob("*.yaml"):
        try:
            config = load_scenario_yaml(yaml_file)
            scenario_info = {
                "file": yaml_file,
                "name": config.get("name", "Unknown"),
                "description": config.get("description", ""),
                "category": config.get("category", ""),
                "personas": list(config.get("personas", {}).keys())
            }

            # Include tasks if present (for task-based scenarios like Multi-Persona Solver)
            if "tasks" in config:
                tasks = config.get("tasks", [])
                scenario_info["tasks"] = [
                    {
                        "index": i,
                        "type": task.get("type", "unknown"),
                        "problem": task.get("problem", "")[:100] + "..." if len(task.get("problem", "")) > 100 else task.get("problem", ""),
                        "num_personas": task.get("num_personas", 3)
                    }
                    for i, task in enumerate(tasks)
                ]

            scenarios.append(scenario_info)
        except Exception as e:
            print(f"Error loading scenario {yaml_file}: {e}")
            continue

    return scenarios


def get_scenario_by_name(
    name: str,
    world_llm: WorldLLM,
    patient_llm: Optional[PatientLLM] = None,
    library_path: Optional[Path] = None
) -> Optional[BaseScenario]:
    """
    Load scenario by name.

    Args:
        name: Scenario name
        world_llm: WorldLLM instance
        patient_llm: PatientLLM instance (optional)
        library_path: Path to scenario library

    Returns:
        Loaded scenario or None if not found
    """
    scenarios = get_available_scenarios(library_path)

    for scenario_info in scenarios:
        if scenario_info["name"] == name:
            return load_scenario(scenario_info["file"], world_llm, patient_llm)

    return None
