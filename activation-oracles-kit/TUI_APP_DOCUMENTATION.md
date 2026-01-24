# TUI Multi-Persona Solver with Token Oracle Chat

## Overview

An interactive terminal UI for exploring LLM activations through the Activation Oracle framework. This tool enables you to:

1. Run multi-persona problem solving tasks with a language model
2. Capture activations at specific token positions
3. Query an oracle LLM to understand the model's internal reasoning

## Features

### Model Loading
- **Model**: Qwen/Qwen3-4B (configurable)
- **Oracle Adapter**: Automatically loaded from HuggingFace
- **Device Support**: ROCm/AMD, CUDA/NVIDIA, MPS/Apple Silicon, CPU
- **Progress Tracking**: Rich progress bars during loading
- **Cleanup**: Automatic GPU cache clearing on exit (Ctrl+C or ESC)

### Multi-Persona Tasks

Five diverse problem-solving tasks available:

1. **Arithmetic** - Use numbers {25, 50, 75, 3, 6, 7} to reach target 483
   - 3 personas
   - Mathematical reasoning

2. **Creative** - Design eco-friendly transportation for 500k city
   - 4 personas
   - Divergent thinking, stakeholder perspectives

3. **Logic** - Einstein's zebra puzzle (who owns the fish?)
   - 3 personas
   - Deductive reasoning

4. **Ethical** - Self-driving car moral dilemma
   - 4 personas
   - Ethical reasoning, multiple frameworks

5. **Strategic** - Corporate strategy decision ($50M redesign vs $200M acquisition vs pivot)
   - 4 personas
   - Business strategy, risk assessment

### Token Selection

**Interactive token browser with visual styling:**

- **XML Tags**: Displayed in dim color (not selectable)
  - `<cast_of_characters>`
  - `<persona1>`, `<persona2>`, etc.
  - `<conversation>`
  - `<think1>`, `<think2>`, etc.
  - `<group_consensus>`

- **Unselected Tokens**: Default color
- **Selected Tokens**: **Bold Yellow**
- **Highlighted Token**: Underline Magenta

**Navigation Controls:**
- `←` (Left Arrow): Previous token
- `→` (Right Arrow): Next token
- `↑` (Up Arrow): Previous line (page scroll)
- `↓` (Down Arrow): Next line (page scroll)
- `Space`: Toggle token selection
- `Q`: Chat with selected tokens
- `C`: Clear all selections
- `N`: New task
- `ESC`: Quit application

### Layer Selection

Choose which layers to capture activations from:

- **Layer 10**: Earlier processing (more abstract concepts)
- **Layer 18**: Middle layer (default, balanced)
- **Layer 25**: Later processing (more task-specific)

Multiple layers can be selected simultaneously.

### Oracle Chat

**Preset Questions:**

1. **What is the model thinking?** - Understand current reasoning process
2. **What word comes next?** - Predict next token
3. **How confident is the model?** - Assess decision confidence
4. **What alternatives considered?** - See alternative options explored
5. **Custom question** - Ask any question about the activations

**Chat Flow:**
- Type `1-5` for preset questions
- Type custom question for specific queries
- `T` returns to token selection (preserves selections)
- `N` starts new task
- `ESC` quits application

### Model Cleanup

Automatic cleanup on exit:
- Deletes model, tokenizer, activation engine, oracle interface
- Runs Python garbage collection
- Clears GPU cache (CUDA/MPS)
- Handles Ctrl+C gracefully with cleanup

## Usage

### Running the TUI App

```bash
# Navigate to the activation-oracles-kit directory
cd AO/activation-oracles-kit

# Activate virtual environment
source .venv/bin/activate

# Run the TUI app
python3 tui_app.py
```

### Workflow

1. **Model Loading**
   - Progress bar shows loading steps
   - Validates model state before adapter loading
   - Checks for offloaded parameters (meta device)
   - Clears GPU cache before adapter loading

2. **Task Selection**
   - Choose 1-5 from menu
   - Each task shows type and description

3. **Generation**
   - Model generates multi-persona response
   - Progress bar during generation
   - Full response displayed with formatting

4. **Layer Selection**
   - Toggle between layers 10, 18, 25
   - Multiple layers can be selected

5. **Token Selection**
   - Browse response with arrow keys
   - Toggle tokens with Space
   - Selected tokens highlighted in bold yellow
   - Current token underlined in magenta

6. **Activation Capture**
   - Captures activations at selected token positions
   - Progress bar for each layer
   - Uses caching for repeated captures

7. **Oracle Chat**
   - Ask questions about selected tokens
   - Chat history maintained in session
   - Returns to token selection with preserved state

## Visual Design

### Color Scheme

- **XML Tags**: Dim gray (structural markers, not selectable)
- **Normal Text**: White/terminal default
- **Selected Tokens**: Bold Yellow (`[bold yellow]`)
- **Highlighted Token**: Underline Magenta (`[underline magenta]`)

### Layout Examples

**Task Selection:**
```
SELECT A TASK
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Num Type         Description                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 1   ARITHMETIC   (3 personas) Using the numbers {25...   ┃
┃ 2   CREATIVE     (4 personas) Design a new public tr...   ┃
┃ 3   LOGIC        (3 personas) Five houses in a row,...    ┃
┃ 4   ETHICAL      (4 personas) A self-driving car is...    ┃
┃ 5   STRATEGIC    (4 personas) A company's flagship...     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Enter choice [1-5]:
```

**Token Selection:**
```
SELECT TOKENS TO CHAT WITH (toggle with Space)
← → : Navigate tokens  ↑ ↓ : Navigate lines  Space: Toggle
Q: Chat with selected  C: Clear all  N: New task  ESC: Quit

<dim><cast_of_characters></dim>
<dim><persona1></dim> The Pragmatist is a data-driven thinker who...
<dim><persona2></dim> The Visionary focuses on big-picture ideas...

<dim><conversation></dim>
<dim><think1></dim> I think we should [bold yellow]consider[/bold yellow] approach...
<dim><think2></dim> Actually, [underline magenta]maybe[/underline magenta] we should focus on...

Selected: 2 tokens | Total: 1847 tokens | Token 42/847
```

**Oracle Chat:**
```
ORACLE CHAT
Ask questions about selected tokens.
Preset questions:
  [1] What is the model thinking?
  [2] What word comes next?
  [3] How confident is the model?
  [4] What alternatives considered?
  [5] Custom question
Commands: T=Back to Tokens, N=New Task, ESC=Quit

── Layer 18 ──
You: 1
Oracle: The model is processing the concept of group agreement and
is about to conclude with a recommendation based on consensus...

You: 2
Oracle: The next likely tokens would be related to reaching a final
decision, such as "conclude", "final", or "agree"...
```

## Architecture

### Key Components

```python
@dataclass
class AppState:
    """Global application state"""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    oracle_adapter_name: str
    activation_engine: ActivationEngine
    oracle_interface: OracleInterface
    current_response: str
    current_token_ids: List[int]
    selected_layers: List[int]
    token_state: TokenSelectionState

@dataclass
class TokenSelectionState:
    """Token selection UI state"""
    selected_positions: Set[int]
    scroll_position: int
    highlighted_token_idx: int
```

### Main Functions

- `load_model_with_progress()`: Load model with validation
- `select_task()`: Task selection UI
- `run_task()`: Generate multi-persona response
- `select_layers()`: Layer selection UI
- `select_tokens_interactive()`: Interactive token browser
- `capture_activations_progress()`: Activation capture with progress
- `oracle_chat_ui()`: Oracle chat interface
- `cleanup_model()`: Model cleanup on exit

### Dependencies

- `rich`: Terminal UI, progress bars, styling
- `torch`: PyTorch for model operations
- `transformers`: HuggingFace model loading
- `peft`: LoRA adapter support
- `numpy`: Activation array operations
- `pyyaml`: Configuration loading

## Troubleshooting

### Common Issues

**1. ROCm/AMD GPU errors during adapter loading**
```
RuntimeError: HIP error: invalid device function
```
- The app now includes automatic recovery attempts
- Tries `torch.no_grad()` wrapper around adapter loading
- Validates model state before loading adapter

**2. "Some parameters are on meta device" warning**
- Occurs when model uses device offloading (accelerate)
- App automatically moves all parameters to GPU before adapter loading
- Clear GPU cache before adapter loading

**3. VRAM out of memory**
- TUI app loads full model (no 8-bit quantization)
- Requires ~12GB VRAM for Qwen 4B
- On smaller GPUs, consider using Qwen 1.7B instead

**4. Slow performance**
- Capturing many tokens can be slow
- Oracle queries take time for each activation
- Use caching for repeated captures (automatic)

## File Structure

```
AO/activation-oracles-kit/
├── tui_app.py              # Main TUI application
├── core/
│   ├── model_manager.py       # Model loading utilities
│   ├── generation.py         # Text generation
│   ├── activation_engine.py  # Activation capture
│   ├── oracle_interface.py   # Oracle queries
│   └── conversation_trace.py # Trace storage
└── scenarios/
    └── library/
        └── multi_persona_solver.yaml # Task definitions
```

## Configuration

### Constants in `tui_app.py`

```python
MODEL_NAME = "Qwen/Qwen3-4B"           # Model to load
ORACLE_LAYERS = [10, 18, 25]            # Available layers
TASKS_FILE = "scenarios/library/multi_persona_solver.yaml"
TOKENS_PER_PAGE = 20                        # Lines per page
```

### Configuration Files

- `config.yaml`: Global settings (ROCm config, paths)
- `scenarios/library/multi_persona_solver.yaml`: Task definitions

## Future Enhancements

Potential improvements:
- Mouse support for token selection
- Token search/filter functionality
- Export selected activations for external analysis
- Save/load oracle conversations
- Multi-token batch querying
- Layer comparison view
- Attention visualization

## References

- Based on: [Activation Oracles Paper](https://arxiv.org/abs/2512.15674)
- Code: [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
- Main App: `AO/activation-oracles-kit/app.py` (Streamlit UI)

## License

Same as parent project (Activation Oracles Research Kit).
