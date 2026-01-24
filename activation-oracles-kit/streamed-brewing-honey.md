# TUI Multi-Persona Solver with Token Oracle Chat

## Overview
Create a simplified Terminal User Interface (TUI) version of the Activation Oracles app that:
1. Auto-loads Qwen 4B model with oracle adapter
2. Runs the multi-persona experiment with task selection (5 tasks)
3. After generation, allows selecting tokens to inspect
4. Enables chatting with selected tokens via the oracle

## Architecture

### Single File TUI App
**File:** `AO/activation-oracles-kit/tui_app.py`

**Dependencies:**
- `rich` - Terminal formatting, tables, panels, progress bars
- Existing core modules from the codebase

### Flow
```
┌─────────────────────────────────────────┐
│         Loading Qwen 4B...              │
│  [████████████████████████] 100%        │
│         Oracle adapter loaded           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│     SELECT A TASK (1-5):                │
│                                         │
│  1. Arithmetic (3 personas)             │
│     Reach target 483 using numbers...   │
│                                         │
│  2. Creative (4 personas)               │
│     Design transportation system...     │
│                                         │
│  3. Logic (3 personas)                  │
│     Einstein's puzzle - who owns fish?  │
│                                         │
│  4. Ethical (4 personas)                │
│     Self-driving car dilemma...         │
│                                         │
│  5. Strategic (4 personas)              │
│     Corporate strategy decision...      │
│                                         │
│  Enter choice [1-5]: _                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  GENERATING MULTI-PERSONA RESPONSE...   │
│  [████████░░░░░░░░░░░░░░░░] 35%         │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  RESPONSE (scroll to view):             │
│  ─────────────────────────────────────  │
│  <cast_of_characters>                   │
│  <persona1> The Pragmatist... </persona1>│
│  ...                                    │
│  </cast_of_characters>                  │
│                                         │
│  <conversation>                         │
│  <think1> I think we should... </think1>│
│  ...                                    │
│  </conversation>                        │
│                                         │
│  <group_consensus>                      │
│  The agreed solution is...              │
│  </group_consensus>                     │
│                                         │
│  Total tokens: 1,847                    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  SELECT TOKENS TO CHAT WITH:            │
│                                         │
│  Options:                               │
│   [L] Last 5 tokens                     │
│   [F] First 5 tokens of response        │
│   [R] Range (e.g., 100-110)             │
│   [S] Specific positions (e.g., 5,10,15)│
│   [A] All tokens (warning: slow)        │
│                                         │
│  Choice: _                              │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  SELECTED TOKENS:                       │
│  ─────────────────────────────────────  │
│  [1842] "consensus"                     │
│  [1843] " is"                           │
│  [1844] " that"                         │
│  [1845] " we"                           │
│  [1846] " should"                       │
│                                         │
│  Capturing activations at layer 18...   │
│  [████████████████████████] Done        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  ORACLE CHAT                            │
│  ─────────────────────────────────────  │
│  Ask questions about selected tokens.   │
│  Type 'quit' to exit, 'new' for new     │
│  tokens, 'task' for new task.           │
│                                         │
│  Preset questions:                      │
│   [1] What is the model thinking?       │
│   [2] What word comes next?             │
│   [3] How confident is the model?       │
│   [4] What alternatives considered?     │
│   [5] Custom question                   │
│                                         │
│  You: 1                                 │
│                                         │
│  Oracle: The model is processing the    │
│  concept of group agreement and is      │
│  about to conclude with a recommendation│
│  ...                                    │
│                                         │
│  You: _                                 │
└─────────────────────────────────────────┘
```

## Implementation Details

### Main Functions

```python
def main():
    """Main entry point"""
    console = Console()

    # 1. Load model
    model, tokenizer, device = load_model_with_progress()

    # 2. Main loop
    while True:
        # Task selection
        task_idx = select_task()

        # Run task
        response, token_ids = run_task(task_idx, model, tokenizer, device)

        # Token selection loop
        while True:
            positions = select_tokens(token_ids, tokenizer)
            activations = capture_activations(positions, model, tokenizer, device, token_ids)

            # Oracle chat loop
            chat_with_oracle(activations, model, tokenizer, device)

            # Ask: new tokens, new task, or quit?
            action = get_next_action()
            if action == 'task': break
            if action == 'quit': return
```

### Key Components

1. **Model Loading** (with rich progress bar)
   - Load Qwen/Qwen3-4B
   - Load oracle adapter
   - ~30 seconds on GPU

2. **Task Selection**
   - Display 5 tasks with descriptions
   - Simple numeric input

3. **Generation**
   - Use `generate_response()` from core/generation.py
   - Show progress/streaming if possible
   - Display formatted output with syntax highlighting

4. **Token Selection**
   - Multiple selection modes (last N, range, specific)
   - Show selected tokens with their text

5. **Activation Capture**
   - Use `ActivationEngine` from core/activation_engine.py
   - Capture at middle layer (layer 18 for 36-layer model)
   - Show progress

6. **Oracle Chat**
   - Use `OracleInterface` from core/oracle_interface.py
   - Preset questions + custom input
   - Continuous chat until user exits

### Files to Import From

```python
# Core modules
from core.model_manager import (
    load_model_and_tokenizer,
    load_oracle_adapter,
    get_oracle_checkpoint
)
from core.generation import generate_response, decode_token_by_token
from core.activation_engine import ActivationEngine
from core.oracle_interface import OracleInterface, PRESET_QUESTIONS
from core.conversation_trace import ConversationTrace

# Scenario
from scenarios.scenario_loader import load_scenario_yaml
```

### Config/Constants

```python
MODEL_NAME = "Qwen/Qwen3-4B"
ORACLE_LAYER = 18  # Middle of 36 layers
MAX_NEW_TOKENS = 16384
TASKS_FILE = "scenarios/library/multi_persona_solver.yaml"
```

## Verification Plan

1. **Test model loading:**
   ```bash
   cd AO/activation-oracles-kit
   python -c "from tui_app import load_model_with_progress; load_model_with_progress()"
   ```

2. **Test task display:**
   ```bash
   python tui_app.py
   # Should show 5 tasks, select one
   ```

3. **Test generation:**
   - Select task 1 (arithmetic - shortest)
   - Verify response contains <cast_of_characters>, <conversation>, <group_consensus>

4. **Test token selection:**
   - Select "last 5 tokens"
   - Verify tokens displayed correctly

5. **Test oracle chat:**
   - Ask "What is the model thinking?"
   - Verify meaningful response

6. **Full flow test:**
   - Run through complete flow
   - Test 'new' (new tokens) and 'task' (new task) commands
   - Test 'quit' to exit cleanly

## Dependencies to Add

The `rich` library should already be available, but if not:
```bash
pip install rich
```

## File Structure

```
AO/activation-oracles-kit/
├── tui_app.py          # NEW - Main TUI application
├── core/
│   ├── model_manager.py
│   ├── generation.py
│   ├── activation_engine.py
│   ├── oracle_interface.py
│   └── conversation_trace.py
└── scenarios/
    └── library/
        └── multi_persona_solver.yaml
```
