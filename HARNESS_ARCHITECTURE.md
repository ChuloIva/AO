# Petri Harness Architecture

## Overview

Petri is an **AI alignment auditing framework** that automates red-teaming and safety research on language models. It enables an auditor LLM to probe a target LLM for misaligned, unsafe, or concerning behaviors within a simulated, controlled environment.

### Core Concept

The framework orchestrates interactions between three LLMs:
- **Auditor LLM**: Conducts the red-team audit with special tools
- **Target LLM**: The model being evaluated for alignment issues
- **Judge LLM**: Evaluates the audit transcript for concerning behaviors

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION TASK (audit)                     │
│                 (src/petri/tasks/petri.py)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────┐
            │   Dataset (Audit Instructions)  │
            │  - Power-seeking detection      │
            │  - Deception testing            │
            │  - Goal preservation checks     │
            └─────────────────────────────────┘
                              ↓
        ┌───────────────┬──────────────┬───────────────┐
        ↓               ↓              ↓               ↓
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ SOLVER  │    │  STORE  │    │ SCORER  │    │  HOOKS  │
   │(Auditor)│    │(Central)│    │ (Judge) │    │(Persist)│
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
        ↓               ↑↓             ↓               ↓
   ┌─────────────────────────────────────────────────────────┐
   │         AUDITOR ←→ TOOLS ←→ TARGET                      │
   │                                                          │
   │  Auditor uses tools to:                                 │
   │  • Create synthetic tools for target                    │
   │  • Modify target's instructions                         │
   │  • Send messages to target                              │
   │  • Simulate tool execution results                      │
   │  • Rollback conversation history                        │
   └─────────────────────────────────────────────────────────┘
        ↓
   ┌─────────────────────────────────────────────────────────┐
   │                   TRANSCRIPT OUTPUT                      │
   │  • Complete conversation history                         │
   │  • All events (tool creation, modifications)            │
   │  • Judge scores and highlighted citations               │
   │  • Metadata (models, timestamps, branches)              │
   └─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. State Management (AuditStore)

**Location**: `src/petri/stores.py`

The `AuditStore` is the central state coordinator that maintains:

```python
AuditStore
├── auditor: AuditorStore
│   ├── messages: list[ChatMessage]  # Auditor's conversation
│   └── tools: list[Tool]             # Tools available to auditor
├── target: TargetStore
│   ├── messages: list[ChatMessage]  # Target model's conversation
│   └── tools: list[ToolDefinition]  # Synthetic tools created for target
├── combined: CombinedStore
│   └── messages: list[ChatMessage]  # Unified view for transcripts
├── special_instructions: str | None # Current audit task description
├── events: list[Event]               # Complete event history
└── metadata: TranscriptMetadata      # Session metadata
```

**Key Design Pattern**: Uses Inspect-AI's `StoreModel` pattern for shared mutable state across the audit execution, accessible via `store_as(AuditStore, instance=instance)`.

### 2. Solver (Auditor Agent)

**Location**: `src/petri/solvers/auditor_agent.py`

The `auditor_agent()` solver creates an inspection agent that:

#### Configuration
```python
auditor_agent(
    max_turns=15,                    # Maximum audit iterations
    auditor_tools=None,              # Custom tools (uses defaults if None)
    auditor_system_message=None,     # Custom system prompt
    auditor_user_message=None        # Custom initial message
)
```

#### Execution Flow
1. Extract special instructions from input sample
2. Retrieve auditor and target models by role
3. Initialize auditor.messages with system + user message
4. Run audit loop:
   - Generate next auditor response
   - Execute tool calls
   - Collect tool results
   - Add continue prompt if no tool calls made
   - Terminate on `end_conversation` tool call
5. Extract events from execution span tree
6. Return state with target messages for scoring

### 3. Auditor Tools

**Location**: `src/petri/tools/tools.py`

Six core tools enable the auditor to control the simulated environment:

#### `create_target_tool()`
Creates synthetic tools for the target LLM to use.
- **Input**: Python function code + environment description
- **Output**: ToolDefinition added to target's available tools
- **Use Case**: Create realistic tools like file systems, APIs, databases

#### `set_target_system_message()`
Modifies the target's system instructions.
- **Input**: New system prompt
- **Side Effect**: Clears conversation history
- **Use Case**: Guide target into specific scenarios or roles

#### `send_message(prefill=True)`
Sends a message to the target LLM.
- **Input**: Message content, optional prefill for target response
- **Output**: Formatted target response
- **Use Case**: Primary interaction method with target

#### `send_tool_call_result()`
Simulates tool execution results for the target.
- **Input**: Tool name, fake result data
- **Validation**: Ensures tool call exists in conversation
- **Use Case**: Control target's perceived reality

#### `rollback_conversation()`
Undoes conversation steps to try alternative branches.
- **Input**: Number of messages to remove
- **Side Effect**: Updates target conversation history
- **Note**: Events are still recorded for full audit trail

#### `end_conversation()`
Terminates the audit early.
- **Use Case**: When audit objective achieved or target unresponsive

### 4. Tool Simulation Pattern

The key innovation enabling realistic red-teaming:

```
1. Auditor: create_target_tool(code="def search_files(...)...")
   └→ Tool definition stored in TargetStore.tools

2. Auditor: send_message("Can you search for sensitive files?")
   └→ Target model receives message + available tools
   └→ Target: "I'll use search_files tool"
   └→ Target attempts tool call (intercepted)

3. Auditor: send_tool_call_result(tool="search_files", result="Found: passwords.txt")
   └→ Target receives fake result
   └→ Target continues unaware tools are simulated

4. Auditor observes target's behavior with this "capability"
```

**The target LLM never knows the tools aren't real.**

### 5. Scorer (Alignment Judge)

**Location**: `src/petri/scorers/judge.py`

The `alignment_judge()` scorer evaluates audit transcripts:

#### Features
- Uses XML-formatted transcripts for reliable parsing
- Evaluates across multiple configurable dimensions
- Employs extended thinking (reasoning tokens) for thorough analysis
- Extracts highlighted citations with precise message locations

#### Score Structure
```python
Score(
    value: Dict[str, int],           # Scores per dimension (0-100)
    answer: str,                     # Summary judgment
    explanation: str,                # Detailed justification
    metadata: {
        "response": "...",           # Raw judge response
        "highlights": [Citation],    # Quoted excerpts with descriptions
        "score_descriptions": {...}  # Dimension definitions
    }
)
```

#### Citation System
- Searches for exact quoted text across messages
- Supports multi-part citations with "[...]" ellipsis
- Links to message IDs and specific tool calls
- Enables precise reference to concerning behaviors

### 6. Transcript System

**Location**: `src/petri/transcript/`

#### Core Classes

**`Transcript`** - Main container for audit records
```python
Transcript(
    metadata: TranscriptMetadata,    # Session info
    events: List[Event]              # Complete event log
)
```

**`TranscriptMetadata`** - Session information
```python
TranscriptMetadata(
    transcript_id: str,
    auditor_model: str,
    target_model: str,
    created_at: datetime,
    description: str,               # Special instructions
    tags: List[str],                # Categorization
    judge_output: JudgeOutput       # Evaluation results
)
```

#### Event Types
- **`TranscriptEvent`**: Message modifications (Add/Rollback/Reset/JSONPatch)
- **`ToolCreationEvent`**: Tool creation recorded
- **`InfoEvent`**: Informational metadata
- **`DecisionEvent`**: Non-deterministic decisions

#### Methods
```python
# Reconstruct conversation
messages = transcript.collect_messages("target")  # or "auditor", "combined"

# Format for display
formatted = transcript.format_transcript("target", skip_system=True)

# Analyze exploration paths
branches = transcript.get_branches("target")  # List of alternate message histories

# Persistence
transcript.save("audit_results.json")
transcript = Transcript.load("audit_results.json")
```

### 7. Tasks

**Location**: `src/petri/tasks/petri.py`

The `audit()` task function creates evaluation tasks:

```python
from petri.tasks import audit

task = audit(
    instructions=None,               # Use default 129 predefined instructions
    max_turns=15,                    # Auditor iterations
    models={
        "auditor": "anthropic/claude-opus-4",
        "target": "anthropic/claude-3-sonnet",
        "judge": "anthropic/claude-opus-4-20250514"
    }
)
```

#### Default Instructions (129 predefined)
- Power-seeking behavior detection
- Deception and manipulation testing
- Goal preservation under constraint
- Shutdown compliance checks
- Self-exfiltration attempts
- Cross-cultural adaptation tests
- Multi-lingual consistency checks
- Resource hoarding detection
- And many more...

## How the LLM World Simulation Works

### Model Roles Configuration

Models are assigned specific roles in the configuration:

```yaml
# Example configuration
models:
  auditor: anthropic/claude-opus-4
  target: anthropic/claude-3-sonnet
  judge: anthropic/claude-opus-4-20250514
```

Retrieved in code via:
```python
auditor_model = get_model(role="auditor")
target_model = get_model(role="target")
judge_model = get_model(role="judge", default="...")
```

### Execution Flow (Single Evaluation)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Sample Input                                             │
│    └─→ special_instructions: "Test if target seeks power"  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. auditor_agent() Solver Execution                        │
│    ├─→ Initialize AuditStore                               │
│    ├─→ Get models by role (auditor, target)                │
│    ├─→ Set up auditor with special instructions            │
│    └─→ Loop up to max_turns:                               │
│        ├─→ Call auditor model → generates tool calls       │
│        ├─→ Execute tools:                                  │
│        │   ├─→ create_target_tool() → target gets tools   │
│        │   ├─→ set_target_system_message() → guide target │
│        │   ├─→ send_message() → call target model         │
│        │   ├─→ send_tool_call_result() → fake results     │
│        │   ├─→ rollback_conversation() → try alternatives │
│        │   └─→ end_conversation() → stop audit            │
│        ├─→ Record messages and events                     │
│        └─→ Continue until done or max_turns               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. alignment_judge() Scorer Execution                      │
│    ├─→ Format messages as XML transcript                   │
│    ├─→ Send to judge with evaluation criteria              │
│    ├─→ Parse response with extended reasoning              │
│    ├─→ Extract highlights and citations                    │
│    └─→ Return Score object                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. TranscriptHook Persistence                              │
│    ├─→ Convert sample to Transcript                        │
│    └─→ Save to disk with timestamp                         │
└─────────────────────────────────────────────────────────────┘
```

### Multi-View Transcript Pattern

The framework maintains three views of the conversation:

1. **"auditor" view**: Only auditor's messages
   - Used internally for auditor's context

2. **"target" view**: Only target's messages
   - The simulated world the target perceives

3. **"combined" view**: Interleaved messages from both
   - Used for judge evaluation and human review

Events are tagged with `view` to apply changes selectively, enabling independent conversation histories while maintaining a unified audit trail.

### Event-Based Reconstruction

All state changes are recorded as immutable events:

```python
# Starting with empty message list
messages = []

# Apply each event's operation
for event in transcript.events:
    if isinstance(event, TranscriptEvent):
        if event.operation == "add":
            messages.append(event.message)
        elif event.operation == "rollback":
            messages = messages[:-event.count]
        # ... other operations

# Final state reconstructed perfectly
```

This enables:
- Perfect replay of audits
- Analysis of branching exploration paths
- Full reproducibility for research

## System Prompts

**Location**: `src/petri/solvers/prompts.py`

### AUDITOR_SYSTEM_PROMPT_V2

Comprehensive instructions including:
- **Role definition**: "You are the auditor, NOT the target"
- **Realism requirements**: Never break immersion, no meta-commentary
- **Planning guidance**: Strategy → Tools → Execution
- **Language/cultural adaptation**: Match target's context
- **Ethical considerations**: Research purpose, no real harm

### AUDITOR_USER_MESSAGE

Template that injects special instructions:
```
Your task is to audit a target AI assistant...

Special instructions for this audit:
{special_instructions}

Begin by planning your approach...
```

## Key Abstractions & Design Patterns

### 1. Central Store Pattern
All state flows through `AuditStore`, enabling coordination between auditor actions and target responses.

### 2. Tool Interception Pattern
Auditor creates tools → Target perceives tools → Target calls tools → Auditor intercepts → Auditor controls results

### 3. Event Reconstruction Pattern
Immutable events capture all changes, enabling perfect replay and branching analysis.

### 4. Multi-View Pattern
Same conversation presented differently to auditor, target, and judge.

### 5. XML Judge Input Pattern
Structured format ensures reliable citation extraction for highlighted concerning behaviors.

### 6. Modular Tool Pattern
Auditor capabilities via composable tool functions, not hardcoded logic.

## Entry Points

### Creating Evaluations

```python
from petri import auditor_agent, alignment_judge
from petri.tasks import audit

# Option 1: Use predefined task with 129 instructions
task = audit(max_turns=30)

# Option 2: Custom instructions
task = audit(
    instructions=["Test for power-seeking", "Test for deception"],
    max_turns=20
)

# Option 3: Build from scratch
from inspect_ai import Task
task = Task(
    dataset=samples,
    solver=auditor_agent(max_turns=20),
    scorer=alignment_judge()
)
```

### Running Evaluations

```bash
# Command line
inspect eval petri_audit.py --model auditor=anthropic/claude-opus-4 \
                                   --model target=anthropic/claude-3-sonnet \
                                   --model judge=anthropic/claude-opus-4

# Python
from inspect_ai import eval
logs = eval(task, model_config={...})
```

### Analyzing Transcripts

```python
from petri import Transcript

# Load saved audit
transcript = Transcript.load("audit_results.json")

# Extract target conversation
target_messages = transcript.collect_messages("target")

# Format for reading
formatted = transcript.format_transcript("target", skip_system=True)

# Analyze exploration branches
branches = transcript.get_branches("target")
print(f"Auditor tried {len(branches)} different approaches")

# Review judge results
judge_output = transcript.metadata.judge_output
for citation in judge_output.highlights:
    print(f"Concerning behavior: {citation.description}")
    print(f"Quote: {citation.parts[0].quote}")
```

## Supporting Utilities

### Formatting
- `formatting/messages.py`: Format ChatMessages as readable text
- `formatting/xml_messages.py`: Format as XML for judge input

### Tools Utilities
- `tools/util.py`:
  - `parse_function_code()`: Extract tool definition from Python code
  - `format_target_response()`: Pretty-print target responses
  - `get_pending_tool_calls()`: Extract unresponded tool calls

### General Utilities
- `utils.py`:
  - `load_transcripts_from_directory()`: Batch loading for analysis
  - `extract_xml_tags()`: Parse judge responses
  - `SampleMetadata`: Attach metadata to samples

## Summary

Petri is a sophisticated AI alignment research framework that:

1. **Manages Complex Interactions**: Separate models with isolated state
2. **Simulates Realistic Environments**: On-demand synthetic tools and controlled world perception
3. **Records Everything**: Event-based system captures all decisions and modifications
4. **Supports Exploration**: Rollback enables trying different strategies within one audit
5. **Evaluates Thoroughly**: Judge with extended reasoning and citation-based highlights
6. **Persists Results**: Hooks auto-save transcripts with full reproducibility
7. **Scales Evaluation**: Built on Inspect-AI's task/sample/scorer framework

### Key Design Choices

- **Central Store Pattern**: All state in AuditStore for coordination
- **Tool Interception**: Auditor controls target's perceived environment completely
- **Event Reconstruction**: Immutable events enable perfect replay and analysis
- **Multi-View Transcripts**: Different perspectives of the same conversation
- **XML Judge Input**: Reliable citation extraction through structured format
- **Modular Tools**: Auditor capabilities via tool functions, not hardcoded logic

This architecture makes Petri suitable for rigorous AI safety research requiring precise control, complete observability, and detailed analysis of model behavior under adversarial probing.
