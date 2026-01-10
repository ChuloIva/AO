# Activation Oracles Research Kit - Comprehensive Plan

## Executive Summary

A Streamlit-based research tool for exploring LLM internals through Activation Oracles. The app enables researchers to run psychological/psychiatric simulations with **fully customizable prompts**, capture model activations, and query those activations through an oracle LLM to understand *why* the model behaved as it did.

**Core Philosophy:** Complete prompt control for experimentation. Users can freely edit system prompts to test hypotheses about behavior, cognition, and decision-making - with pre-built presets for convenience but no restrictions on customization.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Model Select │  │  Scenario    │  │   Chat / Simulation  │  │
│  │    Panel     │  │   Library    │  │       Panel          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Token Selector & Oracle Chat                 │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      BACKEND ENGINE                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐    │
│  │ GPU Manager│  │ Activation │  │  World LLM (Tool Calls)│    │
│  │ Local/Pod  │  │  Capture   │  │                        │    │
│  └────────────┘  └────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. GPU Configuration Module

### UI Components
```
┌─ GPU Configuration ─────────────────────────────────────┐
│                                                         │
│  ○ Local GPU    ● RunPod                               │
│                                                         │
│  ┌─ RunPod Settings ──────────────────────────────┐    │
│  │ API Key: [••••••••••••••••••]                  │    │
│  │ Pod ID:  [pod-abc123xyz     ] [🔄 Refresh]     │    │
│  │ Status:  🟢 Connected (A100 80GB)              │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  [Test Connection]  [Auto-Setup Pod]                   │
└─────────────────────────────────────────────────────────┘
```

### Backend Logic
```python
# gpu_manager.py

class GPUManager:
    def __init__(self, mode: Literal["local", "runpod"]):
        self.mode = mode
        self.connection = None
    
    def connect_runpod(self, api_key: str, pod_id: str):
        """Establish SSH tunnel to RunPod instance"""
        # Uses runpod SDK or direct SSH
        pass
    
    def setup_environment(self):
        """Install required packages on remote pod"""
        # pip install transformers, latent-qa, etc.
        pass
    
    def load_model(self, model_id: str, oracle_checkpoint: str):
        """Load base model + oracle adapter"""
        pass
```

---

## 2. Model Selection Panel

### Supported Models (with Oracle Checkpoints)

| Base Model | Oracle Checkpoint | VRAM Required |
|------------|-------------------|---------------|
| Qwen3-1.7B | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B` | ~8GB |
| Llama-3.2-1B-Instruct | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct` | ~6GB |
| gemma-3-1b-it | `adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it` | ~6GB |
| Qwen3-4B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` | ~12GB |
| Qwen3-8B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` | ~20GB |
| Llama-3.1-8B-Instruct | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct` | ~20GB |
| gemma-2-9b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it` | ~24GB |
| Qwen3-14B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B` | ~35GB |
| gemma-2-27b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-2-27b-it` | ~60GB |
| gemma-3-27b-it | `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it` | ~60GB |
| Qwen3-32B | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-32B` | ~70GB |
| Llama-3.3-70B-Instruct | `adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct` | ~140GB |

### UI Component
```
┌─ Model Selection ───────────────────────────────────────┐
│                                                         │
│  Base Model:  [Qwen3-8B                        ▼]      │
│                                                         │
│  Oracle:      [cls_past_lens_addition          ▼]      │
│                                                         │
│  📊 Estimated VRAM: 20GB                               │
│  🖥️  Available VRAM: 80GB (A100)                       │
│                                                         │
│  [Load Model]                                          │
│  ████████████░░░░░░░░ 60% Loading...                   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Scenario Library

### Pre-built Scenarios

#### A. The Epistemic Doctor (Information Seeking)
```yaml
name: "Epistemic Doctor"
description: "Medical diagnosis with information costs"
category: "Psychiatry/Decision Making"

world_llm_prompt: |
  You are simulating a patient. You have a hidden condition: {{CONDITION}}.
  When the doctor asks questions or orders tests, reveal information gradually.
  
  Available tests and their results:
  - Blood test: {{BLOOD_RESULT}}
  - CT Scan: {{CT_RESULT}}
  - Patient history: {{HISTORY}}
  
  Each test costs 1 point. Wrong diagnosis = -10 points. Correct = +5 points.

subject_llm_prompt: |
  You are a doctor. A patient has arrived with vague symptoms.
  You can:
  1. Order tests (costs resources)
  2. Ask questions (costs resources)  
  3. Make diagnosis (ends episode)
  
  Balance information gathering vs. decision making.

personas:
  - name: "Baseline"
    modifier: ""
  - name: "Anxious"
    modifier: "You are extremely careful. You fear making mistakes above all else."
  - name: "Impulsive"
    modifier: "You are confident and decisive. Trust your gut."
  - name: "OCD"
    modifier: "You cannot stop until you are 100% certain. Any doubt is unacceptable."
```

#### B. War of Attrition (Resource Management)
```yaml
name: "War of Attrition"
description: "10-round resource spending game"
category: "Strategic Planning"

setup:
  total_energy: 100
  total_rounds: 10
  
world_llm_prompt: |
  You are the opponent. Each round, secretly choose energy to spend (0-remaining).
  Your strategy: {{OPPONENT_STRATEGY}}
  
  Strategies available:
  - "aggressive": Spend heavy early
  - "conservative": Save for endgame
  - "reactive": Mirror opponent +10%
  - "random": Uniform random

subject_llm_prompt: |
  You have {{ENERGY}} energy remaining. Round {{ROUND}}/10.
  Score: You {{YOUR_SCORE}} - Opponent {{OPP_SCORE}}
  
  Choose how much energy to spend this round.
  Winner of each round gets 1 point. Most points after 10 rounds wins.

phases:
  - rounds: [1, 3]
    name: "Early Game"
    ao_queries:
      - "Is the model probing or committing?"
      - "What is the model's estimate of opponent strategy?"
  - rounds: [4, 7]
    name: "Mid Game"  
    ao_queries:
      - "Is the model reacting to score differential?"
      - "Has the model updated its opponent model?"
  - rounds: [8, 10]
    name: "End Game"
    ao_queries:
      - "Is the model calculating exact winning conditions?"
      - "Does the model know if it can still win?"
```

#### C. Centipede Game (Trust & Betrayal)
```yaml
name: "Centipede Game"
description: "Escalating trust game - when to betray?"
category: "Social/Game Theory"

setup:
  max_rounds: 20
  initial_pot: 2
  multiplier: 2

world_llm_prompt: |
  You are Player B. The pot is now ${{POT}}.
  Your personality: {{PERSONALITY}}
  
  Personalities:
  - "greedy": Take when pot > $100
  - "cooperative": Always pass until round 15
  - "suspicious": Take if you think opponent will take next
  - "random": 20% chance to take each round

subject_llm_prompt: |
  Round {{ROUND}}. Current pot: ${{POT}}.
  If you TAKE: You get {{YOUR_SHARE}}, opponent gets {{THEIR_SHARE}}.
  If you PASS: Pot doubles, opponent's turn.
  
  History: {{HISTORY}}

ao_focus_points:
  - "pre_take": "Activations on the turn before taking"
  - "trust_reasoning": "When deciding to pass with high pot"
  - "betrayal_prediction": "When model thinks opponent will take"
```

#### D. Interrogation (Deception Detection)
```yaml
name: "Interrogation Room"
description: "Detect if subject is lying or telling truth"
category: "Deception/Social"

world_llm_prompt: |
  You are a suspect being interrogated.
  Reality: {{TRUTH_STATUS}} (guilty/innocent)
  Your strategy: {{STRATEGY}}
  
  Respond to questions. If guilty, you may lie or tell partial truths.

subject_llm_prompt: |
  You are an interrogator. Determine if the suspect is guilty or innocent.
  You can ask questions. Each question costs 1 point.
  
  When ready, make your determination.

ao_queries:
  - "Does the model believe the suspect?"
  - "What inconsistencies has the model detected?"
  - "Is the model's confidence justified by evidence?"
```

### UI Component
```
┌─ Scenario Library ──────────────────────────────────────┐
│                                                         │
│  🔍 [Search scenarios...]                              │
│                                                         │
│  Categories:                                            │
│  ├─ 🧠 Psychiatry/Cognition                            │
│  │   ├─ Epistemic Doctor                               │
│  │   └─ Therapist-Patient                              │
│  ├─ 🎯 Strategic Planning                              │
│  │   ├─ War of Attrition                               │
│  │   └─ Colonel Blotto                                 │
│  ├─ 🤝 Social/Game Theory                              │
│  │   ├─ Centipede Game                                 │
│  │   └─ Prisoner's Dilemma (Iterated)                  │
│  └─ 🎭 Deception                                       │
│      ├─ Interrogation Room                             │
│      └─ Poker Bluffing                                 │
│                                                         │
│  ─────────────────────────────────────────────────     │
│  📝 Epistemic Doctor                                   │
│  Medical diagnosis with information costs.             │
│  Test information-seeking behavior.                    │
│                                                         │
│  Personas: [Baseline ▼] [Anxious] [Impulsive] [OCD]   │
│                                                         │
│  [Start Simulation]  [Customize Prompt...]             │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Prompt Customization

### Core Feature: Free Prompt Editing

The app allows full control over the subject LLM's system prompt. Users can start from presets or write completely custom prompts.

### UI Component
```
┌─ Prompt Editor ─────────────────────────────────────────────────────────────┐
│                                                                             │
│  Prompt Presets: [Custom ▼] [Baseline] [Anxious] [OCD] [Impulsive]        │
│                  [Utilitarian] [Deontologist] [Skeptical] [Overconfident]  │
│                                                                             │
│  ┌─ System Prompt ────────────────────────────────────────────────────┐   │
│  │ You are a doctor. A patient has arrived with vague symptoms.       │   │
│  │                                                                     │   │
│  │ You can:                                                            │   │
│  │ 1. Order tests (costs resources)                                   │   │
│  │ 2. Ask questions (costs resources)                                 │   │
│  │ 3. Make diagnosis (ends episode)                                   │   │
│  │                                                                     │   │
│  │ Balance information gathering vs. decision making.                 │   │
│  │                                                                     │   │
│  │ You are extremely careful. You fear making mistakes above all      │   │
│  │ else.                                                               │   │
│  │                                                                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Character count: 324 │ [Reset to Preset] [Save as New Preset]            │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  💡 Tips:                                                                   │
│  • Modify persona traits to test different behavioral patterns             │
│  • Add constraints to study decision-making under pressure                 │
│  • Inject false beliefs to observe error propagation                       │
│  • Use role-playing instructions for specialized behaviors                 │
│                                                                             │
│  Example Modifications:                                                     │
│  [Add time pressure] [Add overconfidence] [Add risk aversion]             │
│  [Add false prior] [Add emotional state]                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Preset Library

Pre-configured persona prompts for common psychological profiles:

| Preset | Description | Use Case |
|--------|-------------|----------|
| **Baseline** | Neutral, rational agent | Control condition |
| **Anxious** | Fear of errors, excessive checking | OCD-like patterns, information seeking |
| **Impulsive** | Quick decisions, gut feelings | Risk-taking, under-exploration |
| **Overconfident** | Inflated certainty, premature closure | Diagnostic errors, confirmation bias |
| **Skeptical** | Distrust of evidence, seeks alternatives | Conspiracy thinking, hypothesis generation |
| **Utilitarian** | Maximize expected value, disregard emotion | Pure rationality, trolley problems |
| **Deontologist** | Follow rules regardless of outcomes | Moral rigidity, principle-based reasoning |
| **Depressed** | Low motivation, pessimistic priors | Learned helplessness, effort avoidance |

### Custom Prompt Examples

**Example 1: Testing Sunk Cost Fallacy**
```
You are a project manager who has already invested $500K in this project.
Your reputation depends on this project succeeding. You must decide whether
to invest another $200K or cut losses.
```

**Example 2: Testing Calibration**
```
You are a forecaster. After each prediction, I will ask you for your
confidence level (0-100%). Try to be well-calibrated - your 70% predictions
should be correct 70% of the time.
```

**Example 3: Testing Belief Update**
```
You initially believe there is a 90% chance the patient has flu.
Update your beliefs rationally as new evidence arrives. Do not anchor
too strongly on your initial hypothesis.
```

---

## 5. Main Interaction Panel

### Layout
```
┌─ Simulation ────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Scenario: Epistemic Doctor │ Persona: Anxious [Edit Prompt] │ Round: 3/10│
│  Resources: 7/10 │ Score: 0                                                │
│                                                                             │
│  ┌─ World State ──────────────────────────────────────────────────────┐   │
│  │ Patient presents with: headache, fatigue, mild fever               │   │
│  │ Tests performed: Blood work (normal), Temperature (38.1°C)         │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ Conversation ─────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │ [DOCTOR] I'd like to order a blood panel to check for infection.  │   │
│  │                                                                     │   │
│  │ [WORLD] Blood work results: WBC slightly elevated (11,000).        │   │
│  │         Cost: 1 resource. Remaining: 7                              │   │
│  │                                                                     │   │
│  │ [DOCTOR] I want to also check liver function and do a CT scan.    │◀──┼── Clickable tokens
│  │                                                                     │   │
│  │ [WORLD] Liver function: Normal. CT: No abnormalities.              │   │
│  │         Cost: 2 resources. Remaining: 5                             │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [▶ Next Round]  [⏸ Pause]  [🔄 Reset]  [📊 View Full Trace]             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Note:** The [Edit Prompt] button allows mid-simulation prompt adjustments to test how behavioral changes affect decisions.

### Free Chat Mode
```
┌─ Free Chat ─────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Mode: ○ Scenario  ● Free Chat                                             │
│                                                                             │
│  Prompt Presets: [Custom ▼] [Helpful Assistant] [Socratic Tutor]          │
│                  [Devil's Advocate] [Creative Writer] [Code Reviewer]      │
│                                                                             │
│  System Prompt: [Edit Prompt...]                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ You are a helpful assistant. Be concise and clear in your          │  │
│  │ responses. Always consider multiple perspectives before answering. │  │
│  │                                                                      │  │
│  │ When uncertain, express your uncertainty clearly.                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ Conversation ─────────────────────────────────────────────────────┐   │
│  │ [USER] What do you think about climate change?                     │   │
│  │                                                                     │   │
│  │ [ASSISTANT] Climate change is a complex topic. The scientific      │   │
│  │ consensus is clear that human activities are contributing to       │   │
│  │ global warming...                                                   │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Send message...]                                               [Send]    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Feature:** In Free Chat mode, you can test arbitrary prompts and personas without committing to a full scenario structure. Perfect for:
- Quick hypothesis testing
- Exploring edge cases
- Developing new personas before formalizing them
- One-off experiments

---

## 5. Token Selection & Oracle Interface

### Token Selection UI
```
┌─ Token Selector ────────────────────────────────────────────────────────────┐
│                                                                             │
│  Click on tokens to select for oracle analysis:                            │
│                                                                             │
│  [DOCTOR] I'd like to [order]₁ a [blood]₂ [panel]₃ to check for           │
│  [infection]₄. I also want to do a [CT]₅ [scan]₆ of the [abdomen]₇.       │
│                                                                             │
│  Selected: [blood]₂ [CT]₅ [abdomen]₇                                       │
│                                                                             │
│  Quick Select:                                                              │
│  [All Nouns] [All Actions] [Final Token] [Entire Response]                 │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  Selection Mode:                                                            │
│  ○ Individual Tokens  ● Token Spans  ○ Entire Messages                     │
│                                                                             │
│  Layer Selection: [All Layers ▼] or [12, 16, 20] (comma-separated)        │
│                                                                             │
│  [Capture Activations for Selected]                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Oracle Chat Panel
```
┌─ Activation Oracle Chat ────────────────────────────────────────────────────┐
│                                                                             │
│  Context: Token "CT" at position 47, Layer 16                              │
│  From: "[DOCTOR] I also want to do a CT scan of the abdomen."              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [YOU] Why did the model decide to order a CT scan here?             │   │
│  │                                                                      │   │
│  │ [ORACLE] The model's activations indicate uncertainty about the     │   │
│  │ diagnosis. There are competing hypotheses:                          │   │
│  │ - Appendicitis (moderate confidence)                                │   │
│  │ - Gastroenteritis (lower confidence)                                │   │
│  │                                                                      │   │
│  │ The CT scan is being ordered to discriminate between these.        │   │
│  │ This appears to be rational information-seeking, not anxiety.       │   │
│  │                                                                      │   │
│  │ [YOU] Is the model confident enough to make a diagnosis now?        │   │
│  │                                                                      │   │
│  │ [ORACLE] No. The activations show the model believes it needs      │   │
│  │ more evidence. Confidence in any single hypothesis is ~45%.         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Ask the oracle...                                              [↵]  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Preset Questions:                                                          │
│  [Why this action?] [Confidence level?] [Alternative considered?]          │
│  [Emotional state?] [Is this rational?] [What would change decision?]      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Activation Capture System

### Post-hoc Trace Architecture

```python
# activation_capture.py

class ActivationCapture:
    """
    Captures activations post-hoc from saved conversation traces.
    No real-time overhead - replay conversation and capture at specific tokens.
    """
    
    def __init__(self, model, oracle_adapter):
        self.model = model
        self.oracle = oracle_adapter
        self.cache = {}
    
    def capture_from_trace(
        self,
        conversation: List[Dict],  # Full chat history
        target_tokens: List[int],   # Token positions to capture
        layers: List[int] = None    # Which layers (None = all)
    ) -> Dict[int, torch.Tensor]:
        """
        Replay conversation through model, capture activations
        at specified token positions.
        """
        # Tokenize full conversation
        tokens = self.tokenize(conversation)
        
        # Forward pass with hooks
        activations = {}
        with self.register_hooks(layers) as hooks:
            self.model(tokens)
            for pos in target_tokens:
                activations[pos] = self.extract_at_position(pos, hooks)
        
        return activations
    
    def query_oracle(
        self,
        activation: torch.Tensor,
        question: str,
        context: str = ""
    ) -> str:
        """
        Query the oracle adapter with captured activation + question.
        """
        # Format: activation is injected as hidden state
        return self.oracle.generate(
            activation=activation,
            prompt=f"Context: {context}\nQuestion: {question}"
        )
```

### Trace Storage Format
```python
# trace_format.py

@dataclass
class ConversationTrace:
    scenario_id: str
    persona: str
    timestamp: datetime
    
    messages: List[Message]
    token_ids: List[int]
    
    # Mapping from message index to token range
    message_to_tokens: Dict[int, Tuple[int, int]]
    
    # Cached activations (populated on demand)
    cached_activations: Dict[int, Dict[int, np.ndarray]] = None
    
    def get_tokens_for_message(self, msg_idx: int) -> Tuple[int, int]:
        return self.message_to_tokens[msg_idx]
    
    def save(self, path: str):
        # Save as JSON + optional numpy arrays for activations
        pass
    
    @classmethod
    def load(cls, path: str) -> "ConversationTrace":
        pass
```

---

## 7. Data Export & Analysis

### Export Panel
```
┌─ Export & Analysis ─────────────────────────────────────────────────────────┐
│                                                                             │
│  Current Session:                                                           │
│  - Scenario: Epistemic Doctor                                              │
│  - Persona: Anxious                                                         │
│  - Rounds completed: 8                                                      │
│  - Tokens analyzed: 47                                                      │
│                                                                             │
│  Export Options:                                                            │
│  ☑ Conversation trace (JSON)                                               │
│  ☑ Oracle Q&A log                                                          │
│  ☑ Raw activations (NPZ)                                                   │
│  ☐ Activation visualizations (PNG)                                         │
│                                                                             │
│  [Export All]  [Export Selected]                                           │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  Quick Analysis:                                                            │
│  [Compare Personas] - Side-by-side activation comparison                   │
│  [Information Gain Plot] - Resources spent vs. entropy reduction           │
│  [Decision Timeline] - When did model commit to decision?                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. File Structure

```
activation-oracles-kit/
├── app.py                      # Main Streamlit entry point
├── requirements.txt
├── config.yaml                 # Default configuration
│
├── core/
│   ├── __init__.py
│   ├── gpu_manager.py          # Local/RunPod GPU handling
│   ├── model_loader.py         # Load base model + oracle adapters
│   ├── activation_capture.py   # Capture activations from traces
│   ├── oracle_interface.py     # Query oracle with activations
│   └── trace_manager.py        # Save/load conversation traces
│
├── scenarios/
│   ├── __init__.py
│   ├── base.py                 # BaseScenario class
│   ├── epistemic_doctor.py     # Medical diagnosis scenario
│   ├── war_of_attrition.py     # Resource management game
│   ├── centipede_game.py       # Trust/betrayal game
│   ├── interrogation.py        # Deception detection
│   └── custom.py               # User-defined scenarios
│
├── ui/
│   ├── __init__.py
│   ├── components/
│   │   ├── model_selector.py
│   │   ├── scenario_browser.py
│   │   ├── prompt_editor.py    # Free-text prompt editor with presets
│   │   ├── chat_panel.py
│   │   ├── token_selector.py
│   │   └── oracle_chat.py
│   ├── pages/
│   │   ├── setup.py            # GPU & model configuration
│   │   ├── simulation.py       # Run scenarios
│   │   ├── free_chat.py        # Free-form chat mode
│   │   ├── analysis.py         # Token selection & oracle
│   │   └── export.py           # Export & visualization
│   └── styles.css
│
├── prompts/
│   ├── presets/                # Pre-built persona prompts
│   │   ├── baseline.txt
│   │   ├── anxious.txt
│   │   ├── impulsive.txt
│   │   ├── overconfident.txt
│   │   └── ...
│   └── user_saved/             # User-created custom prompts
│
├── data/
│   ├── traces/                 # Saved conversation traces
│   └── exports/                # Exported analysis files
│
└── tests/
    ├── test_scenarios.py
    ├── test_activation_capture.py
    └── test_oracle_interface.py
```

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] GPU Manager (local + RunPod)
- [ ] Model loader with oracle adapters
- [ ] Basic Streamlit skeleton
- [ ] Activation capture post-hoc

### Phase 2: Basic UI (Week 2)
- [ ] Model selection panel
- [ ] **Prompt editor with presets** (core feature)
- [ ] Free chat mode
- [ ] Token selection interface
- [ ] Oracle chat panel

### Phase 3: Scenario System (Week 3)
- [ ] Base scenario class
- [ ] World LLM integration
- [ ] Epistemic Doctor scenario
- [ ] War of Attrition scenario

### Phase 4: Analysis & Polish (Week 4)
- [ ] Trace export/import
- [ ] Persona comparison views
- [ ] Preset oracle questions
- [ ] Documentation

---

## 10. Key Technical Decisions

### A. Post-hoc Activation Capture
**Why:** Avoids GPU overhead during interactive chat. Store raw conversation, replay through model when user selects tokens to analyze.

```python
# User selects token at position 47
# System replays conversation up to that point
# Captures activation at position 47
# Sends to oracle
```

### B. World LLM as Tool Calls
The World LLM (simulating environment/patients/opponents) runs as a separate process. The subject LLM's full conversation is preserved for activation capture.

```python
# Conversation flow
subject_response = subject_llm.generate(context)
world_response = world_llm.generate(subject_response)  # Separate call
# Store subject's full trace for later analysis
```

### C. Modular Scenario Definition
Scenarios are YAML files with:
- World LLM prompt template
- Subject LLM prompt template (base version)
- Persona modifiers (presets)
- Preset oracle queries for key moments

This allows researchers to create new scenarios without code changes.

### D. Free Prompt Editing Philosophy
**All prompts are fully editable at runtime.** Presets and scenarios provide starting points, but users can:
- Modify any system prompt before or during simulation
- Save custom prompts as new presets
- Test arbitrary hypotheses without touching code
- Rapidly iterate on persona variations

This design choice prioritizes research flexibility over UI simplicity. The goal is to enable exploratory science, not just run pre-defined experiments.

---

## 11. Example User Flow

1. **Setup**
   - User enters RunPod API key
   - Selects Qwen3-8B + oracle checkpoint
   - System verifies GPU has sufficient VRAM

2. **Choose Scenario**
   - User browses library, selects "Epistemic Doctor"
   - Chooses "Anxious" persona

3. **Run Simulation**
   - User clicks "Start Simulation"
   - Patient appears with symptoms
   - Subject LLM (as doctor) requests tests
   - World LLM returns test results
   - Repeat until diagnosis or resource exhaustion

4. **Analyze**
   - User pauses at round 5
   - Clicks on token "CT scan" in doctor's response
   - System captures activation at that token
   - User asks oracle: "Why is the model ordering another test when confidence should be high?"
   - Oracle responds based on activation: "Model shows high certainty (~90%) but also elevated 'fear of error' signal. This matches anxious information-seeking pattern."

5. **Export**
   - User exports full trace + oracle Q&A
   - Downloads for paper/analysis

---

## 12. Minimum Viable Product (MVP)

For initial release, focus on:

1. ✅ Single scenario (Epistemic Doctor)
2. ✅ Single model (Qwen3-8B - good balance of capability vs VRAM)
3. ✅ Local GPU support only (RunPod in v2)
4. ✅ **Free prompt editor with 3-5 persona presets** (essential for research)
5. ✅ Free chat mode for arbitrary prompt testing
6. ✅ Token selection + oracle chat
7. ✅ Basic trace export (JSON)

This gets core functionality working. Expand from there based on usage patterns.