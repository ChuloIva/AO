# Activation Oracles Research Kit

A Streamlit-based research tool for exploring LLM internals through Activation Oracles.

## Features

- 🎭 **Scenarios**: Run psychological/psychiatric scenarios with different persona prompts
- 🔬 **Activation Capture**: Post-hoc activation capture from conversation traces
- 🔮 **Oracle Queries**: Query oracle LLM to understand model behavior
- ✏️ **Prompt Customization**: Fully customizable prompts for experimentation

## Setup

### Prerequisites

- Python 3.10+
- CUDA GPU (12GB+ VRAM) OR Mac with Apple Silicon (MPS)
- OpenRouter API key ([get one here](https://openrouter.ai))

### Installation

1. **Clone the repository:**
```bash
cd activation-oracles-kit
```

2. **Install the activation_oracles dependency:**
```bash
cd ../activation_oracles
pip install -e .
cd ../activation-oracles-kit
```

3. **Install requirements:**
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Phase 1: Setup (Current)

1. **Setup Tab**:
   - Select a model (recommended: Qwen3-1.7B for 12GB VRAM)
   - Load model with 8-bit quantization
   - Enter OpenRouter API key
   - Test connection and save configuration

2. **Scenarios Tab**:
   - Select a scenario (Epistemic Doctor currently available)
   - Choose a persona (baseline, anxious, impulsive, ocd, overconfident)
   - Run the scenario
   - Watch the subject LLM (doctor) interact with the world LLM (patient)

### Epistemic Doctor Scenario

A medical diagnosis scenario where the doctor must balance:
- **Information gathering** (ordering tests costs resources)
- **Decision making** (making a diagnosis ends the scenario)

**Personas affect behavior:**
- **Baseline**: Rational, balanced approach
- **Anxious**: Orders many tests, fears mistakes
- **Impulsive**: Quick decisions, trusts gut
- **OCD**: Cannot tolerate uncertainty, exhaustive testing
- **Overconfident**: Diagnoses with limited information

### Coming Soon (Phase 2+)

- **Analysis Tab**: Token selection, activation capture, oracle queries
- **Free Chat Tab**: Custom prompts without scenarios
- Additional scenarios (War of Attrition, Centipede Game)

## Architecture

```
activation-oracles-kit/
├── app.py                  # Main Streamlit app
├── core/                   # Core functionality
│   ├── model_manager.py    # Model loading (Mac + CUDA)
│   ├── conversation_trace.py
│   └── generation.py
├── scenarios/              # Scenario system
│   ├── base.py            # Base scenario class
│   ├── world_llm.py       # OpenRouter integration
│   ├── scenario_loader.py
│   └── library/           # Scenario YAML files
├── ui/                    # Streamlit UI
│   ├── state_manager.py
│   └── tabs/             # Tab implementations
└── prompts/
    └── presets/          # Persona prompts
```

## Supported Models (12GB VRAM Optimized)

| Model | VRAM (8-bit) | Oracle Checkpoint |
|-------|--------------|-------------------|
| **Qwen3-1.7B** | ~4GB | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B` |
| **Qwen3-4B** | ~6GB | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` |
| Llama-3.2-1B | ~3GB | `adamkarvonen/checkpoints_cls_latentqa_past_lens_Llama-3_2-1B-Instruct` |
| gemma-3-1b-it | ~3GB | `adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it` |

## Mac Support

The app works on Mac via PyTorch with MPS backend:
- No MLX support (would require complete rewrite)
- Quantization disabled on MPS (uses full precision)
- Flash attention disabled (uses eager attention)
- Slightly slower than CUDA but fully functional

## Development

### Current Status: Phase 1 MVP

- ✅ Model loading (12GB optimized)
- ✅ Mac support (MPS)
- ✅ OpenRouter World LLM
- ✅ Epistemic Doctor scenario
- ⏳ Activation capture (Phase 2)
- ⏳ Oracle queries (Phase 2)
- ⏳ Free chat (Phase 3)

### Testing

To test the Epistemic Doctor scenario:
1. Load Qwen3-1.7B model
2. Configure OpenRouter (use Claude Sonnet for best results)
3. Start Epistemic Doctor with "anxious" persona
4. Watch it order excessive tests before diagnosing

## Credits

Based on:
- Paper: [Activation Oracles](https://arxiv.org/abs/2512.15674)
- Code: [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)

## License

[Add license information]
