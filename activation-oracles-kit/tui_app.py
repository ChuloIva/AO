"""
TUI Multi-Persona Solver with Token Oracle Chat
Interactive terminal UI for exploring model activations via oracle queries
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Optional

# Import core modules FIRST - this ensures ROCm setup runs before torch is imported
# ROCm environment must be configured before importing torch
from core.model_manager import load_model_and_tokenizer, load_oracle_adapter, get_oracle_checkpoint, get_device
from core.generation import generate_response, generate_response_stream, decode_token_by_token

# Now import torch (ROCm should already be configured)
import torch

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
import select
import tty
import termios

from core.activation_engine import ActivationEngine
from core.oracle_interface import OracleInterface, PRESET_QUESTIONS
from core.conversation_trace import ConversationTrace
from scenarios.scenario_loader import load_scenario_yaml

# Constants
MODEL_NAME = "Qwen/Qwen3-4B"
ORACLE_LAYERS = [10, 18, 25]
TASKS_FILE = "scenarios/library/multi_persona_solver.yaml"
TOKENS_PER_PAGE = 20

# Key constants
KEY_UP = 'UP'
KEY_DOWN = 'DOWN'
KEY_LEFT = 'LEFT'
KEY_RIGHT = 'RIGHT'
KEY_ESCAPE = 'ESCAPE'
KEY_SPACE = 'SPACE'
KEY_ENTER = 'ENTER'


def read_key_raw() -> str:
    """
    Read a single keypress in raw mode, handling escape sequences.
    Must be called while terminal is already in raw mode.

    Returns:
        Key name (KEY_UP, KEY_DOWN, etc.) or the character pressed.
    """
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1).decode('utf-8', errors='ignore')

    if ch == '\x1b':  # Escape sequence start
        # Read more bytes with short timeout
        try:
            # Set non-blocking temporarily
            import fcntl
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            try:
                seq = os.read(fd, 2).decode('utf-8', errors='ignore')
            except BlockingIOError:
                seq = ''
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)

            if seq == '[A':
                return KEY_UP
            elif seq == '[B':
                return KEY_DOWN
            elif seq == '[C':
                return KEY_RIGHT
            elif seq == '[D':
                return KEY_LEFT
            else:
                return KEY_ESCAPE
        except Exception:
            return KEY_ESCAPE
    elif ch == ' ':
        return KEY_SPACE
    elif ch in ('\r', '\n'):
        return KEY_ENTER
    elif ch == '\x03':  # Ctrl+C
        raise KeyboardInterrupt
    else:
        return ch


def setup_terminal_raw():
    """Enter raw terminal mode, return old settings for restore."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old_settings


def restore_terminal(old_settings):
    """Restore terminal to original settings."""
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@dataclass
class TokenSelectionState:
    selected_positions: Set[int] = field(default_factory=set)
    scroll_position: int = 0
    highlighted_token_idx: int = 0


@dataclass
class AppState:
    model = None
    tokenizer = None
    device = None
    oracle_adapter_name = ""
    activation_engine: Optional[ActivationEngine] = None
    oracle_interface: Optional[OracleInterface] = None
    current_response: str = ""
    current_token_ids: List[int] = field(default_factory=list)  # Response tokens only (for browser)
    full_token_ids: List[int] = field(default_factory=list)  # Full sequence (for activation capture)
    prompt_length: int = 0  # Offset to map browser positions to full sequence
    selected_layers: List[int] = field(default_factory=lambda: [18])
    token_state: TokenSelectionState = field(default_factory=TokenSelectionState)


def is_xml_tag(token_text: str) -> bool:
    """Check if token is XML tag"""
    return token_text.startswith('<') and token_text.endswith('>')


def render_token(token_text: str, is_selected: bool, is_highlighted: bool) -> Text:
    """Render token with appropriate styling"""
    if is_xml_tag(token_text):
        return Text(token_text, style="dim")
    
    style = ""
    if is_selected:
        style += "bold yellow"
    if is_highlighted:
        style += " underline magenta"
    
    return Text(token_text, style=style)


def load_model_with_progress(console: Console) -> AppState:
    """Load model with rich progress bar"""
    state = AppState()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Loading Qwen 4B model...", total=100)
        
        progress.update(task, advance=10, description="Loading base model...")
        state.model, state.tokenizer, state.device = load_model_and_tokenizer(
            MODEL_NAME,
            use_8bit=False
        )
        
        # Validate model state before adapter loading
        console.print("\n[info]Validating model state...[/info]")
        
        # Check if device is GPU
        if state.device.type not in ["cuda", "mps"]:
            console.print(f"[yellow]Warning: Model on {state.device}. Adapter loading may be slow.[/yellow]")
        else:
            console.print(f"[green]✓ Model on {state.device}[/green]")
        
        # Check for meta device parameters (offloaded)
        has_meta_params = False
        for name, param in state.model.named_parameters():
            if param.device.type == 'meta':
                has_meta_params = True
                console.print(f"[yellow]Warning: Parameter '{name}' is on meta device (offloaded)[/yellow]")
        
        if has_meta_params:
            console.print("[yellow]Warning: Some parameters are offloaded. This may cause adapter loading issues on ROCm/AMD.[/yellow]")
            console.print("[info]Moving all parameters to GPU...[/info]")
            # Try to move model to device to resolve offloading
            state.model = state.model.to(state.device)
            console.print("[green]✓ All parameters moved to GPU[/green]")
        else:
            console.print("[green]✓ All parameters on device[/green]")
        
        # Clear any existing cache before adapter loading
        if state.device.type == "cuda":
            torch.cuda.empty_cache()
        elif state.device.type == "mps":
            torch.mps.empty_cache()
        console.print("[green]✓ GPU cache cleared[/green]")
        
        progress.update(task, advance=30, description="Loading oracle adapter...")
        oracle_path = get_oracle_checkpoint(MODEL_NAME)
        try:
            state.model, state.oracle_adapter_name = load_oracle_adapter(state.model, oracle_path)
        except RuntimeError as e:
            if "HIP error" in str(e) or "invalid device function" in str(e):
                console.print(f"\n[red]ROCm/AMD GPU error during adapter loading: {e}[/red]")
                console.print("[yellow]This is a known issue with PEFT on ROCm. Trying alternative approach...[/yellow]")
                
                # Try loading adapter with torch.no_grad() to avoid device operations
                console.print("[info]Retrying with torch.no_grad()...[/info]")
                with torch.no_grad():
                    state.model, state.oracle_adapter_name = load_oracle_adapter(state.model, oracle_path)
                console.print("[green]✓ Adapter loaded successfully![/green]")
            else:
                raise
        
        progress.update(task, advance=30, description="Initializing activation engine...")
        state.activation_engine = ActivationEngine(
            state.model,
            state.tokenizer,
            state.device
        )
        
        progress.update(task, advance=20, description="Initializing oracle interface...")
        state.oracle_interface = OracleInterface(
            state.model,
            state.tokenizer,
            state.oracle_adapter_name,
            state.device
        )
        
        progress.update(task, advance=10, description="Complete!")
    
    console.print("\n[green]✓ Model loaded successfully![/green]")
    console.print(f"  Model: {MODEL_NAME}")
    console.print(f"  Device: {state.device}")
    console.print(f"  Oracle: {state.oracle_adapter_name}")
    console.print()
    
    return state


def select_task(console: Console, tasks: List[Dict]) -> int:
    """Display 5 tasks, return selected index (0-4)"""
    table = Table(show_header=False, title="\nSELECT A TASK")
    table.add_column("Num", style="cyan", width=4)
    table.add_column("Type", style="green", width=12)
    table.add_column("Description", style="white")
    
    for i, task in enumerate(tasks):
        task_type = task.get("type", "unknown").upper()
        problem = task.get("problem", "")
        num_personas = task.get("num_personas", 3)
        desc = f"({num_personas} personas) {problem[:70]}{'...' if len(problem) > 70 else ''}"
        table.add_row(str(i + 1), task_type, desc)
    
    console.print(table)
    
    while True:
        try:
            choice = input("\nEnter choice [1-5]: ").strip()
            task_idx = int(choice) - 1
            if 0 <= task_idx < len(tasks):
                return task_idx
            console.print(f"[red]Invalid choice. Please enter 1-{len(tasks)}.[/red]")
        except (ValueError, KeyboardInterrupt):
            console.print("[red]Invalid input. Please enter a number.[/red]")


def display_response(console: Console, response_text: str) -> None:
    """Display the full response in a scrollable panel"""
    panel = Panel(
        response_text,
        title="[bold]GENERATED RESPONSE[/bold]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)


def get_tokens_by_line(response_text: str, tokenizer) -> List[List[Tuple[int, str]]]:
    """Split response into lines, each with list of (token_idx, token_text)"""
    full_tokens = decode_token_by_token(tokenizer, tokenizer.encode(response_text))
    
    lines = response_text.split('\n')
    tokens_by_line = []
    
    token_idx = 0
    for line in lines:
        line_tokens = []
        for char in line:
            while token_idx < len(full_tokens) and char not in full_tokens[token_idx]:
                token_idx += 1
            if token_idx < len(full_tokens):
                line_tokens.append((token_idx, full_tokens[token_idx]))
        tokens_by_line.append(line_tokens)
    
    return tokens_by_line


def select_tokens_interactive(
    console: Console,
    response_text: str,
    token_ids: List[int],
    tokenizer,
    scroll_pos: int = 0,
    selected_positions: Set[int] = None,
    highlighted_token_idx: int = 0,
    selected_layers: List[int] = None
) -> Tuple[Set[int], str, int, int]:
    """
    Interactive token browser with visual styling.

    Returns:
        (selected_positions, action, scroll_pos, highlighted_token_idx)
        action: 'chat', 'new_task', 'quit'
    """
    if selected_positions is None:
        selected_positions = set()
    if selected_layers is None:
        selected_layers = [18]

    token_texts = decode_token_by_token(tokenizer, token_ids)
    total_tokens = len(token_ids)

    if total_tokens == 0:
        console.print("[red]No tokens to display![/red]")
        return selected_positions, 'quit', scroll_pos, highlighted_token_idx

    # Enter raw mode once for entire session
    old_terminal = setup_terminal_raw()

    try:
        while True:
            # Clear screen using ANSI codes (works in raw mode)
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

            # Build output as string (Rich console doesn't work well in raw mode)
            lines = []
            lines.append("\033[1mSELECT TOKENS TO CHAT WITH\033[0m")

            # Layer toggle display
            layer_line = "Layers: "
            for i, layer in enumerate(ORACLE_LAYERS):
                key_num = i + 1
                is_sel = layer in selected_layers
                checkbox = "[X]" if is_sel else "[ ]"
                if is_sel:
                    layer_line += f"\033[36m[{key_num}]\033[32;1m{checkbox}L{layer}\033[0m "
                else:
                    layer_line += f"\033[36m[{key_num}]\033[2m{checkbox}L{layer}\033[0m "
            lines.append(layer_line)

            lines.append("\033[2m←→: Nav  ↑↓: Page  Space: Select  Q: Chat  C: Clear  N: New  ESC: Quit\033[0m")
            lines.append("")

            # Calculate visible window
            window_size = 40  # Show more tokens
            half_window = window_size // 2

            start_idx = max(0, highlighted_token_idx - half_window)
            end_idx = min(total_tokens, start_idx + window_size)

            # Adjust if near end
            if end_idx == total_tokens:
                start_idx = max(0, total_tokens - window_size)

            # Render tokens
            current_line = ""
            max_width = 100

            for idx in range(start_idx, end_idx):
                token = token_texts[idx]
                is_sel = idx in selected_positions
                is_hl = idx == highlighted_token_idx

                # Escape special chars for display
                display_token = token.replace('\n', '↵').replace('\t', '→').replace('\r', '')

                # Build styled token
                if is_hl and is_sel:
                    styled = f"\033[43;35;1;4m{display_token}\033[0m"  # Yellow bg, magenta, bold, underline
                elif is_hl:
                    styled = f"\033[45;1;4m{display_token}\033[0m"  # Magenta bg, bold, underline
                elif is_sel:
                    styled = f"\033[33;1m{display_token}\033[0m"  # Yellow bold
                elif is_xml_tag(token):
                    styled = f"\033[2m{display_token}\033[0m"  # Dim
                else:
                    styled = display_token

                # Check line length (approximate, ignoring ANSI codes)
                plain_len = len(current_line.replace('\033[0m', '').replace('\033[', ''))
                if plain_len + len(display_token) > max_width:
                    lines.append(current_line)
                    current_line = styled
                else:
                    current_line += styled

            if current_line:
                lines.append(current_line)

            lines.append("")

            # Status bar
            current_token_display = token_texts[highlighted_token_idx].replace('\n', '↵')[:20]
            status = f"\033[2mSelected:\033[0m \033[33;1m{len(selected_positions)}\033[0m"
            status += f" \033[2m| Token:\033[0m \033[36;1m{highlighted_token_idx + 1}/{total_tokens}\033[0m"
            status += f" \033[2m| Current:\033[0m \033[35;1m'{current_token_display}'\033[0m"
            lines.append(status)

            # Print all lines
            output = '\r\n'.join(lines)
            sys.stdout.write(output)
            sys.stdout.flush()

            # Read key
            try:
                key = read_key_raw()

                if key == KEY_ESCAPE:
                    return selected_positions, 'quit', scroll_pos, highlighted_token_idx
                elif key in ('q', 'Q'):
                    if selected_positions:
                        return selected_positions, 'chat', scroll_pos, highlighted_token_idx
                    # Flash message - no tokens selected
                    sys.stdout.write('\r\n\033[31mNo tokens selected! Press Space to select.\033[0m')
                    sys.stdout.flush()
                    import time
                    time.sleep(0.8)
                elif key in ('c', 'C'):
                    selected_positions.clear()
                elif key in ('n', 'N'):
                    return selected_positions, 'new_task', scroll_pos, highlighted_token_idx
                elif key == '1':
                    layer = ORACLE_LAYERS[0]
                    if layer in selected_layers:
                        selected_layers.remove(layer)
                    else:
                        selected_layers.append(layer)
                elif key == '2':
                    layer = ORACLE_LAYERS[1]
                    if layer in selected_layers:
                        selected_layers.remove(layer)
                    else:
                        selected_layers.append(layer)
                elif key == '3':
                    layer = ORACLE_LAYERS[2]
                    if layer in selected_layers:
                        selected_layers.remove(layer)
                    else:
                        selected_layers.append(layer)
                elif key == KEY_SPACE:
                    if highlighted_token_idx in selected_positions:
                        selected_positions.remove(highlighted_token_idx)
                    else:
                        selected_positions.add(highlighted_token_idx)
                elif key == KEY_UP:
                    highlighted_token_idx = max(0, highlighted_token_idx - 20)
                elif key == KEY_DOWN:
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 20)
                elif key == KEY_RIGHT:
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)
                elif key == KEY_LEFT:
                    highlighted_token_idx = max(0, highlighted_token_idx - 1)
                # Vim-style
                elif key in ('j',):
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)
                elif key in ('k',):
                    highlighted_token_idx = max(0, highlighted_token_idx - 1)
                elif key in ('h',):
                    highlighted_token_idx = max(0, highlighted_token_idx - 10)
                elif key in ('l',):
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 10)
                # Jump to start/end
                elif key in ('g',):
                    highlighted_token_idx = 0
                elif key in ('G',):
                    highlighted_token_idx = total_tokens - 1

            except KeyboardInterrupt:
                return selected_positions, 'quit', scroll_pos, highlighted_token_idx

    finally:
        # Always restore terminal
        restore_terminal(old_terminal)
        # Clear screen after exit
        console.clear()


def capture_activations_progress(
    console: Console,
    positions: List[int],
    model, tokenizer, device,
    token_ids: List[int],
    layer: int
):
    """Capture activations with progress display"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Capturing activations at layer {layer}...", total=100)
        
        trace = ConversationTrace()
        trace.token_ids = token_ids
        trace.formatted_prompt = ""
        
        activation_engine = ActivationEngine(model, tokenizer, device)
        activations = activation_engine.capture_activations(
            trace,
            positions,
            layer,
            use_cache=True
        )
        
        progress.update(task, completed=100)
    
    console.print(f"\n[green]✓ Captured {len(activations)} activations from layer {layer}[/green]")
    return activations


def oracle_chat_ui(
    console: Console,
    activations_list: List[List],
    model, tokenizer, device,
    oracle_adapter_name: str,
    layers: List[int]
) -> str:
    """Chat with oracle, return action: 'tokens', 'new_task', 'quit'"""
    oracle = OracleInterface(model, tokenizer, oracle_adapter_name, device)

    console.print("\n[bold]ORACLE CHAT[/bold]")
    console.print("[dim]Ask questions about selected tokens.[/dim]")
    console.print()
    console.print("[cyan]Preset questions:[/cyan]")
    console.print("  [bold]1[/bold] What is the model thinking?")
    console.print("  [bold]2[/bold] What word comes next?")
    console.print("  [bold]3[/bold] How confident is the model?")
    console.print("  [bold]4[/bold] What alternatives considered?")
    console.print("  [bold]5[/bold] Enter custom question")
    console.print()
    console.print("[dim]Commands: T=Back to Tokens, N=New Task, Q=Quit[/dim]\n")

    conversation = []

    for layer, activations in zip(layers, activations_list):
        console.print(f"\n[cyan]{'─' * 20} Layer {layer} {'─' * 20}[/cyan]")

        while True:
            try:
                user_input = input("\n[You]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return 'quit'

            if not user_input:
                continue

            lower_input = user_input.lower()

            if lower_input == 't':
                return 'tokens'
            elif lower_input == 'n':
                return 'new_task'
            elif lower_input == 'q':
                return 'quit'
            elif user_input == '1':
                question = PRESET_QUESTIONS.get('reasoning', 'What is the model thinking?')
                console.print(f"[dim]Question: {question}[/dim]")
            elif user_input == '2':
                question = PRESET_QUESTIONS.get('what_next', 'What word comes next?')
                console.print(f"[dim]Question: {question}[/dim]")
            elif user_input == '3':
                question = PRESET_QUESTIONS.get('confidence', 'How confident is the model?')
                console.print(f"[dim]Question: {question}[/dim]")
            elif user_input == '4':
                question = PRESET_QUESTIONS.get('alternatives', 'What alternatives considered?')
                console.print(f"[dim]Question: {question}[/dim]")
            elif user_input == '5':
                try:
                    question = input("Enter your question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return 'quit'
                if not question:
                    continue
            else:
                # Treat any other input as a direct question
                question = user_input

            try:
                console.print("\n[bold green]Oracle:[/bold green] ", end="")
                response = oracle.query_oracle(activations, question)
                console.print(response)
                conversation.append((question, response))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    return 'tokens'


def run_task(console: Console, task: Dict, model, tokenizer, device) -> Tuple[str, List[int], List[int], int]:
    """Generate multi-persona response with streaming display"""
    problem = task.get("problem", "")

    prompt = f"""You are simulating a collaborative group of thinkers solving a problem.

TASK TYPE: {task.get("type", "general")}
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

    messages = [{"role": "user", "content": prompt}]

    # Initialize streaming display
    full_response = ""
    token_count = 0

    # Create initial panel with empty content
    panel = Panel(
        "",
        title="[bold]GENERATING RESPONSE...[/bold]",
        border_style="cyan",
        padding=(1, 2),
        subtitle="[dim]Streaming tokens as generated[/dim]"
    )
    console.print(panel)

    # Stream generation - tokens are yielded in real-time as model generates
    for token_text, token_id, text_so_far in generate_response_stream(
        model, tokenizer, messages, device=device,
        generation_kwargs={"max_new_tokens": 16384}
    ):
        full_response = text_so_far
        token_count += 1

        # Update panel every 5 tokens
        if token_count % 5 == 0:
            updated_panel = Panel(
                full_response,
                title="[bold]GENERATING RESPONSE...[/bold]",
                border_style="cyan",
                padding=(1, 2),
                subtitle=f"[dim]Streaming: {token_count} tokens[/dim]"
            )
            console.clear()
            console.print(updated_panel)

    # Get token IDs for response only (for browser display)
    response_token_ids = tokenizer.encode(full_response, add_special_tokens=False)

    # Get full token IDs (prompt + response) for activation capture
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    full_token_ids = prompt_token_ids + response_token_ids
    prompt_length = len(prompt_token_ids)

    # Final update with completion message
    final_panel = Panel(
        full_response,
        title="[bold green]GENERATED RESPONSE[/bold green]",
        border_style="green",
        padding=(1, 2),
        subtitle=f"[green]✓ Complete: {len(response_token_ids)} tokens[/green]"
    )
    console.clear()
    console.print(final_panel)

    return full_response, response_token_ids, full_token_ids, prompt_length


def cleanup_model(state: AppState) -> None:
    """Clean up model and clear GPU cache"""
    try:
        if hasattr(state, 'activation_engine') and state.activation_engine is not None:
            del state.activation_engine
        if hasattr(state, 'oracle_interface') and state.oracle_interface is not None:
            del state.oracle_interface
        if hasattr(state, 'model') and state.model is not None:
            del state.model
        if hasattr(state, 'tokenizer') and state.tokenizer is not None:
            del state.tokenizer
        
        import gc
        gc.collect()
        
        if hasattr(state, 'device') and state.device is not None:
            if state.device.type == "cuda":
                torch.cuda.empty_cache()
            elif state.device.type == "mps":
                torch.mps.empty_cache()
    except Exception as e:
        pass


def main():
    """Main TUI application loop"""
    console = Console()
    
    console.print("\n[bold cyan]TUI Multi-Persona Solver with Token Oracle Chat[/bold cyan]\n")
    
    state = None
    try:
        state = load_model_with_progress(console)
        
        config = load_scenario_yaml(Path(TASKS_FILE))
        tasks = config.get("tasks", [])
        
        while True:
            task_idx = select_task(console, tasks)
            task = tasks[task_idx]

            console.print(f"\n[bold]Running task {task_idx + 1} of {len(tasks)}[/bold]")

            response, response_token_ids, full_token_ids, prompt_length = run_task(
                console, task, state.model, state.tokenizer, state.device
            )

            console.print(f"\n[green]✓ Generation complete ({len(response_token_ids)} tokens)[/green]")

            state.current_response = response
            state.current_token_ids = response_token_ids
            state.full_token_ids = full_token_ids
            state.prompt_length = prompt_length

            state.token_state = TokenSelectionState()

            # Flush any buffered stdin from streaming phase
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

            while True:
                selected_positions, action, scroll_pos, highlighted_idx = select_tokens_interactive(
                    console,
                    response,
                    response_token_ids,
                    state.tokenizer,
                    state.token_state.scroll_position,
                    state.token_state.selected_positions,
                    state.token_state.highlighted_token_idx,
                    state.selected_layers
                )
                
                state.token_state.scroll_position = scroll_pos
                state.token_state.selected_positions = selected_positions
                state.token_state.highlighted_token_idx = highlighted_idx
                
                if action == 'quit':
                    console.print("\n[yellow]Goodbye![/yellow]\n")
                    return
                elif action == 'new_task':
                    break
                elif action == 'chat':
                    # Map browser positions (response-relative) to full sequence positions
                    browser_positions = sorted(list(selected_positions))
                    full_positions = [pos + state.prompt_length for pos in browser_positions]

                    activations_by_layer = []
                    for layer in state.selected_layers:
                        activations = capture_activations_progress(
                            console,
                            full_positions,
                            state.model,
                            state.tokenizer,
                            state.device,
                            state.full_token_ids,
                            layer
                        )
                        activation_list = [activations[pos] for pos in full_positions]
                        activations_by_layer.append(activation_list)
                    
                    next_action = oracle_chat_ui(
                        console,
                        activations_by_layer,
                        state.model,
                        state.tokenizer,
                        state.device,
                        state.oracle_adapter_name,
                        state.selected_layers
                    )
                    
                    if next_action == 'quit':
                        console.print("\n[yellow]Goodbye![/yellow]\n")
                        return
                    elif next_action == 'new_task':
                        break
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user. Cleaning up...[/yellow]")
    finally:
        if state is not None:
            cleanup_model(state)
            console.print("[green]✓ Model unloaded successfully[/green]\n")


if __name__ == "__main__":
    main()
