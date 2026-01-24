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
import shutil

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
    # Chat persistence
    chat_history: List[Tuple[str, str]] = field(default_factory=list)  # (role, message) tuples
    cached_activations: List[List] = field(default_factory=list)  # Activations by layer
    cached_positions: List[int] = field(default_factory=list)  # Positions used for cached activations
    cached_layers: List[int] = field(default_factory=list)  # Layers used for cached activations


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


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal width and height."""
    size = shutil.get_terminal_size((120, 40))
    return size.columns, size.lines


def draw_box(width: int, height: int, title: str = "", title_style: str = "\033[1;36m") -> Tuple[str, str, str]:
    """Draw a box frame. Returns (top_line, middle_format, bottom_line)."""
    # Box drawing characters
    TL, TR, BL, BR = '┌', '┐', '└', '┘'
    H, V = '─', '│'

    inner_width = width - 2

    if title:
        title_display = f" {title} "
        padding = inner_width - len(title_display)
        left_pad = padding // 2
        right_pad = padding - left_pad
        top = f"{TL}{H * left_pad}{title_style}{title_display}\033[0m{H * right_pad}{TR}"
    else:
        top = f"{TL}{H * inner_width}{TR}"

    bottom = f"{BL}{H * inner_width}{BR}"

    return top, V, bottom


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
    Fullscreen interactive token browser.

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
            # Get terminal size for fullscreen
            term_width, term_height = get_terminal_size()

            # Clear screen using ANSI codes (works in raw mode)
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

            # Calculate layout dimensions
            content_width = min(term_width - 4, 140)  # Max width with margins
            content_start = (term_width - content_width) // 2  # Center content

            # Header area: 6 lines (title, layers, help, separator, blank)
            # Footer area: 4 lines (separator, status, progress bar, blank)
            # Token area: remaining lines
            header_lines = 6
            footer_lines = 4
            token_area_height = term_height - header_lines - footer_lines

            lines = []

            # === HEADER ===
            # Title bar
            title = "TOKEN SELECTOR"
            title_bar = f"\033[1;36m{'═' * ((content_width - len(title) - 2) // 2)} {title} {'═' * ((content_width - len(title) - 2) // 2)}\033[0m"
            lines.append(title_bar.center(term_width))

            # Layer toggle display
            layer_line = "\033[1mLayers:\033[0m "
            for i, layer in enumerate(ORACLE_LAYERS):
                key_num = i + 1
                is_sel = layer in selected_layers
                if is_sel:
                    layer_line += f" \033[36m[{key_num}]\033[0m\033[42;30;1m L{layer} \033[0m"
                else:
                    layer_line += f" \033[36m[{key_num}]\033[0m\033[2m L{layer} \033[0m"
            lines.append(layer_line.center(term_width + 40))  # Extra for ANSI codes

            # Help line
            help_text = "\033[2m←→/hl: Nav │ ↑↓/jk: Jump │ Space: Select │ Q: Chat │ C: Clear │ N: New Task │ g/G: Start/End │ ESC: Quit\033[0m"
            lines.append(help_text.center(term_width + 20))

            # Separator
            lines.append(f"\033[2m{'─' * content_width}\033[0m".center(term_width))
            lines.append("")

            # === TOKEN AREA ===
            # Calculate how many tokens fit in the display area
            # Estimate: each line can hold ~content_width chars, tokens average ~5 chars
            estimated_tokens_per_line = content_width // 6
            tokens_visible = estimated_tokens_per_line * (token_area_height - 2)

            # Calculate window around highlighted token
            half_window = tokens_visible // 2
            start_idx = max(0, highlighted_token_idx - half_window)
            end_idx = min(total_tokens, start_idx + tokens_visible)

            # Adjust if near end
            if end_idx == total_tokens:
                start_idx = max(0, total_tokens - tokens_visible)

            # Build token display
            token_lines = []
            current_line = ""
            current_plain_len = 0

            for idx in range(start_idx, end_idx):
                token = token_texts[idx]
                is_sel = idx in selected_positions
                is_hl = idx == highlighted_token_idx

                # Escape special chars for display
                display_token = token.replace('\n', '↵').replace('\t', '→').replace('\r', '').replace('\x0b', '').replace('\x0c', '')
                token_len = len(display_token)

                # Build styled token
                if is_hl and is_sel:
                    styled = f"\033[43;35;1m {display_token} \033[0m"  # Yellow bg, magenta text
                elif is_hl:
                    styled = f"\033[45;37;1m {display_token} \033[0m"  # Magenta bg, white text
                elif is_sel:
                    styled = f"\033[33;1m{display_token}\033[0m"  # Yellow bold
                elif is_xml_tag(token):
                    styled = f"\033[2;36m{display_token}\033[0m"  # Dim cyan
                else:
                    styled = display_token

                # Check if we need to wrap to next line
                if current_plain_len + token_len + 1 > content_width - 4:
                    token_lines.append(current_line)
                    current_line = styled
                    current_plain_len = token_len
                else:
                    current_line += styled
                    current_plain_len += token_len

            if current_line:
                token_lines.append(current_line)

            # Pad token area to fixed height
            while len(token_lines) < token_area_height:
                token_lines.append("")

            # Add token lines with left margin
            margin = " " * max(2, (term_width - content_width) // 2)
            for i, line in enumerate(token_lines[:token_area_height]):
                lines.append(f"{margin}{line}")

            # === FOOTER ===
            lines.append("")
            lines.append(f"\033[2m{'─' * content_width}\033[0m".center(term_width))

            # Status bar
            current_token_display = token_texts[highlighted_token_idx].replace('\n', '↵').replace('\r', '')[:30]
            status = f"\033[1mSelected:\033[0m \033[33;1m{len(selected_positions)}\033[0m"
            status += f"  \033[2m│\033[0m  \033[1mToken:\033[0m \033[36;1m{highlighted_token_idx + 1}\033[0m\033[2m/\033[0m{total_tokens}"
            status += f"  \033[2m│\033[0m  \033[1mCurrent:\033[0m \033[35;1m'{current_token_display}'\033[0m"
            lines.append(status.center(term_width + 60))

            # Progress bar
            progress_width = min(60, content_width - 20)
            progress_pos = int((highlighted_token_idx / max(1, total_tokens - 1)) * progress_width)
            progress_bar = f"\033[2m[\033[0m\033[36m{'█' * progress_pos}\033[2m{'░' * (progress_width - progress_pos)}\033[0m\033[2m]\033[0m"
            lines.append(progress_bar.center(term_width + 20))

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
                    sys.stdout.write('\r\n\033[31;1m  ⚠ No tokens selected! Press Space on a token to select it.\033[0m')
                    sys.stdout.flush()
                    import time
                    time.sleep(1.0)
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
                    # Jump by roughly one line worth of tokens
                    highlighted_token_idx = max(0, highlighted_token_idx - estimated_tokens_per_line)
                elif key == KEY_DOWN:
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + estimated_tokens_per_line)
                elif key == KEY_RIGHT:
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)
                elif key == KEY_LEFT:
                    highlighted_token_idx = max(0, highlighted_token_idx - 1)
                # Vim-style navigation
                elif key == 'j':
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)
                elif key == 'k':
                    highlighted_token_idx = max(0, highlighted_token_idx - 1)
                elif key == 'h':
                    highlighted_token_idx = max(0, highlighted_token_idx - 1)
                elif key == 'l':
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)
                # Page up/down with ctrl or larger jumps
                elif key == 'J':
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + estimated_tokens_per_line * 3)
                elif key == 'K':
                    highlighted_token_idx = max(0, highlighted_token_idx - estimated_tokens_per_line * 3)
                # Jump to start/end
                elif key == 'g':
                    highlighted_token_idx = 0
                elif key == 'G':
                    highlighted_token_idx = total_tokens - 1
                # Select range (shift+space or 's')
                elif key == 's':
                    # Toggle selection and move forward
                    if highlighted_token_idx in selected_positions:
                        selected_positions.remove(highlighted_token_idx)
                    else:
                        selected_positions.add(highlighted_token_idx)
                    highlighted_token_idx = min(total_tokens - 1, highlighted_token_idx + 1)

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
    layers: List[int],
    chat_history: List[Tuple[str, str]] = None
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Fullscreen chat with oracle.
    Commands use / prefix: /t (tokens), /n (new task), /q (quit), /c (clear chat), /help
    Numbers 1-4 select preset questions.
    Returns (action, chat_history): action is 'tokens', 'new_task', 'quit'
    """
    oracle = OracleInterface(model, tokenizer, oracle_adapter_name, device)
    term_width, term_height = get_terminal_size()
    content_width = min(term_width - 4, 120)

    conversation = chat_history if chat_history is not None else []
    current_layer_idx = 0
    scroll_offset = 0

    def render_chat_screen():
        """Render the fullscreen chat interface."""
        nonlocal scroll_offset
        console.clear()

        # === HEADER ===
        title = f"ORACLE CHAT - Layer {layers[current_layer_idx]}"
        header_bar = f"\033[1;32m{'═' * ((content_width - len(title) - 2) // 2)} {title} {'═' * ((content_width - len(title) - 2) // 2)}\033[0m"
        console.print(header_bar)

        # Layer tabs
        layer_tabs = ""
        for i, layer in enumerate(layers):
            if i == current_layer_idx:
                layer_tabs += f" \033[42;30;1m L{layer} \033[0m "
            else:
                layer_tabs += f" \033[2m L{layer} \033[0m "
        console.print(f"Layers:{layer_tabs}")
        console.print()

        # === CHAT HISTORY ===
        # Calculate space for chat history
        # Header: 3 lines, Footer: 6 lines (help, separator, presets, input)
        chat_height = term_height - 12

        # Build chat display
        chat_lines = []
        for role, msg in conversation:
            if role == 'user':
                prefix = "\033[1;33mYou:\033[0m "
                wrapped = wrap_text(msg, content_width - 8)
                for i, line in enumerate(wrapped):
                    if i == 0:
                        chat_lines.append(f"  {prefix}{line}")
                    else:
                        chat_lines.append(f"       {line}")
            else:  # oracle
                prefix = "\033[1;32mOracle:\033[0m "
                wrapped = wrap_text(msg, content_width - 10)
                for i, line in enumerate(wrapped):
                    if i == 0:
                        chat_lines.append(f"  {prefix}{line}")
                    else:
                        chat_lines.append(f"          {line}")
            chat_lines.append("")  # Spacing between messages

        # Handle scrolling
        if len(chat_lines) > chat_height:
            # Auto-scroll to bottom if not manually scrolled
            visible_lines = chat_lines[-(chat_height):]
        else:
            visible_lines = chat_lines
            # Pad to fill space
            while len(visible_lines) < chat_height:
                visible_lines.append("")

        # Draw chat box
        console.print(f"\033[2m{'─' * content_width}\033[0m")
        for line in visible_lines:
            console.print(line)
        console.print(f"\033[2m{'─' * content_width}\033[0m")

        # === FOOTER ===
        # Preset questions
        console.print("\033[2mPresets:\033[0m \033[36m[1]\033[0m Thinking  \033[36m[2]\033[0m Next word  \033[36m[3]\033[0m Confidence  \033[36m[4]\033[0m Alternatives")

        # Commands help
        console.print("\033[2mCommands:\033[0m /t=Tokens  /n=New Task  /c=Clear Chat  /q=Quit  /layer N=Switch Layer  /help")
        console.print()

    def wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        if not text:
            return [""]
        lines = []
        for paragraph in text.split('\n'):
            while len(paragraph) > width:
                # Find last space before width
                split_idx = paragraph.rfind(' ', 0, width)
                if split_idx == -1:
                    split_idx = width
                lines.append(paragraph[:split_idx])
                paragraph = paragraph[split_idx:].lstrip()
            lines.append(paragraph)
        return lines if lines else [""]

    # Main chat loop
    while True:
        render_chat_screen()

        try:
            # Use regular input for chat (not raw mode)
            user_input = input("\033[1;33m[You]:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            return ('quit', conversation)

        if not user_input:
            continue

        # === COMMAND HANDLING ===
        if user_input.startswith('/'):
            cmd = user_input[1:].lower().split()
            cmd_name = cmd[0] if cmd else ''
            cmd_args = cmd[1:] if len(cmd) > 1 else []

            if cmd_name in ('t', 'tokens', 'back'):
                return ('tokens', conversation)
            elif cmd_name in ('n', 'new', 'newtask'):
                return ('new_task', conversation)
            elif cmd_name in ('q', 'quit', 'exit'):
                return ('quit', conversation)
            elif cmd_name in ('c', 'clear', 'reset', 'newchat'):
                conversation.clear()
                console.print("\033[33m✓ Chat history cleared\033[0m")
                import time
                time.sleep(0.5)
                continue
            elif cmd_name == 'layer' and cmd_args:
                try:
                    layer_num = int(cmd_args[0])
                    if layer_num in layers:
                        current_layer_idx = layers.index(layer_num)
                        console.print(f"\033[33m✓ Switched to Layer {layer_num}\033[0m")
                    else:
                        console.print(f"\033[31mLayer {layer_num} not available. Options: {layers}\033[0m")
                    import time
                    time.sleep(0.5)
                except ValueError:
                    console.print("\033[31mUsage: /layer <number>\033[0m")
                continue
            elif cmd_name == 'help':
                console.clear()
                console.print("\n\033[1;36m=== ORACLE CHAT HELP ===\033[0m\n")
                console.print("\033[1mCommands (start with /):\033[0m")
                console.print("  /t, /tokens, /back  - Return to token selector (keeps chat)")
                console.print("  /n, /new, /newtask  - Start a new task")
                console.print("  /c, /clear          - Clear chat history (keeps tokens)")
                console.print("  /q, /quit, /exit    - Quit the application")
                console.print("  /layer <N>          - Switch to layer N")
                console.print("  /help               - Show this help")
                console.print()
                console.print("\033[1mPreset Questions:\033[0m")
                console.print("  1 - What is the model thinking?")
                console.print("  2 - What word comes next?")
                console.print("  3 - How confident is the model?")
                console.print("  4 - What alternatives are being considered?")
                console.print()
                console.print("\033[1mFree-form Questions:\033[0m")
                console.print("  Just type any question to ask the oracle about the selected tokens.")
                console.print()
                input("\033[2mPress Enter to continue...\033[0m")
                continue
            else:
                console.print(f"\033[31mUnknown command: /{cmd_name}. Type /help for available commands.\033[0m")
                import time
                time.sleep(1)
                continue

        # === PRESET QUESTION HANDLING ===
        if user_input == '1':
            question = PRESET_QUESTIONS.get('reasoning', 'What is the model thinking?')
        elif user_input == '2':
            question = PRESET_QUESTIONS.get('what_next', 'What word comes next?')
        elif user_input == '3':
            question = PRESET_QUESTIONS.get('confidence', 'How confident is the model?')
        elif user_input == '4':
            question = PRESET_QUESTIONS.get('alternatives', 'What alternatives considered?')
        else:
            # Free-form question
            question = user_input

        # Add user message to conversation
        conversation.append(('user', question))

        # Query oracle
        try:
            console.print("\n\033[2mOracle is thinking...\033[0m")
            activations = activations_list[current_layer_idx]
            response = oracle.query_oracle(activations, question)
            conversation.append(('oracle', response))
        except Exception as e:
            error_msg = f"Error: {e}"
            conversation.append(('oracle', error_msg))

    return ('tokens', conversation)


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

    # Get task info for display
    task_type = task.get("type", "general").upper()
    num_personas = task.get("num_personas", 3)

    # Build task info header
    task_info = Text()
    task_info.append("Task: ", style="bold cyan")
    task_info.append(f"{task_type}", style="bold yellow")
    task_info.append(f" ({num_personas} personas)\n", style="dim")
    task_info.append("Problem: ", style="bold cyan")
    task_info.append(f"{problem}\n", style="white")
    task_info.append("─" * 60 + "\n", style="dim")

    # Initialize streaming display
    full_response = ""
    token_count = 0

    # Create initial panel with task info
    panel = Panel(
        task_info,
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
            # Combine task info with response
            display_content = Text()
            display_content.append("Task: ", style="bold cyan")
            display_content.append(f"{task_type}", style="bold yellow")
            display_content.append(f" ({num_personas} personas)\n", style="dim")
            display_content.append("Problem: ", style="bold cyan")
            display_content.append(f"{problem}\n", style="white")
            display_content.append("─" * 60 + "\n\n", style="dim")
            display_content.append(full_response)

            updated_panel = Panel(
                display_content,
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

    # Final display with task info
    final_content = Text()
    final_content.append("Task: ", style="bold cyan")
    final_content.append(f"{task_type}", style="bold yellow")
    final_content.append(f" ({num_personas} personas)\n", style="dim")
    final_content.append("Problem: ", style="bold cyan")
    final_content.append(f"{problem}\n", style="white")
    final_content.append("─" * 60 + "\n\n", style="dim")
    final_content.append(full_response)

    final_panel = Panel(
        final_content,
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

            # Reset state for new task
            state.token_state = TokenSelectionState()
            state.chat_history = []
            state.cached_activations = []
            state.cached_positions = []
            state.cached_layers = []

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

                    # Check if we need to re-capture activations
                    positions_changed = full_positions != state.cached_positions
                    layers_changed = sorted(state.selected_layers) != sorted(state.cached_layers)

                    if positions_changed or layers_changed or not state.cached_activations:
                        # Clear chat if selection changed (user chose different tokens/layers)
                        if (positions_changed or layers_changed) and state.chat_history:
                            console.print("[yellow]Token/layer selection changed - clearing chat history[/yellow]")
                            state.chat_history = []

                        # Capture new activations
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

                        # Cache for next time
                        state.cached_activations = activations_by_layer
                        state.cached_positions = full_positions
                        state.cached_layers = list(state.selected_layers)
                    else:
                        console.print("[green]Using cached activations[/green]")
                        activations_by_layer = state.cached_activations

                    next_action, state.chat_history = oracle_chat_ui(
                        console,
                        activations_by_layer,
                        state.model,
                        state.tokenizer,
                        state.device,
                        state.oracle_adapter_name,
                        state.selected_layers,
                        state.chat_history
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
