"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .session import save_session, load_session, list_sessions
from . import __version__

console = Console()





def _kb_command(args):
    """Handle kb subcommands."""
    from .knowledge import KnowledgeBase
    kb = KnowledgeBase()
    if args.kb_cmd == "add":
        result = kb.add(args.path)
        print(result)
    elif args.kb_cmd == "list":
        docs = kb.list_documents()
        if not docs:
            print("Knowledge base is empty.")
        else:
            print(f"Knowledge base: {sum(d['chunks'] for d in docs)} chunks")
            for d in docs:
                print(f"  {d['source']}: {d['chunks']} chunks, {d['chars']} chars")
    elif args.kb_cmd == "remove":
        result = kb.remove(args.doc_id)
        print(result)
    elif args.kb_cmd == "rebuild":
        result = kb.rebuild()
        print(result)
    elif args.kb_cmd == "stats":
        print(kb.stats())

def _parse_args():
    p = argparse.ArgumentParser(
        prog="infracoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $INFRACODER_MODEL or gpt-5.5)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--mode", choices=["review","coding","document","infra"],
                   help="Tool mode (default: all tools)")
    # kb subcommand
    kb_p = subparsers = p.add_subparsers()
    kb_p = p.add_subparsers(title="kb", dest="kb_cmd")
    kb_p.add_parser("list", help="List indexed documents")
    kb_p.add_parser("stats", help="Show knowledge base statistics")
    kb_p.add_parser("rebuild", help="Rebuild index from scratch")
    
    kb_add = kb_p.add_parser("add", help="Add a file or directory to knowledge base")
    kb_add.add_argument("path", help="Path to file or directory")
    
    kb_rm = kb_p.add_parser("remove", help="Remove a document from knowledge base")
    kb_rm.add_argument("doc_id", help="Document ID to remove")
    
        # kb subcommand
    kb_p = p.add_subparsers(title="kb", dest="kb_cmd")
    kb_p.add_parser("list", help="List indexed documents")
    kb_p.add_parser("stats", help="Show knowledge base statistics")
    kb_p.add_parser("rebuild", help="Rebuild index from scratch")

    kb_add = kb_p.add_parser("add", help="Add a file or directory to knowledge base")
    kb_add.add_argument("path", help="Path to file or directory")

    kb_rm = kb_p.add_parser("remove", help="Remove a document from knowledge base")
    kb_rm.add_argument("doc_id", help="Document ID to remove")

    return p.parse_args()


def main():
    args = _parse_args()
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or INFRACODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 INFRACODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)
    if args.mode:
        agent.set_mode(args.mode)

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
        # kb subcommand
    if hasattr(args, "kb_cmd") and args.kb_cmd:
        _kb_command(args)
        return

    if args.prompt:

        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config, args.mode or "full")


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    try:
        agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    print()


def _repl(agent: Agent, config: Config, mode: str = "full"):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]InfraCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + f"\nMode: [cyan]{agent.tools[0].name if len(agent.tools)==1 else mode}[/cyan]" + "\nTools: " + ", ".join(t.name for t in agent.tools)
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.infracoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/mode" or user_input.startswith("/mode "):
            new_mode = user_input[6:].strip().lower() if user_input.startswith("/mode ") else ""
            if not new_mode:
                from .modes import list_modes
                print(list_modes())
                continue
            try:
                agent.set_mode(new_mode)
                console.print(f"Switched to [cyan]{new_mode}[/cyan] mode")
                console.print("Tools: " + ", ".join(t.name for t in agent.tools))
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: infracoder -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        if user_input == "/profile":
            from .user_config import init_user_config
            uc = init_user_config()
            console.print(f"[cyan]{uc.describe()}[/cyan]")
            continue
        if user_input.startswith("/kb "):
            parts = user_input[4:].strip().split(None, 1)
            if not parts:
                console.print("[yellow]Usage: /kb add <path>, /kb list, /kb remove <id>, /kb rebuild, /kb stats[/yellow]")
                continue
            cmd = parts[0]
            from .knowledge import KnowledgeBase
            kb = KnowledgeBase()
            if cmd == "add":
                if len(parts) < 2:
                    console.print("[yellow]Usage: /kb add <path>[/yellow]")
                    continue
                console.print(kb.add(parts[1]))
            elif cmd == "list":
                docs = kb.list_documents()
                if not docs:
                    console.print("[dim]Knowledge base is empty.[/dim]")
                else:
                    console.print(f"Knowledge base: [cyan]{sum(d['chunks'] for d in docs)}[/cyan] chunks")
                    for d in docs:
                        console.print(f"  [cyan]{d['source']}[/cyan]: {d['chunks']} chunks, {d['chars']} chars")
            elif cmd == "remove":
                if len(parts) < 2:
                    console.print("[yellow]Usage: /kb remove <doc_id>[/yellow]")
                    continue
                console.print(kb.remove(parts[1]))
            elif cmd == "rebuild":
                console.print(kb.rebuild())
            elif cmd == "stats":
                console.print(kb.stats())
            else:
                console.print(f"[yellow]Unknown kb command: {cmd}[/yellow]")
            continue


        # an unknown /command shouldn't be sent to the model as a prompt
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /mode         Show available modes\n"
        + "  /mode <name>   Switch to a different mode\n"
        + "  /diff          Show files modified this session\n"
        "  /profile       Show your personal configuration\n"
        "  /kb <cmd>      Knowledge base: add, list, remove, rebuild, stats\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit InfraCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="InfraCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
