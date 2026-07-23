"""Mode system - control which tools are available based on task type."""

# Map mode names to tool name lists
MODE_TOOLS: dict[str, list[str]] = {
    "review": [
        "read_file",
        "grep",
        "glob",
    ],
    "coding": [
        "read_file",
        "grep",
        "glob",
        "edit_file",
        "write_file",
        "bash",
    ],
    "document": [
        "read_file",
        "grep",
        "glob",
        "edit_file",
        "write_file",
    ],
    "infra": [
        "read_file",
        "grep",
        "glob",
        "gpu_status",
        "vllm_status",
    ],
}

# Short descriptions for CLI and help display
MODE_DESCRIPTIONS: dict[str, str] = {
    "review": "Code review (read/search only)",
    "coding": "Full coding (read, write, edit, bash)",
    "document": "Text document editing (read, write, edit)",
    "infra": "AI infrastructure diagnostics (GPU, vLLM)",
}


def get_tools_for_mode(mode: str | None) -> list:
    """Return tool instances for the given mode.

    If mode is None, returns ALL_TOOLS (all tools available).
    """
    if mode is None:
        from .tools import ALL_TOOLS
        return list(ALL_TOOLS)

    mode = mode.lower()
    if mode not in MODE_TOOLS:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Available modes: {', '.join(sorted(MODE_TOOLS))}"
        )

    tool_names = MODE_TOOLS[mode]
    from .tools import get_tool
    tools = []
    for name in tool_names:
        t = get_tool(name)
        if t:
            tools.append(t)
    return tools


def list_modes() -> str:
    """Return a formatted list of available modes."""
    lines = ["Available modes:"]
    for name in sorted(MODE_TOOLS):
        desc = MODE_DESCRIPTIONS.get(name, "")
        tool_list = ", ".join(MODE_TOOLS[name])
        lines.append(f"  [cyan]{name}[/cyan]: {desc}")
        lines.append(f"         Tools: {tool_list}")
    return "\n".join(lines)
