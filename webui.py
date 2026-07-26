"""
InfraCoder Web UI -- Gradio frontend (Gradio 6.x).
==================================================
Runs on the server.  Department colleagues access via:
    http://192.168.15.119:7860

Start:
    cd /home/ubuntu/XYP/InfraCoder
    source venv/bin/activate
    nohup python3 webui.py > webui.log 2>&1 &
"""

import os
import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import gradio as gr
from infracoder.agent import Agent
from infracoder.llm import LLM
from infracoder.config import Config
from infracoder.tools import ALL_TOOLS


# -- defaults (overridable via .env) ---------------------------------
_MODEL = os.environ.get("INFRACODER_MODEL", "gemma-4-31b-it")
_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("INFRACODER_BASE_URL")
    or "http://127.0.0.1:8000/v1"
)
_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("INFRACODER_API_KEY")
    or "EMPTY"
)

MODE_OPTIONS = [
    ("full -- all tools", "full"),
    ("coding -- code editing", "coding"),
    ("document -- document editing", "document"),
    ("infra -- infra diagnostics", "infra"),
    ("review -- code review", "review"),
]


def _make_agent() -> Agent:
    cfg = Config.from_env()
    if not cfg.api_key:
        cfg.api_key = _API_KEY
    if not cfg.base_url:
        cfg.base_url = _BASE_URL
    if not cfg.model or cfg.model == "gpt-5.5":
        cfg.model = _MODEL
    llm = LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return Agent(llm=llm, max_context_tokens=cfg.max_context_tokens)


def _set_tool_mode(agent: Agent, mode: str) -> None:
    agent.set_mode(mode)


def _chat_impl(message: str, history: list, agent_state, mode: str):
    if agent_state is None:
        agent_state = _make_agent()
    try:
        _set_tool_mode(agent_state, mode)
    except ValueError as e:
        return str(e), agent_state

    tool_log: list[str] = []

    def on_tool(name, kwargs):
        brief = {k: v for k, v in kwargs.items()
                 if not isinstance(v, str) or len(v) < 100}
        tool_log.append(
            "\U0001f527 **{}**\n```json\n{}\n```".format(
                name, json.dumps(brief, ensure_ascii=False, indent=2)
            )
        )

    try:
        result = agent_state.chat(message, on_tool=on_tool)
        parts = tool_log + ["---\n\n" + result] if tool_log else [result]
        return "\n\n".join(parts), agent_state
    except Exception as e:
        return "\u274c Error: {}".format(e), agent_state


def _system_status() -> str:
    import subprocess, urllib.request
    lines = ["## System Status"]
    try:
        req = urllib.request.Request(
            _BASE_URL.rstrip("/") + "/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["id"] for m in data.get("data", [])]
        lines.append("- vLLM: \u2705  |  {}".format(_BASE_URL))
        lines.append("- Model(s): " + ", ".join(models))
    except Exception as e:
        lines.append("- vLLM: \u274c ({})".format(e))
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) >= 4:
                lines.append("- GPU {}: {} | {} / {} MiB | {}*C".format(*parts[:5]))
    except Exception:
        lines.append("- GPU: nvidia-smi unavailable")
    return "\n".join(lines)


def create_ui():
    with gr.Blocks(title="InfraCoder") as demo:
        agent_state = gr.State(None)

        gr.Markdown(
            "# \U0001f680 InfraCoder\n"
            "OpenAI-Compatible API powered AI assistant for code, "
            "documents, and infra diagnostics.  Each session gets its "
            "own Agent instance."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mode_dd = gr.Dropdown(
                    choices=MODE_OPTIONS,
                    value="full",
                    label="Tool mode",
                    info="Which tool set to enable",
                )
            with gr.Column(scale=2):
                status_md = gr.Markdown(_system_status())
                with gr.Row():
                    gr.Button("\U0001f504 Refresh").click(
                        fn=_system_status, outputs=status_md
                    )

            chatbot = gr.Chatbot(
            label="Chat",
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="Input",
                placeholder="Type your message and press Enter...",
                scale=8,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=100)

        gr.Button("\U0001f5d1 Clear").click(
            fn=lambda: ([], None),
            outputs=[chatbot, agent_state],
        )

        def _respond(message, history, agent, mode):
            if not message or not message.strip():
                return "", history, agent
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "\u23f3 Running..."})
            reply, agent = _chat_impl(message, history[:-2], agent, mode)
            history[-1] = {"role": "assistant", "content": reply}
            return "", history, agent

        submit_args = dict(
            fn=_respond,
            inputs=[msg_input, chatbot, agent_state, mode_dd],
            outputs=[msg_input, chatbot, agent_state],
        )
        msg_input.submit(**submit_args)
        send_btn.click(**submit_args)

        demo.load(fn=_system_status, outputs=status_md)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    css_custom = """footer { display: none !important; }"""
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    print("=" * 56)
    print("  InfraCoder Web UI starting...")
    print("  -> http://0.0.0.0:7860")
    print("  (LAN: http://192.168.15.119:7860)")
    print("=" * 56)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None,
        theme=theme,
        css=css_custom,
    )
