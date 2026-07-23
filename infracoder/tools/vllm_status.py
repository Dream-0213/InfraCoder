"""Read-only vLLM service status inspection tool."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from .base import Tool


class VLLMStatusTool(Tool):
    """Inspect vLLM inference service status."""

    name = "vllm_status"
    description = (
        "Inspect vLLM inference service status on the current machine. "
        "Use this tool when the user asks about vLLM service health, "
        "model availability, API connectivity, or request latency. "
        "Checks /v1/models and /v1/chat/completions endpoints "
        "and reports response times and error states. "
        "This tool is read-only and does not modify the system."
    )

    parameters = {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": (
                    "vLLM service base URL. "
                    "Defaults to http://127.0.0.1:8000/v1"
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Request timeout in seconds. Defaults to 15 seconds."
                ),
                "minimum": 1,
                "maximum": 120,
            },
        },
        "required": [],
    }

    _MAX_OUTPUT_CHARS = 8_000

    def execute(
        self, base_url: str = "http://127.0.0.1:8000/v1", timeout: int = 15
    ) -> str:
        """Check vLLM service status and return a diagnostic report."""

        base_url = base_url.rstrip("/")

        if not isinstance(timeout, int):
            return "Error: timeout must be an integer"

        if timeout < 1 or timeout > 120:
            return "Error: timeout must be between 1 and 120 seconds"

        sections: list[str] = []

        sections.append(self._check_models_endpoint(base_url, timeout))
        sections.append(self._check_chat_endpoint(base_url, timeout))
        sections.append(self._check_infracoder_config())

        output = "\n\n".join(s for s in sections if s)

        return self._truncate_output(output)

    @staticmethod
    def _check_models_endpoint(base_url: str, timeout: int) -> str:
        """Check /v1/models endpoint."""

        url = f"{base_url}/models"
        start = time.time()

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - start
                body = json.loads(resp.read().decode("utf-8"))
                models_data = body.get("data", [])
                model_names = [
                    m.get("id", "unknown") for m in models_data
                ]

                return (
                    "## vLLM Models Endpoint\n\n"
                    f"- **URL:** {url}\n"
                    f"- **Status:** OK ({resp.status})\n"
                    f"- **Response time:** {elapsed:.2f}s\n"
                    f"- **Models:** "
                    f"{', '.join(model_names) if model_names else 'none'}"
                )
        except urllib.error.HTTPError as exc:
            elapsed = time.time() - start
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return (
                "## vLLM Models Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** HTTP {exc.code} ({exc.reason})\n"
                f"- **Response time:** {elapsed:.2f}s\n"
                f"- **Error:** {detail}"
            )
        except urllib.error.URLError as exc:
            elapsed = time.time() - start
            return (
                "## vLLM Models Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** CONNECTION FAILED\n"
                f"- **Response time:** {elapsed:.2f}s\n"
                f"- **Error:** {exc.reason}"
            )
        except Exception as exc:
            return (
                "## vLLM Models Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** ERROR\n"
                f"- **Error:** {str(exc)[:500]}"
            )

    @staticmethod
    def _check_chat_endpoint(base_url: str, timeout: int) -> str:
        """Check /v1/chat/completions endpoint with a lightweight request."""

        # First discover the model name
        model_name = ""
        try:
            req = urllib.request.Request(
                f"{base_url}/models", method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                models = body.get("data", [])
                if models:
                    model_name = models[0].get("id", "")
        except Exception:
            pass

        payload_dict = {
            "model": model_name or "default",
            "messages": [
                {"role": "user", "content": "Say exactly one word: hello"}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        payload = json.dumps(payload_dict).encode("utf-8")

        url = f"{base_url}/chat/completions"
        start = time.time()

        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - start
                body = json.loads(resp.read().decode("utf-8"))
                content = ""
                finish_reason = ""
                if "choices" in body and body["choices"]:
                    choice = body["choices"][0]
                    finish_reason = choice.get("finish_reason", "")
                    msg = choice.get("message", {})
                    if msg and "content" in msg:
                        content = (msg["content"] or "")[:200]

                return (
                    "## vLLM Chat Endpoint\n\n"
                    f"- **URL:** {url}\n"
                    f"- **Status:** OK (200)\n"
                    f"- **Response time:** {elapsed:.2f}s\n"
                    f"- **Model:** {model_name or 'unknown'}\n"
                    f"- **Finish reason:** {finish_reason}\n"
                    f"- **Sample response:** {content or '(empty)'}"
                )
        except urllib.error.HTTPError as exc:
            elapsed = time.time() - start
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return (
                "## vLLM Chat Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** HTTP {exc.code} ({exc.reason})\n"
                f"- **Response time:** {elapsed:.2f}s\n"
                f"- **Error:** {detail}"
            )
        except urllib.error.URLError as exc:
            elapsed = time.time() - start
            return (
                "## vLLM Chat Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** CONNECTION FAILED\n"
                f"- **Response time:** {elapsed:.2f}s\n"
                f"- **Error:** {exc.reason}"
            )
        except Exception as exc:
            return (
                "## vLLM Chat Endpoint\n\n"
                f"- **URL:** {url}\n"
                f"- **Status:** ERROR\n"
                f"- **Error:** {str(exc)[:500]}"
            )

    @staticmethod
    def _check_infracoder_config() -> str:
        """Report the vLLM endpoint configured in InfraCoder."""

        output_lines = ["## InfraCoder vLLM Config"]
        found = False

        # Check environment variables (set via .env or shell profile)
        for var_name in ["OPENAI_BASE_URL", "INFRACODER_BASE_URL"]:
            var_value = os.environ.get(var_name)
            if var_value:
                output_lines.append(
                    f"\n- **{var_name}:** `{var_value}`"
                )
                found = True

        # Try importing InfraCoder config in case .env is loadable
        if not found:
            try:
                sys.path.insert(0, "/home/ubuntu/XYP/InfraCoder")
                from infracoder.config import Config
                cfg = Config.from_env()
                if cfg.base_url:
                    output_lines.append(
                        f"\n- **Config.base_url:** `{cfg.base_url}`"
                    )
                    found = True
            except Exception:
                pass

        if not found:
            output_lines.append(
                "\n- No vLLM base URL configured. "
                "Set OPENAI_BASE_URL or INFRACODER_BASE_URL env var, "
                "or add it to a .env file in the project directory."
            )

        return "\n".join(output_lines)

    def _truncate_output(self, text: str) -> str:
        """Keep tool output from overwhelming the model context."""

        if len(text) <= self._MAX_OUTPUT_CHARS:
            return text

        head_size = 5_000
        tail_size = 2_000

        return (
            text[:head_size]
            + (
                f"\n\n... (output truncated from {len(text)} "
                "characters to preserve context) ...\n\n"
            )
            + text[-tail_size:]
        )
