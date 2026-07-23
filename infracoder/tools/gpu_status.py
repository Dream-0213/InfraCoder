"""Read-only NVIDIA GPU status inspection tool."""

from __future__ import annotations

import shutil
import subprocess

from .base import Tool


class GPUStatusTool(Tool):
    """Inspect NVIDIA GPU status through nvidia-smi."""

    name = "gpu_status"
    description = (
        "Inspect NVIDIA GPU status on the current machine. "
        "Use this tool when the user asks about GPU model, memory usage, "
        "utilization, temperature, power, GPU processes, or GPU topology. "
        "This tool is read-only and does not modify the system."
    )

    parameters = {
        "type": "object",
        "properties": {
            "detail": {
                "type": "string",
                "enum": ["summary", "processes", "topology", "all"],
                "description": (
                    "Information to inspect. "
                    "'summary' returns GPU model, memory, utilization, "
                    "temperature and power; "
                    "'processes' returns GPU compute processes; "
                    "'topology' returns GPU interconnect topology; "
                    "'all' returns all available information. "
                    "Defaults to 'summary'."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Command timeout in seconds. Defaults to 15 seconds."
                ),
                "minimum": 1,
                "maximum": 60,
            },
        },
        "required": [],
    }

    _MAX_OUTPUT_CHARS = 12_000

    def execute(self, detail: str = "summary", timeout: int = 15) -> str:
        """Return requested NVIDIA GPU status information."""

        if detail not in {"summary", "processes", "topology", "all"}:
            return (
                "Error: detail must be one of "
                "'summary', 'processes', 'topology', or 'all'"
            )

        if not isinstance(timeout, int):
            return "Error: timeout must be an integer"

        if timeout < 1 or timeout > 60:
            return "Error: timeout must be between 1 and 60 seconds"

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return (
                "Error: nvidia-smi was not found. "
                "This tool currently supports NVIDIA GPUs only. "
                "Check whether the NVIDIA driver is installed and "
                "nvidia-smi is available in PATH."
            )

        sections: list[str] = []

        if detail in {"summary", "all"}:
            result = self._run_command(
                [
                    nvidia_smi,
                    "--query-gpu="
                    "index,"
                    "name,"
                    "uuid,"
                    "driver_version,"
                    "temperature.gpu,"
                    "utilization.gpu,"
                    "utilization.memory,"
                    "memory.total,"
                    "memory.used,"
                    "memory.free,"
                    "power.draw,"
                    "power.limit",
                    "--format=csv,noheader,nounits",
                ],
                timeout=timeout,
            )

            if result.startswith("Error:"):
                sections.append(result)
            else:
                sections.append(self._format_summary(result))

        if detail in {"processes", "all"}:
            result = self._run_command(
                [
                    nvidia_smi,
                    "--query-compute-apps="
                    "gpu_uuid,"
                    "pid,"
                    "process_name,"
                    "used_memory",
                    "--format=csv,noheader,nounits",
                ],
                timeout=timeout,
            )

            if result.startswith("Error:"):
                sections.append(result)
            else:
                sections.append(self._format_processes(result))

        if detail in {"topology", "all"}:
            result = self._run_command(
                [nvidia_smi, "topo", "-m"],
                timeout=timeout,
            )

            if result.startswith("Error:"):
                sections.append(result)
            else:
                sections.append(
                    "## GPU Topology\n\n"
                    f"```text\n{result.strip()}\n```"
                )

        output = "\n\n".join(section for section in sections if section)

        if not output:
            return "Error: no GPU information was returned"

        return self._truncate_output(output)

    @staticmethod
    def _run_command(command: list[str], timeout: int) -> str:
        """Run a read-only nvidia-smi command and return its output."""

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return (
                f"Error: command timed out after {timeout} seconds: "
                f"{' '.join(command)}"
            )
        except OSError as exc:
            return f"Error executing nvidia-smi: {exc}"

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            details = stderr or stdout or "unknown nvidia-smi error"
            return (
                f"Error: nvidia-smi exited with code "
                f"{completed.returncode}: {details}"
            )

        return stdout

    @staticmethod
    def _format_summary(raw: str) -> str:
        """Format CSV GPU summary into readable text."""

        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        if not lines:
            return "## GPU Summary\n\nNo NVIDIA GPUs were reported."

        output = ["## GPU Summary"]

        for line in lines:
            fields = [field.strip() for field in line.split(",")]

            if len(fields) != 12:
                output.append(
                    "\n```text\n"
                    f"Unexpected nvidia-smi output: {line}"
                    "\n```"
                )
                continue

            (
                index,
                name,
                uuid,
                driver_version,
                temperature,
                gpu_utilization,
                memory_utilization,
                memory_total,
                memory_used,
                memory_free,
                power_draw,
                power_limit,
            ) = fields

            output.extend(
                [
                    "",
                    f"### GPU {index}: {name}",
                    f"- UUID: {uuid}",
                    f"- Driver: {driver_version}",
                    f"- Temperature: {temperature} °C",
                    f"- GPU utilization: {gpu_utilization}%",
                    f"- Memory utilization: {memory_utilization}%",
                    (
                        f"- Memory: {memory_used} MiB used / "
                        f"{memory_total} MiB total / "
                        f"{memory_free} MiB free"
                    ),
                    f"- Power: {power_draw} W / {power_limit} W",
                ]
            )

        return "\n".join(output)

    @staticmethod
    def _format_processes(raw: str) -> str:
        """Format GPU process CSV output."""

        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        if not lines:
            return "## GPU Processes\n\nNo active GPU compute processes."

        output = ["## GPU Processes"]

        for line in lines:
            fields = [field.strip() for field in line.split(",", maxsplit=3)]

            if len(fields) != 4:
                output.append(
                    "\n```text\n"
                    f"Unexpected nvidia-smi process output: {line}"
                    "\n```"
                )
                continue

            gpu_uuid, pid, process_name, used_memory = fields

            output.extend(
                [
                    "",
                    f"- PID: {pid}",
                    f"  - Process: {process_name}",
                    f"  - GPU UUID: {gpu_uuid}",
                    f"  - GPU memory: {used_memory} MiB",
                ]
            )

        return "\n".join(output)

    def _truncate_output(self, text: str) -> str:
        """Keep tool output from overwhelming the model context."""

        if len(text) <= self._MAX_OUTPUT_CHARS:
            return text

        head_size = 8_000
        tail_size = 3_000

        return (
            text[:head_size]
            + (
                f"\n\n... (GPU output truncated from {len(text)} "
                "characters to preserve context) ...\n\n"
            )
            + text[-tail_size:]
        )