"""Workflow template system - reusable task templates for common operations.

A workflow template is a pre-defined task instruction that can be reused.
Instead of letting the LLM figure out every step from scratch, the user
(or the LLM) can invoke a workflow like "code_review" or "gpu_check" and
the template handles the step-by-step execution via a sub-agent.

Templates are defined in defaults.py as dicts. Users can add custom templates
by creating a workflows/ directory next to the project and placing .py files
with template definitions.
"""

from __future__ import annotations

from ..tools.base import Tool


class WorkflowTemplate:
    """A reusable task template.

    Each template has:
    - name: unique identifier, used by the WorkflowTool
    - description: what it does, shown to LLM
    - parameters: JSON Schema for user-supplied arguments
    - instruction: the task prompt, with {placeholder} for parameter substitution
    """

    def __init__(
        self,
        name: str,
        description: str,
        instruction: str,
        parameters: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.instruction = instruction
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def render(self, params: dict | None = None) -> str:
        """Fill {placeholders} with user-supplied values."""
        if not params:
            return self.instruction
        return self.instruction.format(**params)


class WorkflowTool(Tool):
    """Tool that executes a pre-defined workflow template via a sub-agent.

    Registered like any other tool. The LLM calls it with a template name
    and optional parameters, and a sub-agent executes the template steps.
    """

    name = "workflow"
    description = "Execute a pre-defined workflow template"
    parameters = {
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "description": "Workflow template name to execute",
            },
            "params": {
                "type": "object",
                "description": "Template parameters (optional, key-value pairs)",
            },
        },
        "required": ["template"],
    }

    _parent_agent = None
    _templates: dict[str, WorkflowTemplate] = {}

    def __init__(self, templates: list[WorkflowTemplate]):
        self._templates = {t.name: t for t in templates}
        names = list(self._templates.keys())

        self.parameters = {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "enum": names,
                    "description": "Workflow template name to execute",
                },
                "params": {
                    "type": "object",
                    "description": "Template parameters (optional, key-value pairs)",
                },
            },
            "required": ["template"],
        }
        if names:
            self.description = (
                "Execute a pre-defined workflow template for common tasks. "
                f"Available: {', '.join(names)}"
            )

    def execute(self, template: str, params: dict | None = None) -> str:
        if self._parent_agent is None:
            return "Error: workflow tool not initialized"

        tpl = self._templates.get(template)
        if not tpl:
            available = ", ".join(self._templates)
            return f"Error: unknown workflow '{template}'. Available: {available}"

        rendered = tpl.render(params)

        from ..agent import Agent

        parent = self._parent_agent
        sub = Agent(
            llm=parent.llm,
            tools=[t for t in parent.tools if t.name not in ("agent", "workflow")],
            max_context_tokens=parent.context.max_tokens,
            max_rounds=30,
        )

        try:
            result = sub.chat(rendered)
            if len(result) > 5000:
                result = result[:4500] + "\n... (workflow output truncated)"
            return f"[Workflow: {template}]\n{result}"
        except Exception as e:
            return f"Workflow error: {e}"


def load_default_templates() -> list[WorkflowTemplate]:
    """Load the built-in default workflow templates."""
    from .defaults import DEFAULT_TEMPLATES
    return [WorkflowTemplate(**t) for t in DEFAULT_TEMPLATES]
