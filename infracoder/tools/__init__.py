"""Tool registry."""
from .bash import BashTool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool
from .gpu_status import GPUStatusTool
from .vllm_status import VLLMStatusTool
from .search_knowledge import SearchKnowledgeTool
from ..workflows import WorkflowTool, load_default_templates


# Load workflow templates and create the workflow tool
_workflow_templates = load_default_templates()
_workflow_tool = WorkflowTool(_workflow_templates)

ALL_TOOLS = [
    GPUStatusTool(),
    VLLMStatusTool(),
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    SearchKnowledgeTool(),
    _workflow_tool,
]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
