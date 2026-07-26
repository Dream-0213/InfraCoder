"""Default workflow templates shipped with InfraCoder.

Each template is a dict with keys:
- name: unique identifier
- description: shown to LLM
- parameters: JSON Schema for user arguments
- instruction: task prompt with {placeholder} support
"""

DEFAULT_TEMPLATES = [
    {
        "name": "gpu_check",
        "description": "Run a comprehensive GPU and vLLM service diagnostics check",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "instruction": (
            "You are running a full GPU diagnostics check.\n\n"
            "Steps:\n"
            "1. Check GPU summary (temperature, memory usage, utilization, power)\n"
            "2. If any GPU reports temperature > 80 C or memory usage > 90%, "
            "also check running processes on that GPU\n"
            "3. Check GPU topology for multi-card connectivity\n"
            "4. Check vLLM service health - model availability and API response time\n"
            "5. Summarize all findings\n\n"
            "Output format:\n"
            "- Start with a status badge (all good / warning / critical)\n"
            "- List each GPU's status\n"
            "- List vLLM service status\n"
            "- Highlight any anomalies"
        ),
    },
    {
        "name": "code_review",
        "description": "Review Python code in a directory with linting and security analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Directory or file path to review",
                },
            },
            "required": ["target"],
        },
        "instruction": (
            "You are performing a code review on \"{target}\".\n\n"
            "Steps:\n"
            "1. Find all Python files in the target directory using 'glob'\n"
            "2. Read each file's content\n"
            "3. Run ruff check on the directory via 'bash'\n"
            "4. Provide a code review covering:\n"
            "   - Security vulnerabilities\n"
            "   - Code style issues\n"
            "   - Potential bugs\n"
            "   - Improvement suggestions\n\n"
            "Format the review with clear headings and file references."
        ),
    },
    {
        "name": "doc_summarize",
        "description": "Read and summarize a document or code file with structured output",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to summarize",
                },
                "style": {
                    "type": "string",
                    "enum": ["detailed", "brief", "bullet"],
                    "description": "Summary style: detailed prose, brief paragraph, or bullet points",
                },
            },
            "required": ["file"],
        },
        "instruction": (
            "Read the file \"{file}\" and provide a summary.\n\n"
            "Include:\n"
            "- What the file does (purpose)\n"
            "- Key functions, classes, or sections\n"
            "- Dependencies or inputs/outputs\n"
            "- Any notable patterns or potential issues\n\n"
            "Output style: {style}"
        ),
    },
    {
        "name": "investigate_error",
        "description": "Investigate an error message, find root cause, and suggest a fix",
        "parameters": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "The error message or description",
                },
                "project": {
                    "type": "string",
                    "description": "Project directory to investigate (optional)",
                },
            },
            "required": ["error"],
        },
        "instruction": (
            "You are investigating this error:\n{error}\n\n"
            "Project directory: {project}\n\n"
            "Steps:\n"
            "1. If a project directory is specified, search for relevant code "
            "(grep for error-related terms)\n"
            "2. Read the relevant files to understand the context\n"
            "3. Identify the root cause\n"
            "4. Propose a fix\n\n"
            "Output format:\n"
            "- Error analysis\n"
            "- Root cause\n"
            "- Suggested fix (with code if applicable)"
        ),
    },
]
