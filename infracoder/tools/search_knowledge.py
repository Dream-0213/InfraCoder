"""Knowledge base search tool - RAG for internal documents."""

from ..knowledge import KnowledgeBase
from .base import Tool


class SearchKnowledgeTool(Tool):
    """Search the company's internal knowledge base."""

    name = "search_knowledge"
    description = (
        "Search the internal knowledge base for documentation, "
        "code references, API docs, or guides. "
        "Use this when you need information about the company's "
        "codebase, internal tools, or project documentation. "
        "This tool is read-only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query describing what you're looking for",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 20)",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, top_k: int = 5) -> str:
        if top_k < 1 or top_k > 20:
            return "Error: top_k must be between 1 and 20"

        try:
            kb = KnowledgeBase()
            results = kb.search(query, top_k)

            if not results:
                # 如果知识库为空，提示用户如何添加文档
                kb_dir = kb.kb_dir
                return (
                    "No results found. "
                    f"The knowledge base is empty. "
                    f"Add documents using: infracoder kb add <path>"
                )

            output = [f"## Knowledge Base Results"]
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                output.append(
                    f"\n### {i}. [{r['source']}]"
                    f"\nScore: {score:.2f}  {score_bar}"
                    f"\n{r['content']}"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error searching knowledge base: {e}"
