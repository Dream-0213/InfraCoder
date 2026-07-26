"""Knowledge base - document indexing and semantic search for RAG.

InfraCoder 的知识库系统。支持：
1. 关键词搜索（内置，无需额外依赖）
2. 语义搜索（如果 LLM 后端支持 embedding）
3. 文档分块存储和检索

用户通过 CLI 管理知识库：
  infracoder kb add <path>     # 添加文档
  infracoder kb list           # 列出已索引文档
  infracoder kb remove <id>    # 删除文档
  infracoder kb rebuild        # 重建索引

Agent 通过 search_knowledge 工具调用知识库。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from collections import Counter
from ..config import _infracoder_dir

# 支持的文件类型
SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".rst", ".yaml", ".yml", ".json", ".csv", ".html"}

# 分块参数
CHUNK_MIN_CHARS = 100    # 小于此长度的块丢弃
CHUNK_MAX_CHARS = 2000   # 大于此长度的块继续拆分


class KnowledgeBase:
    """文档知识库：索引、搜索、管理索引文档。"""

    def __init__(self, kb_dir: str | Path | None = None):
        self.kb_dir = Path(kb_dir or _infracoder_dir() / "knowledge")
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.kb_dir / "_index.json"
        self._chunks: list[dict] = []
        self._sources: list[str] = []
        self._load_index()

    # ---- 公共 API ----

    def add(self, path: str) -> str:
        """添加一个文件或目录到知识库。返回添加结果描述。"""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: {path} not found"

        if p.is_file():
            return self._add_file(p)
        else:
            return self._add_directory(p)

    def remove(self, doc_id: str) -> str:
        """从知识库中移除一个文档的所有块。"""
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.get("source") != doc_id]
        removed = before - len(self._chunks)
        if removed:
            self._save_index()
            return f"Removed {removed} chunks from '{doc_id}'"
        return f"Document '{doc_id}' not found in knowledge base"

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档及其统计。"""
        sources = {}
        for c in self._chunks:
            src = c.get("source", "unknown")
            if src not in sources:
                sources[src] = {"chunks": 0, "chars": 0}
            sources[src]["chunks"] += 1
            sources[src]["chars"] += len(c.get("content", ""))
        return [
            {"source": s, "chunks": v["chunks"], "chars": v["chars"]}
            for s, v in sorted(sources.items())
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索知识库。先尝试语义搜索，回退到关键词搜索。"""
        if not self._chunks:
            return []

        # 尝试语义搜索（需要 embedding API）
        results = self._semantic_search(query, top_k)
        if results is not None:
            return results

        # 回退：关键词搜索
        return self._keyword_search(query, top_k)

    def rebuild(self) -> str:
        """重建索引（从已存储的源文件路径重新读取）。"""
        # 收集所有不同的源文件路径
        sources = set()
        for chunk in self._chunks:
            src = chunk.get("source", "")
            if src:
                sources.add(src)

        old_count = len(self._chunks)
        self._chunks = []

        files_scanned = 0
        for src_path in sorted(sources):
            p = Path(src_path)
            if p.exists():
                self._add_file(p, save=False)
                files_scanned += 1

        self._save_index()
        if files_scanned > 0:
            return f"Rebuilt index: {files_scanned} files scanned, {len(self._chunks)} chunks total"
        else:
            return f"No source files found. Use 'infracoder kb add <path>' to add documents."

    def stats(self) -> str:
        """知识库统计信息。"""
        return (
            f"Knowledge base: {len(self._chunks)} chunks from "
            f"{len(self.list_documents())} documents"
        )

    # ---- 内部方法 ----

    def _add_file(self, path: Path, save: bool = True) -> str:
        """索引单个文件。"""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {path}: {e}"

        chunks = self._chunk_text(text, path)
        new_count = 0
        for chunk in chunks:
            # 检查是否已存在（按 content 去重）
            if not any(c["content"] == chunk["content"] for c in self._chunks):
                self._chunks.append(chunk)
                new_count += 1

        if save:
            self._sources.append(str(path))
            self._save_index()
        return f"Indexed {path.name}: {new_count} new chunks"

    def _add_directory(self, path: Path) -> str:
        """索引整个目录。"""
        total = 0
        results = []
        for ext in SUPPORTED_EXTENSIONS:
            for f in sorted(path.rglob(f"*{ext}")):
                if f.name.startswith("."):
                    continue
                result = self._add_file(f, save=False)
                total += 1
                results.append(result)
        self._save_index()
        docs = self.list_documents()
        return f"Indexed {total} files, {len(self._chunks)} chunks total"

    def _chunk_text(self, text: str, source_path: Path) -> list[dict]:
        """将文本按段落分块，返回块列表。"""
        chunks = []
        source = str(source_path)

        # 按空行分割段落
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < CHUNK_MAX_CHARS:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk and len(current_chunk) >= CHUNK_MIN_CHARS:
                    chunks.append(self._make_chunk(current_chunk, source))
                current_chunk = para

        if current_chunk and len(current_chunk) >= CHUNK_MIN_CHARS:
            chunks.append(self._make_chunk(current_chunk, source))

        return chunks

    def _make_chunk(self, content: str, source: str) -> dict:
        """创建一个块条目。"""
        return {
            "id": f"{source}:#{hash(content) % 10000000}",
            "source": source,
            "content": content,
            "keywords": self._extract_keywords(content),
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """提取关键词（用于关键词搜索）。"""
        # 提取英文词和中文字符
        words = []
        for match in re.finditer(r"[a-zA-Z]+|[\u4e00-\u9fff]+", text.lower()):
            w = match.group()
            if re.match(r"^[\u4e00-\u9fff]+$", w):
                # 中文：拆成单个字和二元组
                for i in range(len(w)):
                    words.append(w[i])
                for i in range(len(w) - 1):
                    words.append(w[i:i+2])
            else:
                words.append(w)
        # 停用词表
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "because", "but", "and", "or",
            "if", "while", "this", "that", "these", "those", "it", "its",
        }
        return [w for w in words if w not in stopwords and len(w) > 1]

    def _keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        """基于关键词匹配的搜索。"""
        query_keywords = set(self._extract_keywords(query))
        if not query_keywords:
            return []

        scored = []
        for chunk in self._chunks:
            chunk_keywords = set(chunk.get("keywords", []))
            matches = len(query_keywords & chunk_keywords)
            if matches > 0:
                scored.append((matches, chunk))

        scored.sort(key=lambda x: -x[0])
        return [
            {
                "source": c["source"],
                "content": c["content"][:500],
                "score": round(s / len(query_keywords), 2),
            }
            for s, c in scored[:top_k]
        ]

    def _semantic_search(self, query: str, top_k: int = 5) -> list[dict] | None:
        """基于 embedding 的语义搜索。需要 LLM 后端支持 embedding。"""
        try:
            # 使用已配置的 OpenAI 兼容客户端请求 embedding
            from openai import OpenAI
            from ..config import Config

            cfg = Config.from_env()
            api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
            base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL", "")

            if not api_key:
                return None

            client = OpenAI(api_key=api_key, base_url=base_url)

            # 检查 embedding 模型中是否有块缓存的 embedding
            # 首次搜索时计算并缓存所有块的 embedding
            embedding_model = "text-embedding-3-small"

            chunks_to_embed = [
                c for c in self._chunks if "embedding" not in c
            ]
            if chunks_to_embed:
                # 分批调 embedding API
                batch_size = 20
                for i in range(0, len(chunks_to_embed), batch_size):
                    batch = chunks_to_embed[i:i + batch_size]
                    texts = [c["content"][:8000] for c in batch]  # 截断长文本
                    resp = client.embeddings.create(
                        model=embedding_model,
                        input=texts,
                    )
                    for chunk, data in zip(batch, resp.data):
                        chunk["embedding"] = data.embedding
                self._save_index()

            # 计算查询 embedding
            q_resp = client.embeddings.create(
                model=embedding_model,
                input=[query],
            )
            q_emb = q_resp.data[0].embedding

            # 余弦相似度搜索
            import math

            def cosine_sim(a, b):
                """计算余弦相似度。"""
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(x * x for x in b))
                return dot / (na * nb) if na and nb else 0

            scored = []
            for chunk in self._chunks:
                emb = chunk.get("embedding")
                if emb:
                    sim = cosine_sim(q_emb, emb)
                    scored.append((sim, chunk))

            scored.sort(key=lambda x: -x[0])
            return [
                {
                    "source": c["source"],
                    "content": c["content"][:500],
                    "score": round(s, 4),
                }
                for s, c in scored[:top_k]
            ]

        except Exception as e:
            # embedding 失败时静默回退到关键词搜索
            return None

    def _load_index(self):
        """从磁盘加载索引。"""
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._chunks = data.get("chunks", [])
                self._sources = data.get("sources", [])
            except (json.JSONDecodeError, OSError):
                self._chunks = []
                self._sources = []

    def _save_index(self):
        """将索引保存到磁盘。"""
        # 收集所有源文件路径用于重建
        sources = list(set(c.get("source", "") for c in self._chunks if c.get("source")))
        data = {"chunks": self._chunks, "sources": sources, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
