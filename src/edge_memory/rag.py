"""Offline retrieval-augmented generation over local memory.

Retrieval always works offline (Qdrant Edge + cached embeddings). Generation uses the
in-process local LLM when a model is loaded; otherwise it falls back to an extractive
answer built from the retrieved passages, so the product still responds with no model.
"""

from __future__ import annotations

from pydantic import BaseModel

from edge_memory.llm.base import LLMProvider
from edge_memory.models import SearchHit
from edge_memory.store import EdgeStore

_SYSTEM = (
    "You are an on-device assistant. Answer ONLY from the provided memory context. "
    "If the context is insufficient, say so. Cite memory numbers like [1], [2]."
)


class RagAnswer(BaseModel):
    answer: str
    citations: list[SearchHit]
    used_llm: bool


class RagPipeline:
    def __init__(self, store: EdgeStore, llm: LLMProvider) -> None:
        self._store = store
        self._llm = llm

    def ask(self, question: str, *, top_k: int = 5) -> RagAnswer:
        hits = self._store.hybrid_search(question, limit=top_k, use_hybrid=True)
        context = "\n".join(
            f"[{i + 1}] {hit.memory.text}" for i, hit in enumerate(hits)
        )
        if not hits:
            return RagAnswer(
                answer="No relevant memories found on this device.",
                citations=[],
                used_llm=False,
            )

        if self._llm.available:
            prompt = f"Memory context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            answer = self._llm.generate(prompt, system=_SYSTEM, max_tokens=400)
            return RagAnswer(answer=answer, citations=hits, used_llm=True)

        # Extractive fallback.
        best = hits[0].memory.text
        return RagAnswer(
            answer=f"Based on local memory [1]: {best}",
            citations=hits,
            used_llm=False,
        )
