"""Runtime wrappers that make tracing and RRF easy to wire into the app."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .rrf import reciprocal_rank_fusion
from .tracing import WhiteBoxTracer


def fuse_with_trace(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Document],
    *,
    tracer: Optional[WhiteBoxTracer] = None,
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Document]:
    """Convert dense candidates to ranked docs, fuse with BM25, and trace all ranks."""
    with (tracer.span("vector_search", metadata={"candidate_count": len(vector_results)}) if tracer else _noop()) as span:
        vector_docs = [item[0] for item in vector_results]
        for rank, doc in enumerate(vector_docs, 1):
            doc.metadata["vector_rank"] = rank
            doc.metadata["vector_score_raw"] = float(vector_results[rank - 1][1])
        if span:
            span.metadata["returned_count"] = len(vector_docs)

    with (tracer.span("bm25_search", metadata={"candidate_count": len(bm25_results)}) if tracer else _noop()) as span:
        for rank, doc in enumerate(bm25_results, 1):
            doc.metadata["bm25_rank"] = rank
        if span:
            span.metadata["returned_count"] = len(bm25_results)

    with (tracer.span("rrf_fusion", metadata={"rrf_k": rrf_k}) if tracer else _noop()) as span:
        fused = reciprocal_rank_fusion([vector_docs, bm25_results], key_fn=_doc_key, k=rrf_k)
        fused = fused[:top_k]
        if span:
            span.metadata["top_k"] = top_k
            span.metadata["returned_count"] = len(fused)
            span.metadata["overlap_count"] = len(set(map(_doc_key, vector_docs)) & set(map(_doc_key, bm25_results)))
    return fused


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata
    return f"{metadata.get('source', '')}:{metadata.get('page', metadata.get('page_start', ''))}:{metadata.get('chunk_id', doc.page_content[:240])}"


class _NoopContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


def _noop():
    return _NoopContext()
