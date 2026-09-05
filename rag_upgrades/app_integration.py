"""Safe integration layer for app_telegram_hybrid_bm25.py.

Import these helpers from the monolithic app rather than duplicating business
logic. The functions are deliberately dependency-light so the existing app can
keep its FastAPI/Telegram lifecycle unchanged.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .evaluation import evaluate_retrieval, answer_keyword_coverage, citation_completeness
from .tracing import WhiteBoxTracer
from .runtime import fuse_with_trace


def new_query_tracer() -> WhiteBoxTracer:
    return WhiteBoxTracer(
        enabled=os.getenv("RAG_TRACING_ENABLED", "true").lower() in {"1", "true", "yes", "y"},
        sink_path=os.getenv("RAG_TRACE_FILE", "logs/rag_traces.jsonl"),
    )


def trace_query_result(
    tracer: WhiteBoxTracer,
    *,
    query: str,
    answer: str,
    documents: List[Document],
    started_at: float,
) -> Dict[str, Any]:
    """Finalize a query trace with non-sensitive pipeline summary fields."""
    with tracer.span("response_finalize", metadata={"retrieved_count": len(documents)}) as span:
        span.metadata.update({
            "query_length": len(query),
            "answer_length": len(answer),
            "total_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "sources": [str(d.metadata.get("source", ""))[:300] for d in documents[:20]],
            "rrf_scores": [d.metadata.get("_rrf_score") for d in documents[:20]],
        })
    return tracer.summary()


def fuse_candidates(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Document],
    tracer: WhiteBoxTracer,
    top_k: int,
    rrf_k: int = 60,
) -> List[Document]:
    return fuse_with_trace(vector_results, bm25_results, tracer=tracer, top_k=top_k, rrf_k=rrf_k)


def evaluate_answer_metrics(answer: str, expected_terms: List[str], required_citations: List[str]) -> Dict[str, float]:
    return {
        "answer_keyword_coverage": answer_keyword_coverage(answer, expected_terms),
        "citation_completeness": citation_completeness(answer, required_citations),
    }


def evaluate_retrieved_documents(
    documents: List[Document],
    relevant_sources: List[str],
    relevant_course_codes: List[str],
    ks=(3, 5, 10),
) -> Dict[str, float]:
    return evaluate_retrieval(
        documents,
        relevant_sources=relevant_sources,
        relevant_course_codes=relevant_course_codes,
        ks=ks,
    )
