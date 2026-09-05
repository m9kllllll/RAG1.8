"""Drop-in integration helpers for app_telegram_hybrid_bm25.py.

Recommended ingestion order:
    docs -> intelligent_chunk_documents -> enrich_document_metadata
    -> build BM25 over retrieval_text -> embed retrieval_text in Qdrant.

Recommended retrieval order:
    vector_results + bm25_results -> reciprocal_rank_fusion -> top-k.
"""
from __future__ import annotations
from typing import List, Tuple
from langchain_core.documents import Document
from .contextual_metadata import enrich_document_metadata
from .intelligent_chunking import intelligent_chunk_documents
from .rrf import reciprocal_rank_fusion


def prepare_documents(documents: List[Document], target_size: int = 1200) -> List[Document]:
    chunks = intelligent_chunk_documents(documents, target_size=target_size)
    return [enrich_document_metadata(d) for d in chunks]


def retrieval_text(doc: Document) -> str:
    """Use metadata + content for lexical retrieval while preserving original content."""
    context = str(doc.metadata.get("retrieval_context", ""))
    return f"{context}\nContent:\n{doc.page_content}".strip()


def fuse_vector_and_bm25(vector_docs: List[Document], bm25_docs: List[Document], rrf_k: int = 60, top_k: int = 5) -> List[Document]:
    fused = reciprocal_rank_fusion([vector_docs, bm25_docs], key_fn=doc_key, k=rrf_k)
    return fused[:top_k]


def doc_key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source','')}:{m.get('page',m.get('page_start',''))}:{m.get('chunk_id',doc.page_content[:240])}"


def annotate_rank_sources(vector_docs: List[Document], bm25_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    for rank, d in enumerate(vector_docs, 1): d.metadata["vector_rank"] = rank
    for rank, d in enumerate(bm25_docs, 1): d.metadata["bm25_rank"] = rank
    return vector_docs, bm25_docs
