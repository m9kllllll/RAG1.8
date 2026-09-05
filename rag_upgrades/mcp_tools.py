"""Small deterministic functions suitable for MCP/agent tool registration."""
from langchain_core.documents import Document
from .intelligent_chunking import intelligent_chunk_documents
from .contextual_metadata import enrich_document_metadata
from .rrf import reciprocal_rank_fusion

def intelligent_chunk_tool(documents, target_size=1200):
    return intelligent_chunk_documents(documents,target_size=target_size)

def enrich_metadata_tool(documents):
    return [enrich_document_metadata(d) for d in documents]

def rrf_tool(result_lists, rrf_k=60):
    return reciprocal_rank_fusion(result_lists,key_fn=lambda d:f"{d.metadata.get('source','')}:{d.metadata.get('page',d.metadata.get('page_start',''))}:{d.page_content[:240]}",k=rrf_k)
