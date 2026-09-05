# RAG1.8 Advanced Retrieval Upgrade

## Pipeline

```text
Web/PDF/PRD
   -> Docling / PDF parser
   -> IntelligentChunker
   -> Contextual metadata enrichment
   -> optional Ollama vision captions
   -> Qdrant dense index + BM25 lexical index
   -> Reciprocal Rank Fusion (RRF)
   -> metadata/academic-level filters
   -> LLM answer + source references
```

## Why this replaces the old scoring

The previous implementation combined a normalized vector score with `1/rank` BM25 and then added heuristic boosts. RRF instead combines the *rank positions* produced by independent retrievers, so it is robust to incompatible score scales.

Default `rrf_k=60` is configurable. Course-code queries can still use the existing academic-level hard filters and heuristic boosts after fusion.

## Intelligent chunking

`IntelligentChunker` keeps tables atomic, respects paragraphs, splits oversized paragraphs at sentence boundaries, and merges tiny fragments. This is intentionally deterministic; it is safer for an academic knowledge base than asking an LLM to invent arbitrary boundaries during every indexing run.

## Contextual metadata

Each chunk receives source, category, programme, academic level, section path, course codes, language hint, table flag and a `retrieval_context` string. Use `retrieval_text()` when building BM25 so lexical matching can see this context.

## Image captioning

`ImageCaptioner` calls the configured Ollama vision model (default `gemma3:4b`) and turns figure/diagram/table images into searchable `[IMAGE_CAPTION]` text. Keep captions in the same metadata lineage as the source page.

## MCP / Skills

`mcp_tools.py` exposes small deterministic functions that can be registered as MCP tools or agent skills: intelligent chunking, metadata enrichment and RRF. MCP should orchestrate these functions; the core ranking/chunking logic should remain deterministic and testable.
