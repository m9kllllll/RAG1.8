from langchain_core.documents import Document
from rag_upgrades.intelligent_chunking import intelligent_chunk_documents
from rag_upgrades.contextual_metadata import enrich_document_metadata
from rag_upgrades.rrf import reciprocal_rank_fusion


def test_intelligent_chunking_preserves_table():
    docs=[Document(page_content="# Programme\n\nIntro text.\n\n[TABLE_START]\n| Code | Title |\n|---|---|\n| ISE4001 | Test |\n[TABLE_END]\n",metadata={"source":"x.pdf"})]
    chunks=intelligent_chunk_documents(docs,target_size=80,min_size=10,max_size=120)
    assert any(c.metadata.get("is_table") for c in chunks)


def test_metadata_enrichment():
    d=enrich_document_metadata(Document(page_content="ISE4001 Course",metadata={"source":"guide.pdf","programme":"ISE","academic_level":"UG"}))
    assert "ISE4001" in d.metadata["course_codes"]
    assert d.metadata["programme"] == "ISE"


def test_rrf_promotes_overlap():
    a=Document(page_content="A",metadata={"source":"a"}); b=Document(page_content="B",metadata={"source":"b"}); c=Document(page_content="C",metadata={"source":"c"})
    out=reciprocal_rank_fusion([[a,b],[b,c]],key_fn=lambda d:d.page_content,k=60)
    assert out[0].page_content == "B"
    assert out[0].metadata["_rrf_score"] > out[-1].metadata["_rrf_score"]
