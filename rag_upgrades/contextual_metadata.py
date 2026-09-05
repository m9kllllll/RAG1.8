from __future__ import annotations
import re
from typing import Any, Dict
from langchain_core.documents import Document

def enrich_document_metadata(doc: Document) -> Document:
    meta=dict(doc.metadata); text=doc.page_content; source=str(meta.get("source",""))
    codes=sorted(set(meta.get("course_codes",[]))|set(re.findall(r"\b[A-Z]{2,5}\d{3,5}\b",text.upper())))
    section=str(meta.get("heading_path") or meta.get("section") or meta.get("subsection") or "Root Section")
    meta.update({"document_type":"pdf" if source.lower().endswith(".pdf") else "web","section_path":section,"course_codes":codes,"language_hint":"zh" if re.search(r"[\u4e00-\u9fff]",text) else "en","has_table":bool(meta.get("is_table") or "[TABLE_START]" in text)})
    meta["retrieval_context"]=f"Source: {source}\nCategory: {meta.get('category','')}\nProgramme: {meta.get('programme','')}\nAcademic level: {meta.get('academic_level','')}\nSection: {section}\nCourse codes: {', '.join(codes)}"
    return Document(page_content=text,metadata=meta)
