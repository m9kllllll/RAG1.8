from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Sequence
from langchain_core.documents import Document

@dataclass
class Boundary:
    text: str
    reason: str

class IntelligentChunker:
    """Deterministic structure-aware chunker: tables > headings > paragraphs > sentences."""
    def __init__(self, target_size: int = 1200, min_size: int = 250, max_size: int = 1700):
        if not (0 < min_size <= target_size <= max_size):
            raise ValueError("Expected min_size <= target_size <= max_size")
        self.target_size, self.min_size, self.max_size = target_size, min_size, max_size

    def _units(self, text: str) -> List[Boundary]:
        parts = re.split(r"(\[TABLE_START\][\s\S]*?\[TABLE_END\])", text)
        out: List[Boundary] = []
        for part in parts:
            if not part.strip(): continue
            if part.startswith("[TABLE_START]"):
                out.append(Boundary(part.strip(), "table")); continue
            for para in re.split(r"\n\s*\n+", part):
                p = para.strip()
                if not p: continue
                if len(p) <= self.max_size:
                    out.append(Boundary(p, "paragraph"))
                else:
                    for sentence in re.split(r"(?<=[。！？.!?])\s+", p):
                        if sentence.strip(): out.append(Boundary(sentence.strip(), "sentence"))
        return out

    def split_document(self, doc: Document) -> List[Document]:
        units = self._units(doc.page_content); chunks: List[Document] = []
        current: List[str] = []; reasons: List[str] = []
        def emit():
            nonlocal current, reasons
            text = "\n\n".join(current).strip()
            if not text: current, reasons = [], []; return
            meta = dict(doc.metadata)
            meta.update({"chunk_strategy":"intelligent_structure_semantic", "chunk_boundary_reasons":sorted(set(reasons)), "chunk_length":len(text)})
            chunks.append(Document(page_content=text, metadata=meta)); current, reasons = [], []
        for unit in units:
            if unit.reason == "table":
                emit(); meta=dict(doc.metadata); meta.update({"chunk_strategy":"intelligent_table_atomic","chunk_boundary_reasons":["table"],"chunk_length":len(unit.text),"is_table":True})
                chunks.append(Document(page_content=unit.text, metadata=meta)); continue
            proposed="\n\n".join(current+[unit.text]).strip()
            if current and len(proposed)>self.target_size: emit()
            current.append(unit.text); reasons.append(unit.reason)
            if len("\n\n".join(current))>=self.target_size: emit()
        emit()
        merged: List[Document]=[]
        for chunk in chunks:
            if merged and len(chunk.page_content)<self.min_size and not chunk.metadata.get("is_table"):
                prev=merged[-1]; text=prev.page_content+"\n\n"+chunk.page_content
                merged[-1]=Document(page_content=text, metadata={**prev.metadata,"chunk_length":len(text),"chunk_strategy":"intelligent_merged_small_fragment"})
            else: merged.append(chunk)
        return merged

def intelligent_chunk_documents(documents: Sequence[Document], target_size=1200, min_size=250, max_size=1700):
    chunker=IntelligentChunker(target_size,min_size,max_size)
    return [chunk for doc in documents for chunk in chunker.split_document(doc)]
