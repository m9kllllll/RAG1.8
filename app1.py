# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from contextlib import asynccontextmanager
import os
import json
import tempfile
import requests
import uvicorn
import random
import asyncio
import shutil
import pytesseract
import uuid
import pdfplumber
from urllib.parse import urlparse
import re
from pdf2image import convert_from_path
from PIL import Image
from typing import List
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import List, Any
from dotenv import load_dotenv
load_dotenv()

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

# Chains and Reranker
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

rag_chain = None
vector_store = None



# --- Configuration ---
CONFIG_FILE = "config.json"
COLLECTION_NAME = "polyu_advisor_semantic_v4"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# --- Security Functions ---
def is_allowed_domain(url_str):
    """URL Whitelisting: Restrict to official PolyU domains."""
    try:
        parsed = urlparse(url_str)
        domain = parsed.netloc.lower()
        return domain.endswith(".polyu.edu.hk") or domain.endswith(".speed-polyu.edu.hk")
    except Exception:
        return False

def sanitize_filename(filename):
    """Secure Filename Sanitization: Strip traversals, add UUID."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in ".-_")
    return f"{uuid.uuid4().hex}_{safe_name}"

# --- PDF Loaders ---

def extract_tables_as_md(page) -> str:
    """Convert pdfplumber tables to Markdown blocks."""
    blocks = []
    for table in page.extract_tables() or []:
        if not table or len(table) < 2:
            continue
        rows = [r for r in table if any(str(c).strip() for c in r)]
        if len(rows) < 2:
            continue
        md = []
        for row in rows:
            cells = [str(c or "").replace("|", "\\|").strip() for c in row]
            md.append("| " + " | ".join(cells) + " |")
        num_cols = len(rows[0])
        md.insert(1, "|" + "---|" * num_cols)
        blocks.append("\n".join(md))
    if blocks:
        return "\n\n[TABLE_START]\n" + "\n\n".join(blocks) + "\n[TABLE_END]\n"
    return ""

def load_pdf_with_structure(pdf_path: str):
    """Legacy page-level loader for non-PRD PDFs. Keeps tables as Markdown."""
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text += extract_tables_as_md(page)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": pdf_path}
                ))
    return docs

def load_prd_by_sections(pdf_path: str, programme: str = "General") -> List[Document]:
    """
    Structure-aware extraction for PolyU PRD PDFs.
    Returns one Document per logical section / subject / subsection.
    """
    full_text = ""
    page_starts = []  # (char_index, page_num)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_starts.append((len(full_text), i + 1)) 
            text = page.extract_text() or ""
            text += extract_tables_as_md(page)
            full_text += text + "\n\n"

    # Regexes that tolerate '# ' markdown prefixes from pdfplumber
    section_re = re.compile(r'SECTION\s+\d+.*?$',re.MULTILINE | re.IGNORECASE)
    print("SECTION MATCHES =", len(section_matches))
    subsection_re = re.compile(r'^#?\s*\d+\.\d+[A-Z][^\n]*$', re.MULTILINE)
    subject_form_re = re.compile(r'^#?\s*Subject Description Form\s*$', re.MULTILINE)

    def page_at(pos: int) -> int:
        page_num = 1
        for start, pnum in page_starts:
            if start <= pos:
                page_num = pnum
            else:
                break
        return page_num

    section_matches = list(section_re.finditer(full_text))
    if not section_matches:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )

        chunks = splitter.split_text(full_text)

        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path,
                    "programme": programme,
                    "section": "Fallback Split",
                    "category": "Official PDF",
                    "chunk_type": "fallback_split"
                }
            )
            for chunk in chunks
        ]

    documents = []

    for idx, match in enumerate(section_matches):
        sec_start = match.start()
        sec_end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(full_text)
        section_text = full_text[sec_start:sec_end].strip()
        section_title = match.group(0).strip().lstrip("#").strip()

        if not section_text:
            continue

        # >>> SPECIAL CASE: Section 8 — split by Subject Description Form <<<
        if "SYLLABUS" in section_title.upper() or "SUBJECT" in section_title.upper():
            subject_matches = list(subject_form_re.finditer(section_text))

            if len(subject_matches) > 1:
                first_hit = subject_matches[0].start()
                if first_hit > 10:
                    header_chunk = section_text[:first_hit].strip()
                    documents.append(Document(
                        page_content=header_chunk,
                        metadata={
                            "source": pdf_path,
                            "programme": programme,
                            "section": section_title,
                            "subsection": "Syllabus Index",
                            "category": "Official PDF",
                            "chunk_type": "syllabus_index",
                            "page_start": page_at(sec_start),
                            "page_end": page_at(sec_start + first_hit)
                        }
                    ))

                for s_idx, s_match in enumerate(subject_matches):
                    s_start = s_match.start()
                    s_end = subject_matches[s_idx + 1].start() if s_idx + 1 < len(subject_matches) else len(section_text)
                    subj_text = section_text[s_start:s_end].strip()

                    code_m = re.search(r'Subject Code\s+([A-Z0-9]+)', subj_text)
                    subj_code = code_m.group(1) if code_m else f"Unknown-{s_idx}"

                    documents.append(Document(
                        page_content=subj_text,
                        metadata={
                            "source": pdf_path,
                            "programme": programme,
                            "section": section_title,
                            "subsection": f"Subject {subj_code}",
                            "subject_code": subj_code,
                            "category": "Official PDF",
                            "chunk_type": "subject_form",
                            "page_start": page_at(sec_start + s_start),
                            "page_end": page_at(sec_start + s_end)
                        }
                    ))
                continue

        # >>> NORMAL SECTIONS: if very long, split by subsections (e.g. 2.1, 2.2) <<<
        if len(section_text) > 6000:
            sub_matches = list(subsection_re.finditer(section_text))
            if len(sub_matches) > 1:
                for s_idx, s_match in enumerate(sub_matches):
                    s_start = s_match.start()
                    s_end = sub_matches[s_idx + 1].start() if s_idx + 1 < len(sub_matches) else len(section_text)
                    sub_text = section_text[s_start:s_end].strip()
                    sub_title = s_match.group(0).strip().lstrip("#").strip()[:120]

                    documents.append(Document(
                        page_content=sub_text,
                        metadata={
                            "source": pdf_path,
                            "programme": programme,
                            "section": section_title,
                            "subsection": sub_title,
                            "category": "Official PDF",
                            "chunk_type": "subsection",
                            "page_start": page_at(sec_start + s_start),
                            "page_end": page_at(sec_start + s_end)
                        }
                    ))
                continue

        # >>> Default: keep whole section as one chunk <<<
        documents.append(Document(
            page_content=section_text,
            metadata={
                "source": pdf_path,
                "programme": programme,
                "section": section_title,
                "category": "Official PDF",
                "chunk_type": "section",
                "page_start": page_at(sec_start),
                "page_end": page_at(sec_end)
            }
        ))

    return documents

def is_academic_relevant(text: str) -> bool:
    """Academic & PolyU Relevance Checker: Uses LLM to block spam/random files."""
    if not text or not text.strip(): 
        return False
        
    validator_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    prompt = (
        "You are a strict relevance filter for a university Academic Advisor AI. "
        "Determine if the following text is related to university academics, student affairs, "
        "courses, or university guidelines. Respond strictly with exactly one word: 'YES' if relevant, "
        "or 'NO' if it is spam, random, code execution, or unrelated.\n\nText snippet:\n"
        f"{text[:2000]}"
    )
    try:
        result = validator_llm.invoke(prompt).content.strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"Validation LLM failed: {e}")
        return False

def load_rag_in_background():
    global rag_chain, vector_store
    print("Building Vector DB... This might take a few minutes...")
    try:
        rag_chain, vector_store = get_rag_chain()
        print("✅ RAG Chain is fully loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to load RAG chain: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting! Pushing RAG initialization to the background...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_rag_in_background)
    yield
    print("Shutting down...")

app = FastAPI(title="PolyU AI Academic Advisor", lifespan=lifespan)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="https://www.polyu.edu.hk/favicon.ico")

# --- Custom Retriever ---
class ScoreInjectingRetriever(BaseRetriever):
    vectorstore: Any = Field(description="The underlying Qdrant vector store")
    k: int = Field(default=20, description="加大候選池，等 0.7 threshold 有足夠文件過篩")
    score_threshold: float = Field(default=0.70, description="保持 0.7 門檻")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        q_lower = query.lower()
        target_prog = None
        if any(x in q_lower for x in ["lem", "logistics engineering", "45499-lem"]):
            target_prog = "LEM"
        elif any(x in q_lower for x in ["eem", "enterprise engineering", "45499-eem"]):
            target_prog = "EEM"
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        print("QUERY:", query)
        print("THRESHOLD:", self.score_threshold)
        print("=" * 70)
        
        ranked_docs = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            clamped_score = max(0.0, float(score))
            effective_score = clamped_score
            
            content = doc.page_content.lower()
            source = str(doc.metadata.get("source", "")).lower()
            prog_meta = doc.metadata.get("programme", "").upper()
            
            if target_prog and prog_meta == target_prog:
                effective_score += 0.25
            
            is_curriculum_query = any(w in q_lower for w in [
                "curriculum", "programme structure", "study plan", 
                "compulsory subjects", "elective subjects", "graduation requirements", "credits"
            ])
            
            if is_curriculum_query:
                if "progression pattern" in content:
                    effective_score += 0.45
                elif "curriculum structure" in content:
                    effective_score += 0.40
                elif "compulsory subjects" in content or "elective subjects" in content:
                    effective_score += 0.30
                elif doc.metadata.get("chunk_type") == "subject_form":
                    effective_score -= 0.40
                elif "intended learning outcomes" in content:
                    effective_score -= 0.20
            
            if "prd" in source:
                effective_score += 0.10
            
            effective_score = min(effective_score, 1.0)
            
            print(f"\n===== RESULT {rank} =====")
            print("Raw Score:", score)
            print("Final Score:", effective_score)
            print("Source:", doc.metadata.get("source"))
            print("Programme:", prog_meta)
            print("Chunk Type:", doc.metadata.get("chunk_type"))
            print("Subject Code:", doc.metadata.get("subject_code"))
            print("CONTENT:", doc.page_content[:500])
            
            if effective_score >= self.score_threshold:
                MAX_DOC_CHARS = 3000
                content_to_use = doc.page_content[:MAX_DOC_CHARS]
                new_meta = doc.metadata.copy()
                new_meta["_score"] = round(effective_score, 3)
                ranked_docs.append(Document(page_content=doc.page_content[:3000], metadata=new_meta))
        
        ranked_docs.sort(key=lambda d: d.metadata.get("_score", 0), reverse=True)
        result_docs = ranked_docs[:1]
        
        print("PASSED THRESHOLD:", len(result_docs))
        
        if not result_docs:
            safe_content = (
                "SYSTEM WARNING: 搵唔到高相關性文件（全部低過 70% 置信度）。"
                "請唔好亂估亂答。禮貌咁叫學生聯絡 Academic Registry（AR）查詢。"
            )
            return [Document(page_content=safe_content, metadata={"source": "System Safeguard", "_score": 0.0})]
        
        return result_docs


def get_rag_chain():
    print("QDRANT_URL =", QDRANT_URL)
    print("OLLAMA_BASE_URL =", OLLAMA_BASE_URL)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url=OLLAMA_BASE_URL
    )
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1200)
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        all_docs = []
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as cfg:
                config_data = json.load(cfg)
                urls = config_data.get("urls", [])
                pdf_paths = config_data.get("pdfs", [])
                word_paths = config_data.get("word_docs", [])
                print("Config loaded")
                print("URLs loaded:", len(urls))
                print("PDFs loaded:", len(pdf_paths))
                print("Word docs loaded:", len(word_paths))
            
            if urls:
                for url in urls:
                    print("Fetching:", url)
                    try:
                        response = requests.get(f"https://r.jina.ai/{url}", timeout=120)
                        if response.status_code == 200:
                            print("Success:", url)
                            all_docs.append(Document(
                                page_content=response.text, 
                                metadata={"source": f"Official Website: {url}", "category": "Official Guidelines"}
                            ))
                    except Exception:
                        pass

            if pdf_paths:
                for pdf_url in pdf_paths:
                    try:
                        print("Fetching PDF:", pdf_url)
                        response = requests.get(pdf_url, timeout=120)
                        if response.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(response.content)
                                temp_pdf = tmp.name

                            prog = "LEM" if any(x in pdf_url.upper() for x in ["LEM", "45499-LEM"]) else \
                                   "EEM" if any(x in pdf_url.upper() for x in ["EEM", "45499-EEM"]) else "General"

                            # 🔥 Route PRD files to structure-aware loader
                            if "PRD" in pdf_url.upper() or "45499" in pdf_url.upper():
                                docs = load_prd_by_sections(temp_pdf, prog)
                                for doc in docs:
                                    doc.metadata["programme"] = prog
                                all_docs.extend(docs)
                                print(f"✅ Loaded PRD by sections: {pdf_url} ({len(docs)} chunks)")
                            else:
                                docs = load_pdf_with_structure(temp_pdf)
                                for doc in docs:
                                    doc.metadata["source"] = pdf_url
                                    doc.metadata["category"] = "Official PDF"
                                    doc.metadata["programme"] = prog
                                all_docs.extend(docs)
                                print(f"✅ Loaded PDF: {pdf_url} ({len(docs)} pages)")

                            os.unlink(temp_pdf)
                    except Exception as e:
                        print(f"❌ PDF FAILED: {pdf_url}")
                        print(e)

            if word_paths:
                for path in word_paths:
                    if os.path.exists(path):
                        try:
                            loader = Docx2txtLoader(path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = f"Official Word Doc: {path}"
                            all_docs.extend(docs)
                        except Exception as e:
                            print(f"Failed to load Word doc from config {path}: {e}")
        
        if all_docs:
            print("Total raw docs:", len(all_docs))

            # 🔥 Separate PRD logical chunks from docs that need splitting
            prd_chunks = [d for d in all_docs if d.metadata.get("chunk_type")]
            raw_docs = [d for d in all_docs if not d.metadata.get("chunk_type")]

            splits = prd_chunks[:]  # PRD chunks are already final

            if raw_docs:
                print("Starting Hybrid Chunking for non-PRD documents...")
                pre_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
                    is_separator_regex=False,
                )
                semantic_splitter = SemanticChunker(
                    embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=85,
                )
                pre_splits = pre_splitter.split_documents(raw_docs)
                for split in pre_splits:
                    if len(split.page_content) > 1500:
                        splits.extend(semantic_splitter.split_documents([split]))
                    else:
                        splits.append(split)
                print("Hybrid Chunking Finished")

            print("Final chunks:", len(splits))

            for i, chunk in enumerate(splits[:5]):
                print(f"\n===== CHUNK {i+1} =====")
                print(chunk.page_content[:1000])
                print("=" * 80)
                print("Metadata:", chunk.metadata)
            
            print("===================================")
            print("Uploading chunks to Qdrant...")
            v_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings
            )

            batch_size = 50
            total_batches = (len(splits) - 1) // batch_size + 1
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                print(f"Uploading batch {i // batch_size + 1}/{total_batches}")
                try:
                    v_store.add_documents(batch)
                except Exception as e:
                    print("UPLOAD FAILED")
                    print(e)
                    continue
            print("Uploaded to Qdrant")

    v_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    global vector_store
    vector_store = v_store
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    retriever = ScoreInjectingRetriever(vectorstore=v_store, k=10, score_threshold=0.70)
    
    contextualize_q_system_prompt = """
    Rewrite the latest user question into a short standalone search query.

    Rules:
    - Return only the search query.
    - Do not answer the question.
    - Keep programme names, subject names and codes.
    - Preserve important keywords.
    - Be concise and retrieval-focused.

    Examples:
    User:
    What subjects do students in EEM need to take?
    Output:
    EEM compulsory subjects
    User:
    What is the curriculum of PIE?
    Output:
    PIE curriculum
    User:
    What are the graduation requirements of LEM?
    Output:
    LEM graduation requirements
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = retriever
    
    system_prompt = """
    You are Alex, an Academic Advisor from the Department of Industrial and Systems Engineering (ISE), PolyU.

    Answer only using the retrieved context.

    Rules:
    1. Use only information found in the retrieved context.
    2. Do not guess.
    3. Do not infer missing information.
    4. Do not combine information from different programmes.
    5. If the answer is not found in the retrieved context, say so clearly.

    Curriculum Questions:

    When answering questions about:
    - curriculum
    - programme structure
    - study plan
    - compulsory subjects
    - elective subjects
    - graduation requirements
    - credits

    Prioritise information from:
    - Programme Structure
    - Curriculum Structure
    - Progression Pattern of the Curriculum
    - Programme Contents
    - Compulsory Subjects
    - Elective Subjects

    Do not use Subject Description Forms unless the user specifically asks about a subject.

    When listing subjects:
    - Copy only the subjects appearing in the retrieved curriculum information.
    - Do not add subjects.
    - Do not complete missing lists.
    - Do not generate your own study plan.

    Reply in the same language as the user.
    IMPORTANT:
    If a curriculum table exists in the retrieved context:
    - Use the curriculum table directly.
    - Ignore programme aims.
    - Ignore intended learning outcomes.
    - Ignore subject description forms.
    - Ignore unrelated programmes.
    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))
    return chain, v_store
    
def auto_crawl_and_index_syllabi(root_url: str):
    """
    Crawls a root website, extracts Markdown links, filters for PolyU course/syllabus 
    documents, and automatically indexes them into Qdrant.
    """
    if not vector_store:
        return {"status": "error", "message": "Vector store not initialized."}

    print(f"🌐 [Auto-Crawler] Starting crawl for: {root_url}")
    
    try:
        response = requests.get(f"https://r.jina.ai/{root_url}", timeout=30)
        if response.status_code != 200:
            return {"status": "error", "message": "Failed to fetch root URL."}
        root_content = response.text
    except Exception as e:
        return {"status": "error", "message": str(e)}

    markdown_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    found_links = re.findall(markdown_link_pattern, root_content)
    
    valid_links = {}
    for text, link in found_links:
        link_lower = link.lower()
        if "polyu.edu.hk" in link_lower or "speed-polyu.edu.hk" in link_lower:
            if any(keyword in link_lower for keyword in ["syllabus", "subject", "course", "programme", ".pdf"]):
                valid_links[link] = text
                
    print(f"🔍 [Auto-Crawler] Found {len(valid_links)} relevant syllabus links.")
    
    new_docs = []
    for link, text in valid_links.items():
        try:
            print(f"📥 Fetching linked syllabus: {link}")
            child_res = requests.get(f"https://r.jina.ai/{link}", timeout=500)
            if child_res.status_code == 200 and len(child_res.text) > 100:
                if is_academic_relevant(child_res.text):
                    new_docs.append(Document(
                        page_content=child_res.text,
                        metadata={
                            "source": f"Auto-Crawled: {link}",
                            "category": "Course Syllabus",
                            "link_text": text
                        }
                    ))
        except Exception as e:
            print(f"❌ Failed to fetch {link}: {e}")

    if new_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(new_docs)
        vector_store.add_documents(splits)
        print(f"✅ [Auto-Crawler] Successfully indexed {len(new_docs)} documents.")
        return {"status": "success", "message": f"Successfully found and indexed {len(new_docs)} linked syllabi documents."}
    
    return {"status": "warning", "message": "Crawled successfully, but no new valid syllabus documents were found."}


# --- HTML UI Template (Unchanged to preserve UI) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PolyU AI Academic Advisor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f5f7; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        .chat-container { flex: 1; display: flex; flex-direction: column; max-width: 800px; margin: 0 auto; width: 100%; background: white; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        .header { background: #800020; color: white; padding: 20px; text-align: center; font-size: 1.2rem; font-weight: bold; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0;z-index: 1000; }
        .messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap; word-wrap: break-word; }
        .user { background: #e1f5fe; color: #0277bd; align-self: flex-end; border-bottom-right-radius: 2px; }
        .ai { background: #f1f1f1; color: #333; align-self: flex-start; border-bottom-left-radius: 2px; }
        .input-area { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; background: #fff; align-items: flex-end; }
        textarea#user_input { 
            flex: 1; 
            padding: 12px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            outline: none; 
            font-size: 1rem; 
            font-family: inherit;
            resize: none;
            min-height: 20px;
            max-height: 150px;
            overflow-y: auto;
            line-height: 1.5;
        }
        button { background: #800020; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: bold; height: 48px; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .header-btns button { border: none; font-size: 0.8rem; padding: 8px 12px; border-radius: 6px; cursor: pointer; color: white; font-weight: bold; height: auto;}
        .quit-btn { background: #555; }
        .quit-btn:hover { background: #333; }
        .restart-btn { background: #4CAF50; display: none; }
        .restart-btn:hover { background: #45a049; }
        think { display: block; font-size: 0.85rem; color: #666; background-color: #e9ecef; padding: 12px; border-radius: 8px; border-left: 4px solid #800020; margin-bottom: 12px; line-height: 1.4; }
        think::before { content: "🧠 Alex is thinking..."; display: block; font-weight: bold; margin-bottom: 6px; color: #555; }
        think[data-time]::after { content: "⏱️ Thinking time: " attr(data-time) " s"; display: block; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ccc; font-size: 0.75rem; color: #888; font-weight: bold; text-align: right; }
        .source-box { margin-top: 12px; font-size: 0.82rem; border-top: 1px dashed #ccc; padding-top: 8px; color: #555; }
        .source-box ul { padding-left: 0; list-style-type: none; margin-top: 8px; }
        .source-box li { margin-bottom: 6px; display: flex; align-items: center; }
        .source-link { color: #800020; text-decoration: none; font-weight: 500; }
        .typing-indicator { display: none; font-style: italic; color: #888; padding: 12px 16px; }
        .conf-badge { padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-right: 8px; min-width: 35px; text-align: center; color: white;}
        .conf-high { background-color: #4CAF50; }
        .conf-med { background-color: #FF9800; }
        .conf-low { background-color: #F44336; }
        #staging-area { padding: 0 20px; display: flex; gap: 10px; flex-wrap: wrap; background: #fff; padding-top: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <span>🎓 Alex - PolyU Academic Advisor</span>
            <div class="header-btns">
                <button id="quit-btn" class="quit-btn" onclick="quitChat()">Quit</button>
                <button id="restart-btn" class="restart-btn" onclick="restartChat()">Restart Chat</button>
            </div>
        </div>
        <div class="messages" id="chat-box"></div>
        <div id="staging-area"></div>
        <form class="input-area" id="chat-form">
            <input type="file" id="file_upload" accept=".pdf,image/*,.doc,.docx*" multiple style="display: none;">
            <select id="doc_category" style="padding: 12px; border: 1px solid #ddd; border-radius: 8px; outline: none; font-size: 0.9rem; background: #f9f9f9;">
                <option value="General">📁 General</option>
                <option value="Course Syllabus">📚 Course Syllabus</option>
                <option value="Academic Rules">⚖️ Academic Rules</option>
                <option value="Student Affairs">🎓 Student Affairs</option>
            </select>
            <button type="button" id="attach-btn" style="background: #e0e0e0; color: #333;" onclick="document.getElementById('file_upload').click()">📎 Attach</button>
            <textarea id="user_input" placeholder="Ask your academic question or attach files/URLs..." autofocus autocomplete="off" rows="1"></textarea>
            <button type="submit" id="send-btn">Send</button>
        </form>
    </div>
    <script>
        const chatBox = document.getElementById("chat-box");
        let chatHistory = []; 
        let stagedFiles = []; 
        const quitMessages = [
            "Goodbye! Good luck with your studies at PolyU!",
            "See you later! Feel free to come back if you have more questions.",
            "Take care! Remember to balance your study and rest.",
            "Farewell! Wishing you a great semester ahead!"
        ];
        window.onload = () => {
            const savedHTML = localStorage.getItem('chat_history_html');
            const savedData = localStorage.getItem('chat_history_data');
            if (savedHTML && savedData) {
                chatBox.innerHTML = savedHTML;
                chatHistory = JSON.parse(savedData);
            } else {
                showGreeting();
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        };
        function showGreeting() {
            chatBox.innerHTML = `<div class="message ai">Hello! I'm Alex, your PolyU academic advisor. Is there anything I can help you with today? 😊</div>`;
        }
        function quitChat() {
            const randomMsg = quitMessages[Math.floor(Math.random() * quitMessages.length)];
            chatBox.innerHTML += `<div class="message ai">${randomMsg}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            document.getElementById('chat-form').style.display = 'none';
            document.getElementById('staging-area').style.display = 'none';
            document.getElementById('quit-btn').style.display = 'none';
            document.getElementById('restart-btn').style.display = 'inline-block';
            chatHistory.push({ role: "ai", content: randomMsg });
            localStorage.setItem('chat_history_html', chatBox.innerHTML);
            localStorage.setItem('chat_history_data', JSON.stringify(chatHistory));
        }
        function restartChat() {
            const welcomeBackMsg = "Welcome back! What else would you like to discuss? 😊";
            chatBox.innerHTML += `<div class="message ai">${welcomeBackMsg}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            chatHistory.push({ role: "ai", content: welcomeBackMsg });
            localStorage.setItem('chat_history_html', chatBox.innerHTML);
            localStorage.setItem('chat_history_data', JSON.stringify(chatHistory));
            document.getElementById('chat-form').style.display = 'flex';
            document.getElementById('staging-area').style.display = 'flex';
            document.getElementById('quit-btn').style.display = 'inline-block';
            document.getElementById('restart-btn').style.display = 'none';
            document.getElementById('user_input').focus();
        }
        function removeStagedFile(index, element) {
            stagedFiles[index] = null; 
            element.parentElement.remove();
        }
        document.getElementById("file_upload").addEventListener("change", function() {
            const stagingArea = document.getElementById("staging-area");
            for (let i = 0; i < this.files.length; i++) {
                stagedFiles.push(this.files[i]);
                const badge = document.createElement("span");
                badge.style.cssText = "background: #e1f5fe; color: #0277bd; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; display: flex; align-items: center; gap: 5px;";
                badge.innerHTML = `📄 ${this.files[i].name} <b style="cursor:pointer; color:red;" onclick="removeStagedFile(${stagedFiles.length - 1}, this)">×</b>`;
                stagingArea.appendChild(badge);
            }
            this.value = ''; 
        });
        const userInput = document.getElementById("user_input");
        userInput.addEventListener("input", function() {
            this.style.height = "auto";
            this.style.height = (this.scrollHeight) + "px";
        });
        userInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                document.getElementById("send-btn").click();
            }
        });
        document.getElementById("chat-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const input = document.getElementById("user_input");
            const val = input.value.trim();
            const filesToUpload = stagedFiles.filter(f => f !== null);
            if (!val && filesToUpload.length === 0) return;
            const displayMsg = val || `[Attached ${filesToUpload.length} file(s)]`;
            chatBox.innerHTML += `<div class="message user">${displayMsg}</div>`;
            chatHistory.push({ role: "user", content: displayMsg });
            input.value = '';
            input.style.height = 'auto'; 
            chatBox.scrollTop = chatBox.scrollHeight;
            const urlRegex = /(https?:\\/\\/[^\\s]+)/g;
            const extractedUrls = val.match(urlRegex) || [];
            if (filesToUpload.length > 0 || extractedUrls.length > 0) {
                const statusId = "status-" + Date.now();
                chatBox.innerHTML += `<div class="message ai" id="${statusId}"><i>Processing your attachments... Please wait.</i></div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
                const formData = new FormData();
                filesToUpload.forEach(file => formData.append("files", file));
                if (extractedUrls.length > 0) {
                    formData.append("links", extractedUrls.join(","));
                }
                formData.append("category", document.getElementById("doc_category").value);
                try {
                    const response = await fetch("/upload_source", { method: "POST", body: formData });
                    const result = await response.json();
                    if (result.status === "success") {
                        document.getElementById(statusId).remove();
                    } else {
                        document.getElementById(statusId).innerHTML = `<span style="color:red;">Attachment Error: ${result.message}</span>`;
                        chatBox.scrollTop = chatBox.scrollHeight;
                        return; 
                    }
                } catch (err) {
                    document.getElementById(statusId).innerHTML = `<span style="color:red;">Failed to process attachments.</span>`;
                    chatBox.scrollTop = chatBox.scrollHeight;
                    return; 
                }
                stagedFiles = [];
                document.getElementById("staging-area").innerHTML = "";
            }
            const aiId = "ai-" + Date.now();
            chatBox.innerHTML += `<div class="message ai" id="${aiId}"><span class="content"></span><div class="sources-area" style="display: none;"></div></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            let fullRaw = "", startTime = 0, duration = 0, cachedSources = [];
            try {
                const res = await fetch('/stream_chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ user_input: displayMsg, history: chatHistory })
                });
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";
                while(true) {
                    const {done, value} = await reader.read();
                    if(done) break;
                    buffer += decoder.decode(value, {stream: true});
                    let parts = buffer.split("\\n\\n");
                    buffer = parts.pop();
                    for(let line of parts) {
                        if(line.startsWith("data: ")) {
                            let data = JSON.parse(line.substring(6));
                            if(data.type === 'chunk') {
                                fullRaw += data.content;
                                if(fullRaw.includes(' \\u003c/think\\u003e') && startTime === 0) startTime = Date.now();
                                if(fullRaw.includes('\\u003c/think\\u003e') && duration === 0) {
                                    duration = ((Date.now() - startTime)/1000).toFixed(1);
                                    let t = document.querySelector(`#${aiId} think`);
                                    if(t) t.setAttribute('data-time', duration);
                                }
                                document.querySelector(`#${aiId} .content`).innerHTML += data.content.replace(/\\n/g, '<br>');
                                chatBox.scrollTop = chatBox.scrollHeight;
                            } else if(data.type === 'sources') {
                                cachedSources = data.content;
                            } else if(data.type === 'done') {
                                if(cachedSources.length > 0) {
                                    let topScore = Math.max(...cachedSources.map(s => s.score));
                                    let avgConf = Math.round(topScore * 100);
                                    let overallColor = avgConf >= 75 ? '#4CAF50' : (avgConf >= 50 ? '#FF9800' : '#F44336');
                                    let sourcesHTML = `<div class="source-box">
                                        <strong>📌 References <span style="font-size:0.8rem; color:${overallColor}; font-weight:normal;">(Max Confidence: ${avgConf}%)</span>:</strong>
                                        <ul>`;
                                    sourcesHTML += cachedSources.map(s => {
                                        let sConf = Math.round(s.score * 100);
                                        let badgeClass = sConf >= 75 ? 'conf-high' : (sConf >= 50 ? 'conf-med' : 'conf-low');
                                        let safeCategory = s.category || "General";
                                        return `<li>
                                            <span style="background: #e9ecef; color: #495057; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-right: 5px; border: 1px solid #ced4da;">${safeCategory}</span>
                                            <span class="conf-badge ${badgeClass}">${sConf}%</span>
                                            <a class="source-link" href="${s.source}" target="_blank">${s.source.substring(0,45)}...</a>
                                        </li>`;
                                    }).join('');
                                    sourcesHTML += `</ul></div>`;
                                    let sArea = document.querySelector(`#${aiId} .sources-area`);
                                    sArea.innerHTML = sourcesHTML;
                                    sArea.style.display = 'block';
                                }
                                chatHistory.push({ role: "ai", content: fullRaw });
                                localStorage.setItem('chat_history_html', chatBox.innerHTML);
                                localStorage.setItem('chat_history_data', JSON.stringify(chatHistory));
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                        }
                    }
                }
            } catch(err) { console.error(err); }
        });
    </script>
</body>
</html>
"""

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_chat_page():
    return HTML_TEMPLATE
  
@app.post("/stream_chat")
async def stream_chat(request: Request):
    print("STREAM_CHAT CALLED")
    data = await request.json()
    user_input = data.get("user_input")
    frontend_history = data.get("history", []) 
    
    global rag_chain
    
    max_history_messages = 4
    history_to_process = frontend_history[-max_history_messages:] if len(frontend_history) > max_history_messages else frontend_history

    langchain_history = [
        ("human" if h["role"] == "user" else "ai", h["content"]) 
        for h in history_to_process
        if not h["content"].startswith("ERROR:")
    ]
    
    async def event_generator():

        if rag_chain is None:
            yield f"data: {json.dumps({'type':'chunk','content':'⏳ Please wait a moment, Alex is still initializing the knowledge base. Try again in a minute!'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        sources = []

        try:

            async for chunk in rag_chain.astream({
                "input": user_input,
                "chat_history": langchain_history
            }):

                print("CHUNK:", chunk)

                if "context" in chunk:

                    sources = [
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "score": doc.metadata.get("_score", 0.0),
                            "category": doc.metadata.get("category", "General")
                        }
                        for doc in chunk["context"]
                    ]

                    yield f"data: {json.dumps({'type':'sources','content':sources})}\n\n"

                if "answer" in chunk:

                    yield f"data: {json.dumps({'type':'chunk','content':chunk['answer']})}\n\n"

        except Exception as e:

            print("\n============= ERROR =============")
            print(e)
            print("=================================\n")

            yield f"data: {json.dumps({'type':'chunk','content':f'ERROR: {str(e)}'})}\n\n"

        yield f"data: {json.dumps({'type':'done'})}\n\n"


        
       

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload_source")
def upload_source(files: List[UploadFile] = File(None), links: str = Form(None), category: str = Form("General")):
    if not rag_chain or not vector_store:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet.")
        
    new_docs = []
    
    if files:
        for file in files:
            if not file.filename:
                continue
            file_ext = file.filename.lower().split('.')[-1]
            
            if file_ext in ["jpg", "jpeg", "png", "bmp", "webp"]:
                try:
                    img = Image.open(file.file)
                    try:
                        text = pytesseract.image_to_string(img, lang="eng+chi_tra")
                    except Exception:
                        text = pytesseract.image_to_string(img)
                    if text.strip():
                        new_docs.append(Document(page_content=text.strip(), metadata={"source": f"User Photo Upload: {file.filename}","category": category}))
                except Exception as e:
                    print(f"Failed to process image {file.filename}: {str(e)}")

            elif file_ext in ["docx", "doc"]:
                tmp_name = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_name = tmp.name
                    loader = Docx2txtLoader(tmp_name)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = f"User Upload: {file.filename}"
                    new_docs.extend(docs)
                    print(f"✅ Successfully processed Word document: {file.filename}")
                except Exception as e:
                    print(f"❌ Failed to process Word Doc {file.filename}: {str(e)}")
                finally:
                    if tmp_name and os.path.exists(tmp_name):
                        os.unlink(tmp_name)

            elif file_ext == "pdf":
                tmp_name = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_name = tmp.name
                    
                    # 🔥 Route user-uploaded PRDs to structure-aware loader
                    if "PRD" in file.filename.upper() or "45499" in file.filename.upper():
                        docs = load_prd_by_sections(tmp_name, programme="General")
                        for doc in docs:
                            doc.metadata["category"] = category
                        new_docs.extend(docs)
                        print(f"✅ Uploaded PRD processed by sections: {file.filename} ({len(docs)} chunks)")
                    else:
                        loader = PyPDFLoader(tmp_name)
                        docs = loader.load()
                        extracted_text = "".join([d.page_content for d in docs]).strip()
                        
                        if len(extracted_text) < 50:
                            images = convert_from_path(tmp_name)
                            ocr_text = ""
                            for idx, page_img in enumerate(images):
                                try:
                                    p_text = pytesseract.image_to_string(page_img, lang="eng+chi_tra")
                                except Exception:
                                    p_text = pytesseract.image_to_string(page_img)
                                ocr_text += f"\\n--- Page {idx + 1} ---\\n" + p_text
                            if ocr_text.strip():
                                new_docs.append(Document(page_content=ocr_text.strip(), metadata={"source": f"User Scanned PDF: {file.filename}","category": category}))
                        else:
                            for doc in docs:
                                doc.metadata["source"] = f"User Upload: {file.filename}"
                                doc.metadata["category"] = category
                            new_docs.extend(docs)
                except Exception as e:
                    print(f"Failed to process PDF {file.filename}: {str(e)}")
                finally:
                    if tmp_name and os.path.exists(tmp_name):
                        os.unlink(tmp_name)

    if links:
        url_list = [url.strip() for url in links.split(",") if url.strip()]
        for url in url_list:
            try:
                response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
                if response.status_code == 200:
                    new_docs.append(Document(page_content=response.text, metadata={"source": f"User Link: {url}","category": category}))
            except Exception as e:
                print(f"Failed to fetch link {url}: {str(e)}")

    if new_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(new_docs)
        vector_store.add_documents(splits)
        return {"status": "success", "message": f"Processed {len(files) if files else 0} files and {len(url_list) if links else 0} links."}
    
    return {"status": "error", "message": "No valid text could be collected from the provided files or links."}
    
@app.post("/deep_crawl")
async def deep_crawl_endpoint(url: str = Form(...)):
    if not is_allowed_domain(url):
        return {"status": "error", "message": "Domain not allowed. Please use official PolyU domains."}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, auto_crawl_and_index_syllabi, url)
    return result