# -*- coding: utf-8 -*-
import os
import json
import tempfile
import requests
import uvicorn
import asyncio
import sqlite3
import re
import time
import logging
from urllib.parse import urljoin
from typing import List, Tuple, Any
from contextlib import asynccontextmanager

# FastAPI Imports (REQUIRED)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from pydantic import Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Telegram Imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

# LangChain & Qdrant Imports
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv(override=True)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# --- Global Configurations & Environment Variables ---
CONFIG_FILE = "config.json"
COLLECTION_NAME = "polyu_advisor_semantic"
DB_FILE = "polyu_advisor.db"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

rag_chain = None
vector_store = None
tg_app = None

# ==========================================
# SQLITE DATABASE SETUP & SESSION TRACKING
# ==========================================
def init_sqlite_db():
    """Initializes SQLite tables for requirements, knowledge base, and student sessions."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS polyu_requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        sub_category TEXT,
        code TEXT UNIQUE,
        title TEXT NOT NULL,
        credits INTEGER DEFAULT 3,
        description TEXT NOT NULL,
        department_owner TEXT DEFAULT 'ISE'
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ise_knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        student_year INTEGER,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        vector_embedded BOOLEAN DEFAULT 0
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS student_sessions (
        student_chat_id TEXT PRIMARY KEY,
        current_faculty TEXT,
        last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()
    print("✅ SQLite database initialized successfully.")

def update_student_session(chat_id: str, faculty: str = None):
    """Logs or updates student interaction state and faculty choice in SQLite."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO student_sessions (student_chat_id, current_faculty, last_interaction)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(student_chat_id) DO UPDATE SET
                current_faculty = COALESCE(excluded.current_faculty, student_sessions.current_faculty),
                last_interaction = CURRENT_TIMESTAMP;
        """, (str(chat_id), faculty))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ SQLite session update failed: {e}")

def clear_user_history(user_id: str) -> bool:
    """Removes stored session history records for the given user ID."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_sessions WHERE student_chat_id = ?", (str(user_id),))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Error clearing SQLite history for user {user_id}: {e}")
        return False


# ==========================================
# GLOSSARY INDEX SETUP & CORRECTION FUNCTION
# ==========================================
TERM_INDEX = {}
if os.path.exists("term_index.json"):
    try:
        with open("term_index.json", "r", encoding="utf-8") as f:
            TERM_INDEX = json.load(f)
        print(f"✅ Loaded {len(TERM_INDEX)} terms from term_index.json")
    except Exception as e:
        print(f"❌ Error loading term_index.json: {e}")

def apply_index_corrections(text: str) -> Tuple[str, str]:
    """
    Enriches user queries with degree titles, abbreviations, PolyU programme codes,
    and JUPAS codes from term_index.json to improve Qdrant vector retrieval.
    """
    if not text or not TERM_INDEX:
        return text, "General"
        
    detected_category = "General"
    
    for key, data in TERM_INDEX.items():
        # 1. Fallback for simple non-dictionary entries
        if not isinstance(data, dict):
            pattern = rf'\b{re.escape(key)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, f"{key} ({data})", text, flags=re.IGNORECASE, count=1)
            continue
            
        # 2. Extract full metadata attributes
        eng = data.get("english", key)
        chi = data.get("chinese", "")
        cat = data.get("category", "General")
        abbr = data.get("abbreviation", key)
        prog_code = data.get("programme_code", "")
        jupas_code = data.get("jupas_code", "")
        
        # 3. Assemble expansion text
        expansion_parts = []
        if abbr and abbr != key and abbr != eng:
            expansion_parts.append(f"Abbr: {abbr}")
        if eng and eng != key: 
            expansion_parts.append(f"Title: {eng}")
        if chi: 
            expansion_parts.append(chi)
        if prog_code: 
            expansion_parts.append(f"PolyU Code: {prog_code}")
        if jupas_code: 
            expansion_parts.append(f"JUPAS: {jupas_code}")
            
        expansion_str = f" [{', '.join(expansion_parts)}]" if expansion_parts else ""
        
        # 4. Check for matches across key, abbreviation, English title, or Chinese title
        tokens_to_check = list(filter(None, set([key, abbr])))
        matched_token = None
        
        for token in tokens_to_check:
            if re.search(rf'\b{re.escape(token)}\b', text, re.IGNORECASE):
                matched_token = token
                break
                
        if not matched_token:
            if eng and eng.lower() in text.lower():
                matched_token = eng
            elif chi and chi in text:
                matched_token = chi
                
        # 5. Enrich text safely on first match
        if matched_token:
            pattern = rf'\b{re.escape(matched_token)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, f"{matched_token}{expansion_str}", text, flags=re.IGNORECASE, count=1)
            else:
                # If matched via Chinese text (no word boundaries), append expansion
                text = f"{text}{expansion_str}"
                
            detected_category = cat
            
    return text, detected_category

def strip_think_tags(text: str) -> str:
    """Strips out DeepSeek-R1 <think>...</think> internal reasoning blocks."""
    if not text:
        return ""
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return cleaned.strip()

def split_text(text: str, max_length: int = 4000) -> list[str]:
    """Splits a long message into chunks, preferring newline boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try splitting at a newline before the limit
        split_point = text.rfind("\n", 0, max_length)
        # Fall back to space if no newline exists
        if split_point == -1:
            split_point = text.rfind(" ", 0, max_length)
        # Fall back to hard cutoff if no space exists
        if split_point == -1:
            split_point = max_length

        chunks.append(text[:split_point].strip())
        text = text[split_point:].strip()

    return chunks


async def send_chunked_message(
    update: Update,
    text: str,
    parse_mode: str = "Markdown",
    reply_to_message_id: int | None = None,
):
    """Sends long text in multiple Telegram messages sequentially."""
    chunks = split_text(text)

    for i, chunk in enumerate(chunks):
        # Only reply to the original user message on the first chunk
        msg_reply_id = reply_to_message_id if i == 0 else None

        try:
            await update.message.reply_text(
                chunk,
                parse_mode=parse_mode,
                reply_to_message_id=msg_reply_id,
            )
        except Exception:
            # Fallback: Send plain text if Markdown formatting breaks across chunk boundaries
            await update.message.reply_text(
                chunk, reply_to_message_id=msg_reply_id
            )
# ==========================================
# WEB & DOCUMENT SCRAPING HELPERS
# ==========================================
def scrape_webpage_and_embedded_docs(url: str) -> List[Document]:
    """
    Scrapes text from a target webpage and automatically detects, downloads,
    and indexes embedded .docx and .pdf documents found on the page.
    """
    scraped_docs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        print(f"🌐 Crawling webpage: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch webpage {url} (HTTP {response.status_code})")
            return scraped_docs

        # 1. Scrape main page content using Jina Reader
        jina_url = f"https://r.jina.ai/{url}"
        jina_resp = requests.get(jina_url, headers=headers, timeout=30)
        if jina_resp.status_code == 200:
            scraped_docs.append(Document(
                page_content=jina_resp.text,
                metadata={"source": url, "category": "Official Webpage"}
            ))
        level = classify_academic_level(url, jina_resp.text)
        scraped_docs.append(Document(
         page_content=jina_resp.text,
         metadata={
            "source": url, 
            "category": "Official Webpage",
            "academic_level": level  # <-- Attached to Qdrant vector payload
        }
        ))

        # 2. Parse HTML to auto-detect embedded .docx and .pdf links with anchor metadata
        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            link_text = a_tag.get_text(strip=True)
            href = a_tag["href"].strip()
            full_doc_url = urljoin(url, href)

            # Auto-process .docx links
            if full_doc_url.lower().endswith(".docx") or ".docx?" in full_doc_url.lower():
                print(f"  └─ 📄 Found embedded DOCX link: {full_doc_url}")
                try:
                    doc_resp = requests.get(full_doc_url, headers=headers, timeout=30)
                    if doc_resp.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                            tmp.write(doc_resp.content)
                            tmp_path = tmp.name
                        
                        loader = Docx2txtLoader(tmp_path)
                        loaded_docx = loader.load()
                        for doc in loaded_docx:
                            doc.metadata["source"] = full_doc_url
                            doc.metadata["link_text"] = link_text or "Embedded Document"
                            doc.metadata["category"] = "Embedded DOCX Document"
                        scraped_docs.extend(loaded_docx)
                        os.unlink(tmp_path)
                except Exception as doc_err:
                    print(f"  ❌ Failed to parse embedded DOCX {full_doc_url}: {doc_err}")

            # Auto-process .pdf links
            elif full_doc_url.lower().endswith(".pdf") or ".pdf?" in full_doc_url.lower():
                print(f"  └─ 📄 Found embedded PDF link: {full_doc_url}")
                try:
                    pdf_resp = requests.get(full_doc_url, headers=headers, timeout=30)
                    if pdf_resp.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_resp.content)
                            tmp_path = tmp.name
                        
                        loader = PyPDFLoader(tmp_path)
                        loaded_pdf = loader.load()
                        for doc in loaded_pdf:
                            doc.metadata["source"] = full_doc_url
                            doc.metadata["link_text"] = link_text or "Embedded Document"
                            doc.metadata["category"] = "Embedded PDF Document"
                        scraped_docs.extend(loaded_pdf)
                        os.unlink(tmp_path)
                except Exception as pdf_err:
                    print(f"  ❌ Failed to parse embedded PDF {full_doc_url}: {pdf_err}")

    except Exception as e:
        print(f"⚠️ Web scraping error for {url}: {e}")

    return scraped_docs

# ==========================================
# RAG CHAIN SETUP & CUSTOM RETRIEVER
# ==========================================
class ScoreInjectingRetriever(BaseRetriever):
    vectorstore: Any = Field(description="The underlying Qdrant vector store")
    k: int = Field(default=10)
    score_threshold: float = Field(default=0.65)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Perform initial vector similarity search
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        ranked_docs = []
        
        query_lower = query.lower()
        # 1. Detect if the user is explicitly asking about Master / Postgraduate studies
        is_pg_query = any(kw in query_lower for kw in ["master", "msc", "postgraduate", "pgd", "pg"])

        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            clamped_score = max(0.0, float(score))
            effective_score = clamped_score
            
            source = str(doc.metadata.get("source", "")).lower()
            content = doc.page_content
            content_lower = content.lower()
            level_meta = str(doc.metadata.get("academic_level", "")).upper()

            # 2. Check for PolyU Subject Code Patterns via Regex
            # ISE1xxx - ISE4xxx = Undergraduate | ISE5xxx - ISE6xxx = Taught Master
            has_ug_code = bool(re.search(r'\bISE[1-4]\d{3}\b', content, re.IGNORECASE))
            has_pg_code = bool(re.search(r'\bISE[5-6]\d{3}\b', content, re.IGNORECASE))

            # 3. Apply Degree Level Boosting & Penalties
            if is_pg_query:
                # Query is for Postgraduate (MSc)
                if has_pg_code or level_meta == "PG":
                    effective_score += 0.25
                if has_ug_code or level_meta == "UG":
                    effective_score -= 0.20
            else:
                # Default / Query is for Undergraduate (BSc / BEng / Degree)
                if has_ug_code or level_meta == "UG":
                    effective_score += 0.25
                if has_pg_code or level_meta == "PG":
                    effective_score -= 0.35  # Heavy penalty for 5000-level Master subjects in UG queries

            # 4. Document Source Type Weights
            if "prd" in source or "programme_def" in source:
                effective_score += 0.15
            elif "student_handbook" in source:
                effective_score -= 0.05

            # 5. High-Value Curriculum Keyword Weights
            if "programme structure" in content_lower:
                effective_score += 0.20
            if "compulsory subjects" in content_lower:
                effective_score += 0.25
            if "elective subjects" in content_lower:
                effective_score += 0.15
            if "credit requirements" in content_lower or "graduation requirements" in content_lower:
                effective_score += 0.20

            # Clamp final effective score between 0.0 and 1.0
            effective_score = min(max(effective_score, 0.0), 1.0)

            # Keep chunk if it passes the score threshold
            if effective_score >= self.score_threshold:
                new_meta = doc.metadata.copy()
                new_meta["_score"] = effective_score
                ranked_docs.append(Document(page_content=doc.page_content, metadata=new_meta))

        # Sort documents by adjusted score in descending order
        ranked_docs.sort(key=lambda d: d.metadata.get("_score", 0), reverse=True)
        result_docs = ranked_docs[:5] if len(ranked_docs) > 5 else ranked_docs

        # Safeguard fallback if vector retrieval confidence is low
        if not result_docs:
            safe_content = (
                "SYSTEM WARNING: No highly relevant internal documents found matching the required degree level. "
                "DO NOT GUESS OR INVENT AN ANSWER. Politely explain that you do not have "
                "the exact curriculum details for this degree level and advise contacting the Academic Registry (AR)."
            )
            return [Document(page_content=safe_content, metadata={"source": "System Safeguard", "_score": 0.0})]

        return result_docs
def get_rag_chain():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    print("🔌 Initializing Ollama & Qdrant Client...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url=ollama_url,
        num_ctx=2048
    )
    
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=120,
        check_compatibility=False
    )
    
    print(f"📡 Checking Qdrant collection: '{COLLECTION_NAME}'...")
    need_indexing = False
    if not client.collection_exists(COLLECTION_NAME):
        need_indexing = True
    else:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            print(f"⚠️ Collection '{COLLECTION_NAME}' exists on Qdrant Cloud but is empty (0 vectors). Triggering re-indexing...")
            need_indexing = True

    if need_indexing:
        print(f"⚙️ Creating/Populating collection '{COLLECTION_NAME}' on Qdrant Cloud...")
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME, 
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        
        all_docs = []
        if os.path.exists(CONFIG_FILE):
            print(f"📖 Reading files and URLs from {CONFIG_FILE}...")
            with open(CONFIG_FILE, "r", encoding="utf-8") as cfg:
                config_data = json.load(cfg)
                urls = config_data.get("urls", [])
                pdf_paths = config_data.get("pdfs", [])
                docx_paths = config_data.get("docx", [])

            # 1. Process target URLs + auto-scrape embedded documents
            if urls:
                for url in urls:
                    extracted_docs = scrape_webpage_and_embedded_docs(url)
                    all_docs.extend(extracted_docs)

            # 2. Process standalone DOCX files
            if docx_paths:
                for docx_url in docx_paths:
                    try:
                        print(f"📄 Downloading standalone DOCX: {docx_url}")
                        resp = requests.get(docx_url, timeout=30)
                        if resp.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                                tmp.write(resp.content)
                                tmp_path = tmp.name
                            loader = Docx2txtLoader(tmp_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = docx_url
                                doc.metadata["category"] = "Standalone DOCX"
                            all_docs.extend(docs)
                            os.unlink(tmp_path)
                    except Exception as e:
                        print(f"⚠️ Standalone DOCX download failed {docx_url}: {e}")

            # 3. Process standalone PDF files
            if pdf_paths:
                for pdf_url in pdf_paths:
                    try:
                        print(f"📄 Downloading PDF: {pdf_url}")
                        response = requests.get(pdf_url, timeout=120)
                        if response.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(response.content)
                                temp_pdf = tmp.name
                            loader = PyPDFLoader(temp_pdf)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = pdf_url
                                doc.metadata["category"] = "Official PDF"
                            all_docs.extend(docs)
                            os.unlink(temp_pdf)
                    except Exception as e:
                        print(f"⚠️ PDF Failed: {pdf_url} - {e}")
        
        if all_docs:
            print("✂️ Chunking documents with RecursiveCharacterTextSplitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
                length_function=len
            )
            splits = text_splitter.split_documents(all_docs)
            print(f"📦 Generated {len(splits)} chunks. Uploading to Qdrant Cloud...")
            
            v_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
            batch_size = 50
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                try: 
                    v_store.add_documents(batch)
                    print(f"  └─ Uploaded batch {i // batch_size + 1}/{(len(splits) - 1) // batch_size + 1}")
                except Exception as e: 
                    print(f"❌ Upload Batch Failed: {e}")

    print("🧠 Initializing LangChain RAG pipeline...")
    v_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    global vector_store
    vector_store = v_store

    # Forced num_ctx=4096 to prevent Ollama from allocating 128k context and offloading to CPU
    llm = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0.2, num_ctx=4096)
    retriever = ScoreInjectingRetriever(vectorstore=v_store, k=10, score_threshold=0.70)

    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, rewrite it into a standalone search query. "
        "Return ONLY the rewritten search query."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # Advanced System Prompt for PolyU Academic Advisor persona
    system_prompt = (
        "You are Alex, a warm, highly structured, and professional Academic Advisor for the Department of "
        "Industrial and Systems Engineering (ISE) at Hong Kong Polytechnic University (PolyU).\n\n"

        "DEPARTMENT BOUNDARY & DISAMBIGUATION RULES:\n"
        "1. EXTERNAL DEPARTMENTS (e.g., AAE, ME, COMP, EEE):\n"
        "   - You ONLY represent the Department of Industrial and Systems Engineering (ISE).\n"
        "   - 'AAE' stands for the Department of Aeronautical and Aviation Engineering, which is an INDEPENDENT department from ISE.\n"
        "   - If a student states they are from AAE (or another external department), politely clarify that AAE is an independent department from ISE. Mention that while you can answer general PolyU requirements (GUR/CAR/WIE), specific AAE programme queries should be directed to the AAE General Office.\n"
        "   - NEVER claim that AAE is under or part of the ISE Department.\n\n"

        "OFFICIAL ISE PROGRAMME CATALOGUE:\n"
        "2. UNDERGRADUATE (UG) DEGREES:\n"
        "   - BSc (Hons) in Aviation Operations and Systems (AOS)\n"
        "   - BSc (Hons) in Logistics Engineering with Management (LEM)\n"
        "   - BSc (Hons) in Enterprise Engineering with Management (EEM)\n"
        "   - BEng (Hons) in Product Engineering with Marketing (PEM)\n"
        "   - BEng (Hons) in Industrial and Systems Engineering (ISE)\n"
        "   - BEng (Hons) Scheme in Product and Industrial Engineering (PIE)\n\n"

        "3. TAUGHT POSTGRADUATE (TPg) & RESEARCH PROGRAMMES:\n"
        "   - MSc in Knowledge and Technology Management (KTM)\n"
        "   - MSc in Industrial Logistics Systems (ILS)\n"
        "   - MSc in Smart Manufacturing (SM)\n"
        "   - MSc in Information Systems in Xi'an\n"
        "   - Integrated Graduate Development Scheme (IGDS): MSc in Engineering Business Management (EBM) & MSc in Supply Chain and Logistics Management (SCLM)\n"
        "   - Engineering Doctorate (EngD)\n"
        "   - Research Postgraduate Programmes (MPhil / PhD)\n\n"

        "CRITICAL DEGREE LEVEL & SUBJECT CODE RULES:\n"
        "4. POLYU SUBJECT CODE LEVELS:\n"
        "   - Level 1 to Level 4 series (e.g., ISE1xxx, ISE2xxx, ISE3xxx, ISE4xxx) are strictly UNDERGRADUATE (BSc / BEng Degree).\n"
        "   - Level 5 to Level 6 series (e.g., ISE5xxx, ISE6xxx) are strictly POSTGRADUATE (Taught Master / MSc / EngD / PhD).\n\n"

        "5. UNDERGRADUATE vs MASTER DISAMBIGUATION:\n"
        "   - When answering questions about Degree/Undergraduate programmes (e.g., EEM, ISE, LEM, PEM, AOS, PIE), ONLY include Level 1 to Level 4 subjects.\n"
        "   - NEVER mix Level 5–6 Master subjects into an Undergraduate degree response unless explicitly tagged as an approved cross-level elective.\n"
        "   - If a student query is ambiguous (e.g., 'What is EEM?', 'What is LEM?', 'What is PIE?', 'What is WIE?', 'ISE degree'), specify that you are providing the Undergraduate Degree curriculum by default, or ask if they mean Master (MSc).\n\n"

        "6. SPECIFIC ACRONYM RULES:\n"
        "   - In PolyU, 'WIE' strictly stands for 'Work-Integrated Education' (校企協作教育/實習). Politely correct any claim otherwise.\n\n"

        "ANSWERING & CITATION REQUIREMENTS:\n"
        "7. Base your answers ONLY on the provided context. ALWAYS include exact Subject Codes (e.g., ISE3013) and Credit Values.\n"
        "8. NEVER tell students to 'check the handbook' or 'contact AR' unless system safeguards explicitly instruct you to do so.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))
    return chain, v_store

async def run_rag_query(query_text: str) -> str:
    """Invokes the local RAG chain with timeout safeguard and strips DeepSeek-R1 think tags."""
    global rag_chain
    if rag_chain is None:
        return "⏳ Alex 正在啟動知識庫與本地 AI 模型，請稍等約 30 秒再試！"
    
    enriched_input, _ = apply_index_corrections(query_text)
    try:
        # 90-second timeout safeguard
        result = await asyncio.wait_for(
            rag_chain.ainvoke({"input": enriched_input, "chat_history": []}),
            timeout=90.0
        )
        raw_answer = result.get("answer", "抱歉，我無法檢索到相關解答。")
        return strip_think_tags(raw_answer)
    except asyncio.TimeoutError:
        print("⚠️ RAG Query Timed Out (Ollama response took > 90 seconds)")
        return "⚠️ 本地 AI 模型 (Ollama) 運算逾時，請確認系統資源是否過載或再次重試。"
    except Exception as e:
        print(f"❌ RAG Execution Error: {e}")
        return f"抱歉，系統運算時發生技術故障：{e}"

# ==========================================
# TELEGRAM BOT HANDLERS & NAVIGATION
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start command with faculty selection keyboard."""
    chat_id = update.effective_chat.id
    update_student_session(chat_id)
    
    keyboard = [
        [InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)", callback_data="faculty_ise")],
        [InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)", callback_data="faculty_gur")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = (
        f"👋 歡迎使用香港理工大學 (PolyU) 學術諮詢 AI 助手 (Alex)！\n\n"
        "你可以隨時向我查詢 ISE 學科要求、WIE 實習指引及學術規條。\n"
        "請選擇你所屬的學系或你想諮詢的範疇："
    )
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command explaining knowledge domain."""
    help_text = (
        "📚 **Alex Knowledge Base & Advice Scope**\n\n"
        "I am trained on official guidelines from the Department of Industrial and "
        "Systems Engineering (ISE) at PolyU.\n\n"
        "**Example questions you can ask me:**\n"
        "• *\"What are the WIE requirements for ISE students?\"*\n"
        "• *\"How do I fulfill my Capstone Project prerequisites?\"*\n"
        "• *\"What is the credit distribution for EEM core subjects?\"*\n\n"
        "🔄 **Start fresh:** Type `/clear` anytime to reset conversation memory."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /clear command to reset SQLite conversation memory."""
    user_id = str(update.effective_user.id)
    success = clear_user_history(user_id)
    
    if success:
        message = "🔄 **Session reset!** Your chat history with Alex has been cleared. What would you like to discuss next?"
    else:
        message = "⚠️ Could not reset session history right now, but you can continue asking questions!"
        
    await update.message.reply_text(message, parse_mode="Markdown")

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    if query.data == "faculty_ise":
        update_student_session(chat_id, faculty="ISE")
        keyboard = [
            [InlineKeyboardButton("📋 CAR / GUR 學分要求", callback_data="ise_car")],
            [InlineKeyboardButton("💼 WIE 實習 / 課外活動要求", callback_data="ise_wie")],
            [InlineKeyboardButton("🎓 Capstone 畢業論文選題", callback_data="ise_capstone")],
            [InlineKeyboardButton("🔙 返回主選單", callback_data="go_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="📍 **你已進入 ISE 學術諮詢專區**\n請選擇你想了解的疑問範疇，或直接在下方**輸入文字**向我提問：",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
    elif query.data == "ise_car":
        update_student_session(chat_id)
        await query.edit_message_text(text="🔍 正在為你檢索 PolyU CAR 要求，請稍候...")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        response = await run_rag_query("請問 ISE 學生 CAR 和 GUR 的學分要求是什麼？")
        try:
            await query.delete_message()
        except Exception:
            pass
        await send_chunked_message(update, response, parse_mode="Markdown")
        
    elif query.data == "ise_wie":
        update_student_session(chat_id)
        await query.edit_message_text(text="🔍 正在為你檢索 WIE 實習要求，請稍候...")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        response = await run_rag_query("ISE 的 WIE 實習 (Work-Integrated Education) 有什麼要求？")
        try:
            await query.delete_message()
        except Exception:
            pass
        await send_chunked_message(update, response, parse_mode="Markdown")
        
    elif query.data == "ise_capstone":
        update_student_session(chat_id)
        await query.edit_message_text(
            text="🎓 **[ISE Capstone 提示]**\n"
                 "ISE 的畢業論文 (Capstone Project) 通常在 Year 3 下學期開始選題，涵蓋智能製造、物流管理及數據分析等範疇。"
                 "如需查詢選題指引，請直接在此輸入具體問題！"
        )

    elif query.data == "go_main":
        update_student_session(chat_id, faculty="General")
        keyboard = [
            [InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)", callback_data="faculty_ise")],
            [InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)", callback_data="faculty_gur")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("請選擇你所屬的學系或你想諮詢的範疇：", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    student_text = update.message.text
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name or "Student"
    
    update_student_session(chat_id)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    placeholder_msg = await update.message.reply_text("🤔 Alex 正在思考並查閱 PolyU 學術指引...")
    
    start_time = time.perf_counter()
    ai_response = await run_rag_query(student_text)
    elapsed = time.perf_counter() - start_time
    
    print(f"⏱️ [Latency Log] User: {user_name} ({chat_id}) | Query: '{student_text}' | Latency: {elapsed:.2f}s")
    
    # 1. Delete placeholder message
    try:
        await placeholder_msg.delete()
    except Exception:
        pass
        
    # 2. Send response safely using chunking logic
    await send_chunked_message(update, ai_response, parse_mode="Markdown")

# ==========================================
# FASTAPI LIFESPAN & APPLICATION STARTUP
# ==========================================
def load_rag_in_background():
    global rag_chain, vector_store
    print("🚀 Loading RAG Vector Database & Ollama Model in background...")
    try:
        rag_chain, vector_store = get_rag_chain()
        print("✅ RAG Chain is fully loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to load RAG chain: {e}")
        

def classify_academic_level(url: str, text: str) -> str:
    """Classifies content into Undergraduate (UG) or Postgraduate (PG)."""
    url_lower = url.lower()
    text_lower = text.lower()
    
    if any(k in url_lower for k in ["undergraduate", "beng", "bsc", "ug"]) or \
       any(k in text_lower for k in ["bachelor of", "bsc (hons)", "beng (hons)"]):
        return "UG"
    elif any(k in url_lower for k in ["postgraduate", "msc", "master", "pg"]) or \
         any(k in text_lower for k in ["master of", "msc in", "postgraduate scheme"]):
        return "PG"
    
    return "General"

async def start_telegram_bot():
    global tg_app
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️ No valid TELEGRAM_BOT_TOKEN set in .env. Skipping Telegram setup.")
        return

    tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register Handlers
    tg_app.add_handler(CommandHandler("start", start_command))
    tg_app.add_handler(CommandHandler("help", help_command))
    tg_app.add_handler(CommandHandler("clear", clear_command))
    tg_app.add_handler(CallbackQueryHandler(button_click))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await tg_app.initialize()
    await tg_app.start()
    await tg_app.updater.start_polling()
    print("✅ Telegram Bot polling started successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sqlite_db()
    
    # Load RAG DB & Ollama model asynchronously
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_rag_in_background)
    
    await start_telegram_bot()
    
    yield
    
    if tg_app:
        print("Shutting down Telegram Bot...")
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()

app = FastAPI(title="PolyU AI Academic Advisor", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    return "<h1>PolyU AI Academic Advisor (Alex) backend is running!</h1><p>Telegram Bot active.</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)