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

# FastAPI Imports
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Telegram Imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
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
COLLECTION_NAME = "polyu_advisor_semantic_v2"
DB_FILE = "polyu_advisor.db"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Make sure to set WEBAPP_URL in your .env (e.g., https://your-domain.ngrok-free.app)
WEBAPP_URL = os.getenv("WEBAPP_URL")

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
    if not text or not TERM_INDEX:
        return text, "General"
        
    detected_category = "General"
    
    for key, data in TERM_INDEX.items():
        if not isinstance(data, dict):
            pattern = rf'\b{re.escape(key)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, f"{key} ({data})", text, flags=re.IGNORECASE, count=1)
            continue
            
        eng = data.get("english", key)
        chi = data.get("chinese", "")
        cat = data.get("category", "General")
        abbr = data.get("abbreviation", key)
        prog_code = data.get("programme_code", "")
        jupas_code = data.get("jupas_code", "")
        
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
                
        if matched_token:
            pattern = rf'\b{re.escape(matched_token)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, f"{matched_token}{expansion_str}", text, flags=re.IGNORECASE, count=1)
            else:
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
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        split_point = text.rfind("\n", 0, max_length)
        if split_point == -1:
            split_point = text.rfind(" ", 0, max_length)
        if split_point == -1:
            split_point = max_length

        chunks.append(text[:split_point].strip())
        text = text[split_point:].strip()

    return chunks

async def send_chunked_message(update: Update, text: str, parse_mode: str = "Markdown", reply_to_message_id: int | None = None):
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        msg_reply_id = reply_to_message_id if i == 0 else None
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode, reply_to_message_id=msg_reply_id)
        except Exception:
            await update.message.reply_text(chunk, reply_to_message_id=msg_reply_id)

# ==========================================
# WEB & DOCUMENT SCRAPING HELPERS
# ==========================================
def classify_academic_level(url: str, text: str) -> str:
    url_lower = url.lower()
    text_lower = text.lower()
    
    if any(k in url_lower for k in ["undergraduate", "beng", "bsc", "ug"]) or \
       any(k in text_lower for k in ["bachelor of", "bsc (hons)", "beng (hons)"]):
        return "UG"
    elif any(k in url_lower for k in ["postgraduate", "msc", "master", "pg"]) or \
         any(k in text_lower for k in ["master of", "msc in", "postgraduate scheme"]):
        return "PG"
    return "General"

def fetch_via_jina(url: str) -> str | None:
    """Fetches any URL (Webpage, PDF, DOCX) through Jina AI Reader and returns Markdown text."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = requests.get(jina_url, headers=headers, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
    except Exception as e:
        print(f"⚠️ Jina fetch failed for {url}: {e}")
    return None

def scrape_webpage_and_embedded_docs(url: str) -> List[Document]:
    """Scrapes a webpage and its embedded PDF/DOCX files, all via Jina AI Reader."""
    scraped_docs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    print(f"🌐 [Jina] Scraping main webpage: {url}")
    main_text = fetch_via_jina(url)
    if main_text:
        level = classify_academic_level(url, main_text)
        scraped_docs.append(Document(
            page_content=main_text,
            metadata={"source": url, "category": "Official Webpage", "academic_level": level}
        ))

    # Parse raw HTML to auto-discover embedded PDF / DOCX document links
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                link_text = a_tag.get_text(strip=True)
                href = a_tag["href"].strip()
                full_doc_url = urljoin(url, href)
                
                # Check for embedded PDF or DOCX links
                is_pdf = full_doc_url.lower().endswith(".pdf") or ".pdf?" in full_doc_url.lower()
                is_docx = full_doc_url.lower().endswith(".docx") or ".docx?" in full_doc_url.lower()

                if is_pdf or is_docx:
                    doc_type = "PDF" if is_pdf else "DOCX"
                    print(f"  └─ 📄 [Jina] Found embedded {doc_type}: {full_doc_url}")
                    
                    doc_text = fetch_via_jina(full_doc_url)
                    if doc_text:
                        level = classify_academic_level(full_doc_url, doc_text)
                        scraped_docs.append(Document(
                            page_content=doc_text,
                            metadata={
                                "source": full_doc_url,
                                "link_text": link_text or "Embedded Document",
                                "category": f"Embedded {doc_type} Document",
                                "academic_level": level
                            }
                        ))
    except Exception as e:
        print(f"⚠️ Error scanning embedded links on {url}: {e}")

    return scraped_docs

# ==========================================
# RAG CHAIN SETUP & CUSTOM RETRIEVER
# ==========================================
class ScoreInjectingRetriever(BaseRetriever):
    vectorstore: Any = Field(description="The underlying Qdrant vector store")
    k: int = Field(default=20)
    score_threshold: float = Field(default=0.85)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        ranked_docs = []
        
        query_lower = query.lower()
        is_pg_query = any(kw in query_lower for kw in ["master", "msc", "postgraduate", "pgd", "pg"])
        
        # 🆕 Detect if user is asking about curriculum / 4-year study scheme
        is_curriculum_query = any(kw in query_lower for kw in [
            "curriculum", "study pattern", "4-year", "four year", "4 year", 
            "programme structure", "progression", "study scheme", "subjects by year"
        ])
        is_wie_query = any(kw in query_lower for kw in ["wie", "work-integrated","work-integrated education", "internship", "實習", "校企協作"])
        
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            clamped_score = max(0.0, float(score))
            effective_score = clamped_score
            
            source = str(doc.metadata.get("source", "")).lower()
            content = doc.page_content
            content_lower = content.lower()
            level_meta = str(doc.metadata.get("academic_level", "")).upper()

            has_ug_code = bool(re.search(r'\bISE[1-4]\d{3}\b', content, re.IGNORECASE))
            has_pg_code = bool(re.search(r'\bISE[5-6]\d{3}\b', content, re.IGNORECASE))

            if is_pg_query:
                if has_pg_code or level_meta == "PG":
                    effective_score += 0.25
                if has_ug_code or level_meta == "UG":
                    effective_score -= 0.20
            else:
                if has_ug_code or level_meta == "UG":
                    effective_score += 0.25
                if has_pg_code or level_meta == "PG":
                    effective_score -= 0.35

            if "prd" in source or "programme_def" in source or ".pdf" in source:
                effective_score += 0.20

            if is_curriculum_query:
                if any(kw in content_lower for kw in ["year 1", "year 2", "year 3", "year 4", "semester 1", "semester 2"]):
                    effective_score += 0.30
                if "programme structure" in content_lower or "curriculum" in content_lower:
                    effective_score += 0.25

            if "compulsory subjects" in content_lower or "core subjects" in content_lower:
                effective_score += 0.15
            if "credit requirements" in content_lower or "graduation requirements" in content_lower:
                effective_score += 0.15
            
            if is_wie_query:
                if "wie" in source or "wie_official" in source:
                    effective_score += 0.35
                if "work-integrated education" in content_lower or "100 hours" in content_lower:
                    effective_score += 0.25

            effective_score = min(max(effective_score, 0.0), 1.0)

            if effective_score >= self.score_threshold:
                new_meta = doc.metadata.copy()
                new_meta["_score"] = effective_score
                ranked_docs.append(Document(page_content=doc.page_content, metadata=new_meta))

        ranked_docs.sort(key=lambda d: d.metadata.get("_score", 0), reverse=True)
        result_docs = ranked_docs[:8] if len(ranked_docs) > 8 else ranked_docs

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
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url, num_ctx=2048)
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120, check_compatibility=False)
    
    need_indexing = False
    if not client.collection_exists(COLLECTION_NAME):
        need_indexing = True
    else:
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            need_indexing = True

    if need_indexing:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME, 
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        
        all_docs = []
        if os.path.exists(CONFIG_FILE):
            print(f"📖 Reading target links from {CONFIG_FILE}...")
            with open(CONFIG_FILE, "r", encoding="utf-8") as cfg:
                config_data = json.load(cfg)
                urls = config_data.get("urls", [])
                pdf_paths = config_data.get("pdfs", [])
                docx_paths = config_data.get("docx", [])

            # 1. Process target Webpage URLs + auto-discover embedded docs
            if urls:
                for url in urls:
                    extracted_docs = scrape_webpage_and_embedded_docs(url)
                    all_docs.extend(extracted_docs)

            # 2. Process standalone PDF links directly via Jina Reader
            if pdf_paths:
                for pdf_url in pdf_paths:
                    print(f"📄 [Jina] Scraping standalone PDF: {pdf_url}")
                    pdf_text = fetch_via_jina(pdf_url)
                    if pdf_text:
                        level = classify_academic_level(pdf_url, pdf_text)
                        all_docs.append(Document(
                            page_content=pdf_text,
                            metadata={
                                "source": pdf_url,
                                "category": "Official PDF",
                                "academic_level": level
                            }
                        ))

            # 3. Process standalone DOCX links directly via Jina Reader
            if docx_paths:
                for docx_url in docx_paths:
                    print(f"📄 [Jina] Scraping standalone DOCX: {docx_url}")
                    docx_text = fetch_via_jina(docx_url)
                    if docx_text:
                        level = classify_academic_level(docx_url, docx_text)
                        all_docs.append(Document(
                            page_content=docx_text,
                            metadata={
                                "source": docx_url,
                                "category": "Standalone DOCX",
                                "academic_level": level
                            }
                        ))
        
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=250, length_function=len)
            splits = text_splitter.split_documents(all_docs)
            v_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
            batch_size = 50
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                try: 
                    v_store.add_documents(batch)
                except Exception as e: 
                    print(f"❌ Upload Batch Failed: {e}")

    v_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    global vector_store
    vector_store = v_store

    llm = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0.2, num_ctx=16384)
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
        "6. SPECIFIC ACRONYM & WIE RULES:\n"
        "   - In PolyU, 'WIE' strictly stands for 'Work-Integrated Education' (校企協作教育/實習).\n"
        "   - CRITICAL: THERE ARE NO TAUGHT WIE COURSES/SUBJECTS FOR ISE STUDENTS. WIE is a non-credit training graduation requirement fulfilled through practical work experience / industrial placements (minimum required hours), NOT an academic classroom subject.\n"
        "   - Politely clarify to students that they do not enroll in a WIE class, but rather complete approved internship placements and submit evaluation logbooks.\n\n"
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
    global rag_chain
    if rag_chain is None:
        return "⏳ Alex 正在啟動知識庫與本地 AI 模型，請稍等約 30 秒再試！"
    
    enriched_input, _ = apply_index_corrections(query_text)
    try:
        result = await asyncio.wait_for(
            rag_chain.ainvoke({"input": enriched_input, "chat_history": []}),
            timeout=90.0
        )
        raw_answer = result.get("answer", "抱歉，我無法檢索到相關解答。")
        return strip_think_tags(raw_answer)
    except asyncio.TimeoutError:
        return "⚠️ 本地 AI 模型 (Ollama) 運算逾時，請確認系統資源是否過載或再次重試。"
    except Exception as e:
        return f"抱歉，系統運算時發生技術故障：{e}"

# ==========================================
# TELEGRAM BOT HANDLERS & NAVIGATION
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start command, displaying Telegram Web App Launch Button."""
    chat_id = update.effective_chat.id
    update_student_session(chat_id)
    
    # URL pointing to the FastAPI endpoint hosting the WebApp
    app_url = f"{WEBAPP_URL}/webapp"
    
    keyboard = [
        [InlineKeyboardButton("🚀 啟動 Academic Advisor Web App", web_app=WebAppInfo(url=app_url))],
        [InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)", callback_data="faculty_ise")],
        [InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)", callback_data="faculty_gur")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = (
        f"👋 歡迎使用香港理工大學 (PolyU) 學術諮詢 AI 助手 (Alex)！\n\n"
        "點擊下方按鈕啟動全新的 **Telegram Web App ( Mini App )**，或直接在聊天室點選選題或提問："
    )
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    user_id = str(update.effective_user.id)
    success = clear_user_history(user_id)
    message = "🔄 **Session reset!** Your chat history has been cleared." if success else "⚠️ Could not reset session history."
    await update.message.reply_text(message, parse_mode="Markdown")

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    if query.data == "faculty_ise":
        update_student_session(chat_id, faculty="ISE")
        keyboard = [
            [InlineKeyboardButton("📋 CAR / GUR 學分要求", callback_data="ise_car")],
            [InlineKeyboardButton("💼 WIE 實習要求 (按課程選擇)", callback_data="ise_wie_menu")],
            [InlineKeyboardButton("🔙 返回主選單", callback_data="go_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="📍 **你已進入 ISE 學術諮詢專區**\n請選擇你想了解的疑問範疇，或直接在下方**輸入文字**向我提問：",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
    elif query.data == "ise_wie_menu":
        update_student_session(chat_id)
        keyboard = [
            [InlineKeyboardButton("BSc LEM (物流工程兼管理)", callback_data="wie_prog_lem")],
            [InlineKeyboardButton("BSc EEM (企業工程兼管理)", callback_data="wie_prog_eem")],
            [InlineKeyboardButton("BEng PEM (產品工程兼營銷)", callback_data="wie_prog_pem")],
            [InlineKeyboardButton("BEng ISE (工業及系統工程)", callback_data="wie_prog_ise")],
            [InlineKeyboardButton("BSc AOS (航空運算及系統學)", callback_data="wie_prog_aos")],
            [InlineKeyboardButton("BEng SM (智能製造)", callback_data="wie_prog_sm")],
            [InlineKeyboardButton("BEng PISM (產品創新及智能製造)", callback_data="wie_prog_pism")],
            [InlineKeyboardButton("BEng PIM (產品創新兼市場學)", callback_data="wie_prog_pim")],
            [InlineKeyboardButton("BEng ISCEM (智能供應鏈及工程管理)", callback_data="wie_prog_iscem")],
            [InlineKeyboardButton("BEng ISC (智能供應鏈)", callback_data="wie_prog_isc")],
            [InlineKeyboardButton("BEng EMDA (工程管理兼數據分析)", callback_data="wie_prog_emda")],
            [InlineKeyboardButton("🔙 返回 ISE 選單", callback_data="faculty_ise")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="💼 **請選擇你所修讀的專修課程 (Programme)：**\n不同專修課程的 WIE 實習安排可能有所不同。",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    elif query.data.startswith("wie_prog_"):
        prog_code = query.data.replace("wie_prog_", "").upper()
        prog_names = {
            "LEM": "BSc (Hons) in Logistics Engineering with Management (LEM)",
            "EEM": "BSc (Hons) in Enterprise Engineering with Management (EEM)",
            "PEM": "BEng (Hons) in Product Engineering with Marketing (PEM)",
            "ISE": "BEng (Hons) in Industrial and Systems Engineering (ISE)",
            "AOS": "BSc (Hons) in Aviation Operations and Systems (AOS)",
            "SM":  "BEng (Hons) in Smart Manufacturing (SM)",
            "PIM": "BEng (Hons) in Product Innovation with Marketing (PIM)",
            "PISM": "BEng (Hons) Scheme in Product Innovation and Smart Manufacturing (PISM)",
            "ISCEM": "BSc (Hons) Scheme in Intelligent Supply Chain and Engineering Management (ISCEM)",
            "ISC": "Bachelor of Science (Honours) in Intelligent Supply Chain (ISC)",
            "EMDA": "BEng (Hons) Scheme in Product and Industrial Engineering (EMDA)"
            
        }
        full_prog = prog_names.get(prog_code, prog_code)

        await query.edit_message_text(text=f"🔍 正在為你檢索 {prog_code} 的 WIE 實習要求，請稍候...")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        prompt_query = f"請問 PolyU ISE 的 {full_prog} 課程，WIE (Work-Integrated Education) 實習要求的具體內容是什麼？"
        response = await run_rag_query(prompt_query)
        try:
            await query.delete_message()
        except Exception:
            pass
        await send_chunked_message(update, response, parse_mode="Markdown")

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

    elif query.data == "go_main":
        update_student_session(chat_id, faculty="General")
        app_url = f"{WEBAPP_URL}/webapp"
        keyboard = [
            [InlineKeyboardButton("🚀 啟動 Academic Advisor Web App", web_app=WebAppInfo(url=app_url))],
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
    
    try:
        await placeholder_msg.delete()
    except Exception:
        pass
        
    await send_chunked_message(update, ai_response, parse_mode="Markdown")

def fetch_via_jina(url: str) -> str | None:
    """Fetches any URL via Jina AI Reader and logs character count + text preview."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    jina_url = f"https://r.jina.ai/{url}"
    
    print(f"🌐 [Jina Requesting] {url}")
    try:
        resp = requests.get(jina_url, headers=headers, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            char_count = len(resp.text)
            # Take a 120-character preview of the scraped Markdown text
            preview = resp.text[:120].replace("\n", " ")
            
            print(f"✅ [Jina Scraped Success] {url}")
            print(f"   ├─ Extracted: {char_count:,} characters")
            print(f"   └─ Preview: \"{preview}...\"\n")
            return resp.text
        else:
            print(f"⚠️ [Jina HTTP {resp.status_code}] Could not scrape {url}\n")
    except Exception as e:
        print(f"❌ [Jina Fetch Error] {url}: {e}\n")
    return None
# ==========================================
# FASTAPI BACKGROUND SETUP & ROUTING
# ==========================================
def load_rag_in_background():
    global rag_chain, vector_store
    print("🚀 Loading RAG Vector Database & Ollama Model in background...")
    try:
        rag_chain, vector_store = get_rag_chain()
        print("✅ RAG Chain is fully loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to load RAG chain: {e}")

async def start_telegram_bot():
    global tg_app
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️ No valid TELEGRAM_BOT_TOKEN set in .env. Skipping Telegram setup.")
        return

    tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
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
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_rag_in_background)
    await start_telegram_bot()
    yield
    if tg_app:
        print("Shutting down Telegram Bot...")
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()

app = FastAPI(title="PolyU AI Academic Advisor WebApp", lifespan=lifespan)

# Allow Cross-Origin Requests for WebApp integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REST API & TELEGRAM WEB APP ENDPOINTS
# ==========================================
class ChatRequest(BaseModel):
    chat_id: str | None = None
    message: str

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """API Endpoint called directly by Telegram Web App JavaScript UI."""
    if req.chat_id:
        update_student_session(req.chat_id)
    answer = await run_rag_query(req.message)
    return {"status": "success", "response": answer}

@app.get("/webapp", response_class=HTMLResponse)
async def get_web_app():
    """Serves the Web App interface designed for Telegram WebApp environment."""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-HK">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>PolyU Academic Advisor</title>
        <!-- Telegram Web App JavaScript SDK -->
        <script src="https://telegram.org/js/telegram-web-app.js"></script>
        <style>
            :root {
                --bg-color: var(--tg-theme-bg-color, #f4f4f7);
                --text-color: var(--tg-theme-text-color, #1a1a1a);
                --hint-color: var(--tg-theme-hint-color, #8e8e93);
                --button-color: var(--tg-theme-button-color, #007aff);
                --button-text-color: var(--tg-theme-button-text-color, #ffffff);
                --secondary-bg: var(--tg-theme-secondary-bg-color, #ffffff);
            }

            body {
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                display: flex;
                flex-direction: column;
                height: 100vh;
                overflow: hidden;
            }

            .header {
                padding: 12px 16px;
                background-color: var(--secondary-bg);
                border-bottom: 1px solid rgba(0, 0, 0, 0.08);
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }

            .header img {
                width: 36px;
                height: 36px;
                border-radius: 50%;
            }

            .header-info h2 {
                margin: 0;
                font-size: 15px;
                font-weight: 600;
            }

            .header-info p {
                margin: 2px 0 0;
                font-size: 12px;
                color: var(--hint-color);
            }

            .quick-topics {
                display: flex;
                gap: 8px;
                padding: 8px 12px;
                overflow-x: auto;
                background-color: var(--secondary-bg);
                border-bottom: 1px solid rgba(0,0,0,0.05);
            }

            .chip {
                background-color: var(--bg-color);
                color: var(--text-color);
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 16px;
                padding: 6px 12px;
                font-size: 12px;
                white-space: nowrap;
                cursor: pointer;
            }

            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .message {
                max-width: 80%;
                padding: 10px 14px;
                border-radius: 16px;
                font-size: 14px;
                line-height: 1.45;
                word-wrap: break-word;
                white-space: pre-wrap;
            }

            .message.user {
                align-self: flex-end;
                background-color: var(--button-color);
                color: var(--button-text-color);
                border-bottom-right-radius: 4px;
            }

            .message.bot {
                align-self: flex-start;
                background-color: var(--secondary-bg);
                color: var(--text-color);
                border-bottom-left-radius: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.06);
            }

            .input-container {
                padding: 10px 12px;
                background-color: var(--secondary-bg);
                border-top: 1px solid rgba(0,0,0,0.08);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .input-container input {
                flex: 1;
                border: 1px solid rgba(0, 0, 0, 0.12);
                border-radius: 20px;
                padding: 10px 14px;
                font-size: 14px;
                background-color: var(--bg-color);
                color: var(--text-color);
                outline: none;
            }

            .input-container button {
                background-color: var(--button-color);
                color: var(--button-text-color);
                border: none;
                border-radius: 50%;
                width: 38px;
                height: 38px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-weight: bold;
            }

            .loading-dots {
                display: flex;
                gap: 4px;
            }
            .dot {
                width: 6px;
                height: 6px;
                background-color: var(--hint-color);
                border-radius: 50%;
                animation: pulse 1.2s infinite ease-in-out;
            }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }

            @keyframes pulse {
                0%, 100% { opacity: 0.3; transform: scale(0.8); }
                50% { opacity: 1; transform: scale(1.1); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-info">
                <h2>PolyU Academic Advisor (Alex)</h2>
                <p>Industrial and Systems Engineering (ISE)</p>
            </div>
        </div>

        <div class="quick-topics">
            <div class="chip" onclick="sendQuickQuery('CAR/GUR 學分要求是什麼？')">📋 CAR/GUR 要求</div>
            <div class="chip" onclick="sendQuickQuery('LEM 課程的 WIE 實習要求是什麼？')">💼 LEM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('EEM 課程的 WIE 實習要求是什麼？')">💼 EEM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('ISE 課程的 WIE 實習要求是什麼？')">💼 ISE WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('AOS 課程的 WIE 實習要求是什麼？')">💼 AOS WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('PEM 課程的 WIE 實習要求是什麼？')">💼 PEM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('SM 課程的 WIE 實習要求是什麼？')">💼 SM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('PIM 課程的 WIE 實習要求是什麼？')">💼 PIM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('EMDA 課程的 WIE 實習要求是什麼？')">💼 EMDA WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('ISC 課程的 WIE 實習要求是什麼？')">💼 ISC WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('ISCEM 課程的 WIE 實習要求是什麼？')">💼 ISCEM WIE 要求</div>
            <div class="chip" onclick="sendQuickQuery('PISM 課程的 WIE 實習要求是什麼？')">💼 PISM WIE 要求</div>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message bot">👋 Hello! 我係 Alex，PolyU ISE 學術諮詢助手。請點選上方快捷選項或直接在下方輸入問題向我查詢！</div>
        </div>

        <div class="input-container">
            <input type="text" id="userInput" placeholder="輸入你的問題..." onkeypress="handleKeyPress(event)" />
            <button onclick="sendMessage()">➔</button>
        </div>

        <script>
            // Initialize Telegram WebApp SDK
            const tg = window.Telegram.WebApp;
            tg.expand();

            let chatId = tg.initDataUnsafe?.user?.id || "webapp_user";

            function appendMessage(text, sender) {
                const container = document.getElementById("chatContainer");
                const msgDiv = document.createElement("div");
                msgDiv.className = `message ${sender}`;
                msgDiv.innerText = text;
                container.appendChild(msgDiv);
                container.scrollTop = container.scrollHeight;
                return msgDiv;
            }

            function appendLoading() {
                const container = document.getElementById("chatContainer");
                const msgDiv = document.createElement("div");
                msgDiv.className = "message bot";
                msgDiv.id = "loadingBubble";
                msgDiv.innerHTML = `<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
                container.appendChild(msgDiv);
                container.scrollTop = container.scrollHeight;
            }

            function removeLoading() {
                const loader = document.getElementById("loadingBubble");
                if (loader) loader.remove();
            }

            async function sendMessage() {
                const input = document.getElementById("userInput");
                const text = input.value.trim();
                if (!text) return;

                appendMessage(text, "user");
                input.value = "";
                appendLoading();

                try {
                    const response = await fetch("/api/chat", {
                        method: "POST",
                        headers: { 
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true" // 👈 Add this line to bypass ngrok warning!
    },
                        body: JSON.stringify({ chat_id: String(chatId), message: text })
                    });
                    const data = await response.json();
                    removeLoading();
                    appendMessage(data.response, "bot");
                } catch (err) {
                    removeLoading();
                    appendMessage("⚠️ 網絡連線錯誤，請稍後重試。", "bot");
                }
            }

            function sendQuickQuery(text) {
                document.getElementById("userInput").value = text;
                sendMessage();
            }

            function handleKeyPress(e) {
                if (e.key === 'Enter') sendMessage();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/")
async def get_chat_page():
    return "<h1>PolyU AI Academic Advisor Backend Active</h1><p>Visit /webapp to access Telegram Mini App interface.</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)