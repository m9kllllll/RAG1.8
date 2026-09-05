# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import warnings

# --- Suppress PyTorch Dynamo and Transformer warnings ---
os.environ["TORCH_LOGS"] = "-dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.fx")

import json
import tempfile
import uvicorn
import asyncio
import sqlite3
import re
import nltk
import time
import httpx
import logging
import hashlib
import urllib.parse
from urllib.parse import urldefrag
from typing import List, Dict, Tuple, Any, Optional, Set
from contextlib import asynccontextmanager

from bs4 import BeautifulSoup
import pdfplumber
from pypdf import PdfReader
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Telegram Imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.constants import ChatAction
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# LangChain & Qdrant Imports
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter)

# Optional Advanced PDF Loaders
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Standard LangChain chain factory functions
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# RAG upgrades: intelligent chunking, contextual metadata, image captioning,
# Reciprocal Rank Fusion (RRF), and white-box observability.
from rag_upgrades.intelligent_chunking import intelligent_chunk_documents
from rag_upgrades.contextual_metadata import enrich_document_metadata
from rag_upgrades.image_captioning import ImageCaptioner
from rag_upgrades.rrf import reciprocal_rank_fusion
from rag_upgrades.tracing import WhiteBoxTracer

load_dotenv(override=True)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("PolyUAdvisor")

# --- Global Configurations & Environment Variables ---
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
COLLECTION_NAME = "polyu_advisor_telegram_hybrid_bm25"
DB_FILE = "polyu_advisor.db"
INDEX_META_FILE = os.getenv("INDEX_META_FILE", "rag_index_meta.json")
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() in {"1", "true", "yes", "y"}
MAX_SCRAPE_WORKERS = int(os.getenv("MAX_SCRAPE_WORKERS", "5"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "200"))
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(75 * 1024 * 1024)))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1500"))
RESPONSE_CACHE_MAX = int(os.getenv("RESPONSE_CACHE_MAX", "128"))
RAG_TRACE_FILE = os.getenv("RAG_TRACE_FILE", "logs/rag_traces.jsonl")
RAG_TRACING_ENABLED = os.getenv("RAG_TRACING_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_FINAL_K = int(os.getenv("RRF_FINAL_K", "4"))
RAG_IMAGE_CAPTIONING_ENABLED = os.getenv("RAG_IMAGE_CAPTIONING_ENABLED", "false").lower() in {"1", "true", "yes", "y"}
RAG_IMAGE_CAPTION_MAX_PER_PDF = int(os.getenv("RAG_IMAGE_CAPTION_MAX_PER_PDF", "8"))

DEFAULT_FAST_MODEL = os.getenv("DEFAULT_FAST_MODEL", "qwen2.5:3b")
THINKING_MODEL = os.getenv("THINKING_MODEL", "deepseek-r1:1.5b")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")
OLLAMA_MODEL_PROFILE = os.getenv("OLLAMA_MODEL_PROFILE", "fast").lower()
OLLAMA_MODEL_RECOMMENDATIONS = {
    "fast": DEFAULT_FAST_MODEL,
    "thinking": THINKING_MODEL,
    "vision": VISION_MODEL,
}
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_RECOMMENDATIONS.get(OLLAMA_MODEL_PROFILE, DEFAULT_FAST_MODEL))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBAPP_URL = os.getenv("WEBAPP_URL", "http://localhost:8000")

rag_chain = None
vector_store = None
tg_app = None
bm25_retriever = None
rag_status = {"state": "starting", "message": "RAG is starting", "started_at": time.time()}
active_stream_sessions: Set[str] = set()
response_cache: Dict[str, str] = {}
response_cache_lock = asyncio.Lock()

# Per-request tracer. ContextVar keeps concurrent Telegram/WebApp requests isolated.
from contextvars import ContextVar
_active_tracer: ContextVar[Optional[WhiteBoxTracer]] = ContextVar("active_rag_tracer", default=None)

def get_active_tracer() -> Optional[WhiteBoxTracer]:
    return _active_tracer.get()

class ChatHistoryItem(BaseModel):
    role: str = "user"
    content: str = ""

class ChatRequest(BaseModel):
    chat_id: Optional[str] = None
    message: str = Field(..., min_length=1)

class AstreamRequest(BaseModel):
    input: str = Field(..., min_length=1)
    chat_history: List[ChatHistoryItem] = Field(default_factory=list)
    browserID: Optional[str] = None
    ip: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

class ClearContextRequest(BaseModel):
    browserID: Optional[str] = None
    session_id: Optional[str] = None


@tool
async def essential_info_tool(query: str) -> str:
    """當需要最新官方資料時，從 PolyU ISE 學系網站及理大官網搜尋文件與資源。"""
    logger.info(f"🔍 [Essential-Info-Tool] 搜尋 PolyU ISE 官方資料：{query}")
    search_query = f"{query} site:polyu.edu.hk/ise OR site:polyu.edu.hk"
    jina_url = f"https://r.jina.ai/https://www.google.com/search?q={urllib.parse.quote(search_query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "X-No-Cache": "true"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 200 and len(resp.text.strip()) > 100:
                extracted_text = resp.text.strip()[:2000]
                return (
                    "【PolyU ISE 官方網站即時搜尋結果】\n"
                    f"搜尋主題：{query}\n"
                    "官方資源內容與連結：\n"
                    f"{extracted_text}\n\n"
                    "（請整理上述資訊，並附上相關官方下載或參考連結）"
                )
    except Exception as e:
        logger.warning(f"⚠️ Essential-Info-Tool 執行失敗: {e}")
    return "⚠️ 目前未能獲取 PolyU ISE 官方網站即時搜尋資料。"


def set_rag_status(state: str, message: str) -> None:
    rag_status.update({"state": state, "message": message, "updated_at": time.time()})
    logger.info(f"📍 RAG status: {state} - {message}")

def normalize_doc_url(url: str) -> str:
    return urldefrag(url.strip())[0]

def compute_index_fingerprint(config_data: Dict[str, Any], embedding_model: str = "nomic-embed-text") -> str:
    payload = {
        "config": config_data,
        "embedding_model": embedding_model,
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "scraper_version": "optimized-v10-intelligent-contextual-rrf",
        "chunking_strategy": "intelligent_structure_semantic",
        "rrf_k": RRF_K,
        "image_captioning": RAG_IMAGE_CAPTIONING_ENABLED,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def load_index_meta() -> Dict[str, Any]:
    if not os.path.exists(INDEX_META_FILE): return {}
    try:
        with open(INDEX_META_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Could not load index metadata: {e}")
        return {}

def save_index_meta(fingerprint: str, points_count: int) -> None:
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "points_count": points_count, "updated_at": time.time()}, f, indent=2)

def is_valid_pdf(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f: return f.read(4) == b"%PDF"
    except Exception: return False

async def download_to_tempfile_async(url: str, suffix: str, headers: Dict[str, str], timeout: int = 30, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Optional[str]:
    tmp_path = None
    safe_url = urllib.parse.quote(urllib.parse.unquote(url.strip()), safe="%/:=&?~#+!$,;'@()*[]")
    try:
        async with httpx.AsyncClient(timeout=float(timeout), follow_redirects=True, verify=False) as client:
            async with client.stream("GET", safe_url, headers=headers) as resp:
                if resp.status_code != 200:
                    logger.warning(f"⚠️ Failed to fetch {safe_url} (HTTP {resp.status_code})")
                    return None
                content_type = resp.headers.get("Content-Type", "").lower()
                if suffix == ".pdf" and "html" in content_type:
                    logger.warning(f"⚠️ URL returned an HTML page (SSO login): {safe_url}")
                    return None
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp_path = tmp.name; total = 0
                    async for chunk in resp.aiter_bytes():
                        if not chunk: continue
                        total += len(chunk)
                        if total > max_bytes: raise ValueError(f"Download exceeded limit of {max_bytes} bytes")
                        tmp.write(chunk)
        if suffix == ".pdf" and not is_valid_pdf(tmp_path):
            logger.warning(f"⚠️ Downloaded file for {safe_url} is HTML/Invalid binary. Skipping.")
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            return None
        return tmp_path
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as te:
        logger.warning(f"⏱️ Network timeout/connection blocked for {safe_url}: {type(te).__name__}. Skipping.")
    except Exception as e:
        logger.warning(f"⚠️ Download failed for {safe_url}: [{type(e).__name__}] {str(e)}")
    if tmp_path and os.path.exists(tmp_path):
        try: os.unlink(tmp_path)
        except Exception: pass
    return None

async def fetch_via_jina_ai_async(target_url: str) -> Optional[str]:
    sso_url_keywords = ["cas.polyu.edu.hk", "/_sso/", "sso.", "login", "signin", "auth"]
    if any(kw in target_url.lower() for kw in sso_url_keywords):
        logger.warning(f"🛡️ Skipping SSO/Authentication URL to avoid login wall: {target_url}")
        return None
    safe_url = urllib.parse.quote(urllib.parse.unquote(target_url.strip()), safe="%/:=&?~#+!$,;'@()*[]")
    jina_url = f"https://r.jina.ai/{safe_url}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "X-No-Cache": "true"}
    try:
        logger.info(f"🤖 Requesting Jina AI parsing for: {safe_url}")
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, verify=False) as client:
            resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                text = resp.text.strip()
                if len(text) > 100:
                    text_lower = text.lower()
                    indicators = ["single sign-on", "netid", "polyu netid", "cas login", "sign in with your netid", "please log in"]
                    if any(indicator in text_lower for indicator in indicators):
                        logger.warning(f"🛡️ Detected SSO login wall in Jina AI response for: {safe_url}. Skipping.")
                        return None
                    logger.info(f"✅ Jina AI successfully extracted {len(text)} characters.")
                    return text
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError):
        logger.warning(f"⏱️ Jina AI connection timeout for {safe_url}. Skipping parser.")
    except Exception as e:
        logger.warning(f"⚠️ Jina AI request exception for {safe_url}: [{type(e).__name__}]")
    return None

CAR_GUR_SOURCE_URL = "https://www.polyu.edu.hk/cus/student/4-year-undergraduate-student/general-university-requirements/curriculum-framework-of-the-general-university-requirements"
ISE_PROGRAMME_DETAILS_URL = "https://www.polyu.edu.hk/ise/study/undergraduate-programmes/beng-hons-scheme-in-product-and-industrial-engineering/bachelor-of-engineering-honours-in-industrial-and-systems-engineering/programme-details"
ISE_CONTACT_URL = "https://www.polyu.edu.hk/ise/about-ise/contact-us/"

CURATED_FACT_INDEX = {
    "car_gur": {
        "query_terms": ["CAR/GUR", "CAR", "GUR", "Cluster-Area Requirements", "General University Requirements", "學分要求", "通識", "大學核心", "PolyU ISE CAR credits", "Senior Year Intake GUR", "高年級銜接"],
        "facts": [
            "核心學制關係：CAR (Cluster-Area Requirements) 是 GUR (General University Requirements) 旗下的子項目（包含關係），絕非相互獨立或相加的兩門課程。",
            "四年制新生（Year 1 Entry）GUR 要求：2025/26 學年起入學總 GUR 為 27 學分（含 9 學分 CAR）；2022/23 至 2024/25 學年入學總 GUR 為 30 學分（含 12 學分 CAR）。",
            "高年級銜接生（Senior Year Intake / Articulated Degree）：GUR 要求大幅豁免，通常只需修讀 9 個 GUR 學分（包含 6 個 CAR 學分），具體依入學審查免修結果為準。",
            "修讀彈性與限制：PolyU 並未硬性規定每學期必須修讀多少 CAR/GUR 學分，學生可自主彈性安排，無每學期最低門檻。",
            "畢業時間與排課建議：常規最高修讀上限為每學期 21 學分。學生通常會在大一與大二（前 3 到 4 個學期）將多數 GUR 與 CAR 完成，以便大三大四專心進行 Capstone 畢業論文與 WIE (Work-Integrated Education) 實習。",
            "CAR 語言要求：CAR 課程同時用作滿足英文閱讀寫作 (ER/EW) 及中文閱讀寫作 (CR/CW) 語言要求。",
        ],
        "answer_format": ["使用繁體中文回答，結構必須極度清晰。", "首先明確澄清：CAR 屬於 GUR 的一部分（包含關係），切勿將兩者學分簡單相加。", "區分四年制直入新生 (Year 1 Entry) 與高年級銜接生 (Senior Year Intake) 的不同學分門檻。", "說明排課彈性：每學期無硬性門檻，通常可在 3 至 4 個學期內修完。", "引導詢問用戶屬於哪一種入學身份（Year 1 Entry 還是 Senior Year Intake），以提供最精準的建議。"],
        "sources": [CAR_GUR_SOURCE_URL, ISE_PROGRAMME_DETAILS_URL, ISE_CONTACT_URL],
    }
}

TERM_INDEX = {}
if os.path.exists("term_index.json"):
    try:
        with open("term_index.json", "r", encoding="utf-8") as f: TERM_INDEX = json.load(f)
        logger.info(f"✅ Loaded {len(TERM_INDEX)} terms from term_index.json")
    except Exception as e: logger.error(f"❌ Error loading term_index.json: {e}")

def sanitize_hallucinations(text: str) -> str:
    if not text: return ""
    pattern = r"Women\s*in\s*Engineering"
    if re.search(pattern, text, re.IGNORECASE):
        logger.warning("🚨 Hallucination caught: 'Women in Engineering' replaced with 'Work-Integrated Education'")
        text = re.sub(pattern, "Work-Integrated Education (校企協作教育)", text, flags=re.IGNORECASE)
    return text.replace("WomeninEngineering", "Work-Integrated Education")

def normalize_and_expand_query(query: str) -> str:
    cantonese_map = {"有甚麼": "要求 指引", "有咩": "要求 指引", "點樣": "流程 方法", "幾多": "學分 數量", "邊啲": "課程 科目"}
    for cant, std in cantonese_map.items(): query = query.replace(cant, std)
    if re.search(r'\bWIE\b', query, re.IGNORECASE): query += " Work-Integrated Education 校企協作教育 實習 實習要求 實習表格"
    if re.search(r'\bCAR\b', query, re.IGNORECASE): query += " Cluster-Area Requirements 通識教育 通識學分要求 GUR 包含關係"
    if re.search(r'\bGUR\b', query, re.IGNORECASE): query += " General University Requirements 大學核心課程要求"
    return query

def is_car_gur_query(query_text: str) -> bool:
    normalized = query_text.lower().replace("／", "/")
    has_car = bool(re.search(r"\bcar\b|cluster-area|cluster area|學群|群組", normalized, re.IGNORECASE))
    has_gur = bool(re.search(r"\bgur\b|general university requirements|大學核心|通識|學分要求", normalized, re.IGNORECASE))
    return has_car or has_gur

def expand_query_with_index(query_text: str) -> str:
    if not is_car_gur_query(query_text): return query_text
    index = CURATED_FACT_INDEX["car_gur"]
    return " | ".join([query_text, *index["query_terms"], *index["facts"][:4]])

def build_index_documents() -> List[Document]:
    return [Document(page_content="\n".join([f"Topic: {key}", "Search terms: " + ", ".join(item["query_terms"]), "Facts:", *[f"- {fact}" for fact in item["facts"]], "Answer format:", *[f"- {fmt}" for fmt in item["answer_format"]], "Sources:", *[f"- {source}" for source in item["sources"]]]), metadata={"source": item["sources"][0], "category": f"Programmatic Index: {key}", "academic_level": "UG", "priority": "programmatic_index"}) for key, item in CURATED_FACT_INDEX.items()]

def build_term_index_documents() -> List[Document]:
    docs = []
    if not TERM_INDEX: return docs
    for key, item in TERM_INDEX.items():
        if isinstance(item, dict):
            content_lines = [f"Term / Keyword: {key}", f"English Name: {item.get('english', '')}", f"Chinese Name: {item.get('chinese', '')}", f"Abbreviation: {item.get('abbreviation', key)}", f"Category: {item.get('category', 'Glossary')}"]
            if item.get("programme_code"): content_lines.append(f"Programme Code: {item['programme_code']}")
            if item.get("jupas_code"): content_lines.append(f"JUPAS Code: {item['jupas_code']}")
            content = "\n".join(content_lines)
        else: content = f"Term: {key}\nDefinition: {item}"
        docs.append(Document(page_content=content, metadata={"source": "term_index.json", "category": item.get("category", "Glossary Index") if isinstance(item, dict) else "Glossary Index", "priority": "programmatic_index"}))
    logger.info(f"✅ Built {len(docs)} enriched documents from term_index.json")
    return docs

def extract_course_codes(text: str) -> List[str]: return sorted(set(re.findall(r"\b[A-Z]{2,5}\d{3,5}\b", text.upper())))

def classify_query(query: str) -> Dict[str, Any]:
    query_lower = query.lower(); course_codes = extract_course_codes(query)
    comparison_terms = ["compare", "versus", " vs ", "better", "suitable", "比較", "分別", "哪個"]
    prerequisite_terms = ["prerequisite", "pre-requisite", "先修", "要求"]
    programme_terms = ["programme", "curriculum", "credit", "car", "gur", "wie", "課程", "學分", "實習"]
    if any(term in query_lower for term in comparison_terms) and len(course_codes) >= 2: query_type, complexity = "COURSE_COMPARISON", "complex"
    elif course_codes and any(term in query_lower for term in prerequisite_terms): query_type, complexity = "PREREQUISITE", "medium"
    elif course_codes: query_type, complexity = "COURSE_LOOKUP", "simple"
    elif any(term in query_lower for term in programme_terms): query_type, complexity = "PROGRAMME", "medium"
    else: query_type, complexity = "GENERAL_POLYU", "medium"
    if complexity == "simple": vector_k, bm25_k, final_k = 8, 8, 3
    elif complexity == "complex": vector_k, bm25_k, final_k = 15, 15, 5
    else: vector_k, bm25_k, final_k = 10, 10, 4
    return {"query_type": query_type, "complexity": complexity, "course_codes": course_codes, "vector_k": vector_k, "bm25_k": bm25_k, "final_k": final_k, "bm25_weight": 0.50, "vector_weight": 0.50}

def compress_document_context(doc: Document, query: str, max_chars: int = 2000) -> Document:
    meta = doc.metadata.copy(); meta.setdefault("source", "PolyU ISE Official Resource"); meta.setdefault("category", "Official Document"); meta.setdefault("_score", 0.0); meta.setdefault("_confidence", "MEDIUM")
    if meta.get("is_table") and meta.get("raw_table"):
        meta["context_compressed"] = False
        return Document(page_content=f"Official Table Data:\n{meta['raw_table']}", metadata=meta)
    if len(doc.page_content) <= max_chars: return Document(page_content=doc.page_content, metadata=meta)
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", query)]
    sentences = re.split(r"(?<=[。.!?])\s+|\n+", doc.page_content)
    selected = [s.strip() for s in sentences if s.strip() and any(t in s.lower() for t in terms)]
    content = "\n".join(selected[:15]) if selected else doc.page_content[:max_chars]
    if len(content) > max_chars: content = content[:max_chars].rsplit(" ", 1)[0]
    meta["context_compressed"] = True
    return Document(page_content=content, metadata=meta)

def extract_tables_and_text(text: str) -> List[Dict[str, Any]]:
    blocks = []; parts = re.split(r'(\[TABLE_START\][\s\S]*?\[TABLE_END\])', text)
    for part in parts:
        if not part.strip(): continue
        blocks.append({"type": "table" if part.startswith("[TABLE_START]") and part.endswith("[TABLE_END]") else "text", "content": part.strip()})
    return blocks

def sentence_level_split(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = nltk.sent_tokenize(text); chunks=[]; current_chunk=[]; current_length=0
    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk)); current_chunk=[sentence]; current_length=len(sentence)
        else: current_chunk.append(sentence); current_length += len(sentence)
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

def advanced_multi_strategy_chunker(documents: List[Document], target_chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    # Kept for backward compatibility with older imports. Production indexing uses IntelligentChunker below.
    final_chunks=[]
    headers_to_split_on=[("#","Header_1"),("##","Header_2"),("###","Header_3")]
    markdown_splitter=MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    recursive_splitter=RecursiveCharacterTextSplitter(chunk_size=target_chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n","\n"," ",""])
    for doc in documents:
        base_metadata=doc.metadata.copy()
        for block_idx, block in enumerate(extract_tables_and_text(doc.page_content)):
            if block["type"] == "table":
                meta=base_metadata.copy(); meta.update({"is_table":True,"chunk_strategy":"table_aware","block_index":block_idx}); final_chunks.append(Document(page_content=block["content"],metadata=meta)); continue
            for h_split in markdown_splitter.split_text(block["content"]):
                heading_path=" > ".join([val for key,val in h_split.metadata.items() if key.startswith("Header_")])
                meta=base_metadata.copy(); meta.update(h_split.metadata); meta.update({"heading_path":heading_path or "Root Section","is_table":False})
                recs=recursive_splitter.split_text(h_split.page_content) if len(h_split.page_content)>target_chunk_size else [h_split.page_content]
                for rec_text in recs:
                    if len(rec_text)>target_chunk_size*1.2:
                        for sent_text in sentence_level_split(rec_text,target_chunk_size):
                            m=meta.copy(); m["chunk_strategy"]="heading+recursive+sentence"; final_chunks.append(Document(page_content=sent_text,metadata=m))
                    else:
                        m=meta.copy(); m["chunk_strategy"]="heading+recursive" if len(h_split.page_content)>target_chunk_size else "heading_direct"; final_chunks.append(Document(page_content=rec_text,metadata=m))
    for chunk in final_chunks: chunk.metadata["course_codes"]=extract_course_codes(chunk.page_content)
    return final_chunks

def init_sqlite_db():
    conn=sqlite3.connect(DB_FILE); cursor=conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS polyu_requirements (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, sub_category TEXT, code TEXT UNIQUE, title TEXT NOT NULL, credits INTEGER DEFAULT 3, description TEXT NOT NULL, department_owner TEXT DEFAULT 'ISE');""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS ise_knowledge_base (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, student_year INTEGER, question TEXT NOT NULL, answer TEXT NOT NULL, vector_embedded BOOLEAN DEFAULT 0);""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS student_sessions (student_chat_id TEXT PRIMARY KEY, current_faculty TEXT, last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
    conn.commit(); conn.close(); logger.info("✅ SQLite database initialized successfully.")

def update_student_session(chat_id: str, faculty: Optional[str] = None):
    try:
        conn=sqlite3.connect(DB_FILE); cursor=conn.cursor()
        cursor.execute("""INSERT INTO student_sessions (student_chat_id,current_faculty,last_interaction) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(student_chat_id) DO UPDATE SET current_faculty=COALESCE(excluded.current_faculty,student_sessions.current_faculty), last_interaction=CURRENT_TIMESTAMP;""",(str(chat_id),faculty)); conn.commit(); conn.close()
    except Exception as e: logger.error(f"❌ SQLite session update failed: {e}")

def clear_user_history(user_id: str) -> bool:
    try:
        conn=sqlite3.connect(DB_FILE); cursor=conn.cursor(); cursor.execute("DELETE FROM student_sessions WHERE student_chat_id = ?",(str(user_id),)); conn.commit(); conn.close(); return True
    except Exception as e: logger.error(f"⚠️ Error clearing SQLite history for user {user_id}: {e}"); return False

def strip_think_tags(text: str) -> str: return re.sub(r'<think>[\s\S]*?</think>','',text).strip() if text else ""

def split_text(text: str, max_length: int = 4000) -> List[str]:
    if len(text)<=max_length: return [text]
    chunks=[]
    while text:
        if len(text)<=max_length: chunks.append(text); break
        split_point=text.rfind("\n",0,max_length)
        if split_point==-1: split_point=text.rfind(" ",0,max_length)
        if split_point==-1: split_point=max_length
        chunks.append(text[:split_point].strip()); text=text[split_point:].strip()
    return chunks

async def send_chunked_message(update: Update, text: str, parse_mode: str = "Markdown", reply_to_message_id: Optional[int] = None):
    for i,chunk in enumerate(split_text(text)):
        msg_reply_id=reply_to_message_id if i==0 else None
        try: await update.message.reply_text(chunk,parse_mode=parse_mode,reply_to_message_id=msg_reply_id)
        except Exception: await update.message.reply_text(chunk,reply_to_message_id=msg_reply_id)

def classify_academic_level(url: str, text: str) -> str:
    url_lower=url.lower(); text_lower=text.lower()
    if any(k in url_lower for k in ["undergraduate","beng","bsc","ug"]) or any(k in text_lower for k in ["bachelor of","bsc (hons)","beng (hons)"]): return "UG"
    if any(k in url_lower for k in ["postgraduate","msc","master","pg"]) or any(k in text_lower for k in ["master of","msc in","postgraduate scheme"]): return "PG"
    return "General"

def extract_tables_as_md(page) -> str:
    blocks=[]
    for table in page.extract_tables() or []:
        if not table or len(table)<2: continue
        rows=[r for r in table if any(str(c).strip() for c in r)]
        if len(rows)<2: continue
        md=[]
        for row in rows: md.append("| " + " | ".join([str(c or "").replace("|","\\|").strip() for c in row]) + " |")
        md.insert(1,"|"+"---|"*len(rows[0])); blocks.append("\n".join(md))
    return "\n\n[TABLE_START]\n"+"\n\n".join(blocks)+"\n[TABLE_END]\n" if blocks else ""

def load_pdf_with_structure(pdf_path: str) -> List[Document]:
    docs=[]
    with pdfplumber.open(pdf_path) as pdf:
        for i,page in enumerate(pdf.pages):
            text=page.extract_text() or ""; text+=extract_tables_as_md(page)
            if text.strip(): docs.append(Document(page_content=text,metadata={"page":i+1,"source":pdf_path}))
    return docs

async def load_pdf_with_docling_async(pdf_path: str, source_url: str) -> List[Document]:
    if DocumentConverter is not None:
        def _docling_parse():
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            pipeline_options=PdfPipelineOptions(); pipeline_options.do_ocr=False; pipeline_options.do_table_structure=True
            converter=DocumentConverter(format_options={"pdf":PdfFormatOption(pipeline_options=pipeline_options)})
            return converter.convert(pdf_path).document.export_to_markdown()
        try:
            logger.info(f"📄 Parsing PDF with Docling: {source_url}"); markdown_text=await asyncio.to_thread(_docling_parse)
            if markdown_text and len(markdown_text.strip())>50:
                return [Document(page_content=markdown_text,metadata={"source":source_url,"category":"Docling Parsed PDF","academic_level":classify_academic_level(source_url,markdown_text)})]
        except Exception as e: logger.warning(f"⚠️ Docling parsing failed for {source_url}: {e}. Triggering standard PDF fallback...")
    return await load_pdf_safely_async(pdf_path,source_url)

async def load_pdf_safely_async(pdf_path: str, source_url: str) -> List[Document]:
    docs=[]
    try:
        docs=await asyncio.to_thread(PyPDFLoader(pdf_path).load)
        if docs: return docs
    except Exception as e: logger.warning(f"⚠️ PyPDFLoader failed for {source_url}: {e}. Trying lenient PdfReader...")
    def _read_pdf_lenient():
        reader=PdfReader(pdf_path,strict=False); extracted=[]
        for i,page in enumerate(reader.pages):
            text=page.extract_text() or ""
            if text.strip(): extracted.append(Document(page_content=text,metadata={"page":i+1,"source":source_url}))
        return extracted
    try:
        docs=await asyncio.to_thread(_read_pdf_lenient)
        if docs: return docs
    except Exception as e: logger.warning(f"⚠️ Lenient PdfReader failed for {source_url}: {e}. Trying pdfplumber...")
    try:
        docs=await asyncio.to_thread(load_pdf_with_structure,pdf_path)
        for doc in docs: doc.metadata["source"]=source_url
        if docs: return docs
    except Exception as e: logger.warning(f"⚠️ pdfplumber failed for {source_url}: {e}. Triggering Jina AI fallback...")
    if source_url.startswith(("http://","https://")):
        jina_text=await fetch_via_jina_ai_async(source_url)
        if jina_text: return [Document(page_content=jina_text,metadata={"source":source_url,"category":"Jina AI Parsed PDF"})]
    return docs

async def scrape_webpage_async(url: str) -> List[Document]:
    try:
        logger.info(f"🌐 Scraping webpage with Jina AI: {url}"); web_text=await fetch_via_jina_ai_async(url)
        if web_text: return [Document(page_content=web_text,metadata={"source":url,"category":"Official Webpage (Jina AI)","academic_level":classify_academic_level(url,web_text)})]
    except Exception as e: logger.error(f"⚠️ Webpage scraping error for {url}: {e}")
    return []

def load_prd_by_sections(pdf_path: str, source_url: str, programme: str = "General") -> List[Document]:
    full_text=""; page_starts=[]
    with pdfplumber.open(pdf_path) as pdf:
        for i,page in enumerate(pdf.pages):
            page_starts.append((len(full_text),i+1)); full_text+=(page.extract_text() or "")+extract_tables_as_md(page)+"\n\n"
    section_re=re.compile(r'SECTION\s+\d+.*?$',re.MULTILINE|re.IGNORECASE); subsection_re=re.compile(r'^#?\s*\d+\.\d+[A-Z][^\n]*$',re.MULTILINE); subject_form_re=re.compile(r'^#?\s*Subject Description Form\s*$',re.MULTILINE)
    def page_at(pos):
        page_num=1
        for start,pnum in page_starts:
            if start<=pos: page_num=pnum
            else: break
        return page_num
    section_matches=list(section_re.finditer(full_text))
    if not section_matches:
        chunks=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=100).split_text(full_text)
        return [Document(page_content=c,metadata={"source":source_url,"programme":programme,"section":"Fallback Split","category":"Official PDF","chunk_type":"fallback_split"}) for c in chunks]
    documents=[]
    for idx,match in enumerate(section_matches):
        sec_start=match.start(); sec_end=section_matches[idx+1].start() if idx+1<len(section_matches) else len(full_text); section_text=full_text[sec_start:sec_end].strip(); section_title=match.group(0).strip().lstrip("#").strip()
        if not section_text: continue
        if "SYLLABUS" in section_title.upper() or "SUBJECT" in section_title.upper():
            subject_matches=list(subject_form_re.finditer(section_text))
            if len(subject_matches)>1:
                first_hit=subject_matches[0].start()
                if first_hit>10: documents.append(Document(page_content=section_text[:first_hit].strip(),metadata={"source":source_url,"original_filename":os.path.basename(source_url),"programme":programme,"section":section_title,"category":"Official Programme Definition Document (PRD)","chunk_type":"section"}))
                for s_idx,s_match in enumerate(subject_matches):
                    s_start=s_match.start(); s_end=subject_matches[s_idx+1].start() if s_idx+1<len(subject_matches) else len(section_text); subj_text=section_text[s_start:s_end].strip(); code_m=re.search(r'Subject Code\s+([A-Z0-9]+)',subj_text); subj_code=code_m.group(1) if code_m else f"Unknown-{s_idx}"
                    documents.append(Document(page_content=subj_text,metadata={"source":source_url,"programme":programme,"section":section_title,"subsection":f"Subject {subj_code}","subject_code":subj_code,"category":"Official PDF","chunk_type":"subject_form","page_start":page_at(sec_start+s_start),"page_end":page_at(sec_start+s_end)}))
                continue
        if len(section_text)>6000:
            sub_matches=list(subsection_re.finditer(section_text))
            if len(sub_matches)>1:
                for s_idx,s_match in enumerate(sub_matches):
                    s_start=s_match.start(); s_end=sub_matches[s_idx+1].start() if s_idx+1<len(sub_matches) else len(section_text); sub_text=section_text[s_start:s_end].strip(); sub_title=s_match.group(0).strip().lstrip("#").strip()[:120]
                    documents.append(Document(page_content=sub_text,metadata={"source":source_url,"programme":programme,"section":section_title,"subsection":sub_title,"category":"Official PDF","chunk_type":"subsection","page_start":page_at(sec_start+s_start),"page_end":page_at(sec_start+s_end)}))
                continue
        documents.append(Document(page_content=section_text,metadata={"source":source_url,"programme":programme,"section":section_title,"category":"Official PDF","chunk_type":"section","page_start":page_at(sec_start),"page_end":page_at(sec_end)}))
    return documents


class ScoreInjectingRetriever(BaseRetriever):
    """Hybrid Qdrant + BM25 retriever using true Reciprocal Rank Fusion."""
    vectorstore: Any = Field(description="The underlying Qdrant vector store")
    bm25: Any = Field(default=None, description="Optional BM25 lexical retriever")
    k: int = Field(default=4)
    score_threshold: float = Field(default=0.0)
    rrf_k: int = Field(default=60)

    def _perform_hybrid_search(self, raw_query: str) -> List[Document]:
        tracer=get_active_tracer(); query=normalize_and_expand_query(raw_query); plan=classify_query(query); start=time.perf_counter()
        def vector_lookup():
            with (tracer.span("retrieval.vector_search",metadata={"k":plan["vector_k"]}) if tracer else _null_span()) as sp:
                q=raw_query.lower(); is_pg=any(x in q for x in ["master","msc","postgraduate","pgd","pg"]); is_ug=any(x in q for x in ["bachelor","bsc","beng","undergraduate","ug"]); filt=None
                if is_pg and not is_ug: filt=Filter(must=[FieldCondition(key="metadata.academic_level",match=MatchAny(any=["PG","General"]))])
                elif is_ug and not is_pg: filt=Filter(must=[FieldCondition(key="metadata.academic_level",match=MatchAny(any=["UG","General"]))])
                results=self.vectorstore.similarity_search_with_score(query,k=plan["vector_k"],filter=filt)
                if tracer: sp.metadata["result_count"]=len(results)
                return results
        def bm25_lookup():
            if not self.bm25: return []
            with (tracer.span("retrieval.bm25_search",metadata={"k":plan["bm25_k"]}) if tracer else _null_span()) as sp:
                q=raw_query.lower(); is_pg=any(x in q for x in ["master","msc","postgraduate","pgd","pg"]); is_ug=any(x in q for x in ["bachelor","bsc","beng","undergraduate","ug"]); old=self.bm25.k; self.bm25.k=plan["bm25_k"]*3
                try:
                    try: raw=self.bm25.invoke(query)
                    except AttributeError: raw=self.bm25.get_relevant_documents(query)
                finally: self.bm25.k=old
                filtered=[]
                for doc in raw:
                    level=str(doc.metadata.get("academic_level","General")).upper()
                    if is_pg and not is_ug and level not in ["PG","GENERAL"]: continue
                    if is_ug and not is_pg and level not in ["UG","GENERAL"]: continue
                    filtered.append(doc)
                    if len(filtered)>=plan["bm25_k"]: break
                if tracer: sp.metadata["result_count"]=len(filtered)
                return filtered
        docs_and_scores=vector_lookup(); bm25_docs=bm25_lookup(); vector_docs=[d for d,_ in docs_and_scores]
        with (tracer.span("retrieval.rrf",metadata={"vector_candidates":len(vector_docs),"bm25_candidates":len(bm25_docs),"rrf_k":self.rrf_k}) if tracer else _null_span()) as sp:
            fused=reciprocal_rank_fusion([vector_docs,bm25_docs],key_fn=self._doc_key,k=self.rrf_k)
            if tracer: sp.metadata["fused_candidates"]=len(fused)
        ranked=[]; ql=query.lower(); is_pg=any(x in ql for x in ["master","msc","postgraduate","pgd","pg"]); is_wie="wie" in raw_query.lower() or "實習" in raw_query; qcodes=set(plan["course_codes"])
        vector_ranks={self._doc_key(d):i for i,(d,_) in enumerate(docs_and_scores,1)}; bm25_ranks={self._doc_key(d):i for i,d in enumerate(bm25_docs,1)}
        with (tracer.span("retrieval.filter_and_compress",metadata={"input_count":len(fused)}) if tracer else _null_span()) as sp:
            for rrf_rank,doc in enumerate(fused,1):
                key=self._doc_key(doc); rrf_score=float(doc.metadata.get("_rrf_score",0.0)); content=doc.page_content.lower(); level=str(doc.metadata.get("academic_level","")).upper(); tie=0.0
                if is_wie and ("work-integrated" in content or "wie" in content or "實習" in content): tie+=0.001
                if is_pg and (re.search(r'\bISE[5-6]\d{3}\b',doc.page_content,re.I) or level=="PG"): tie+=0.001
                elif not is_pg and (re.search(r'\bISE[1-4]\d{3}\b',doc.page_content,re.I) or level=="UG"): tie+=0.001
                if qcodes and qcodes.intersection(set(doc.metadata.get("course_codes",[]))): tie+=0.001
                meta=doc.metadata.copy(); meta.update({"_score":round(rrf_score,8),"_rrf_rank":rrf_rank,"_vector_rank":vector_ranks.get(key),"_bm25_rank":bm25_ranks.get(key),"_query_type":plan["query_type"],"_complexity":plan["complexity"],"_domain_tiebreak":tie,"_confidence":"HIGH" if len(meta.get("_rrf_ranks",[]))>=2 else "MEDIUM"})
                ranked.append(compress_document_context(Document(page_content=doc.page_content,metadata=meta),query))
            ranked.sort(key=lambda d:(d.metadata.get("_score",0.0),d.metadata.get("_domain_tiebreak",0.0)),reverse=True)
            if tracer: sp.metadata["result_count"]=len(ranked)
        result=ranked[:plan["final_k"]]; elapsed=time.perf_counter()-start
        if tracer: tracer.annotate(retrieval_candidates=len(fused),retrieval_results=len(result),retrieval_latency_ms=round(elapsed*1000,3))
        logger.info("🔎 RRF retrieval plan=%s/%s candidates=%s results=%s elapsed=%.2fs",plan["query_type"],plan["complexity"],len(fused),len(result),elapsed)
        return result

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        ranked=self._perform_hybrid_search(query)
        if not ranked:
            logger.info("🌐 RRF returned no documents; activating Essential-Info-Tool.")
            try: official=asyncio.run(essential_info_tool.ainvoke({"query":query}))
            except Exception: official=None
            if official and "⚠️" not in official:
                ranked.insert(0,Document(page_content=official,metadata={"source":ISE_PROGRAMME_DETAILS_URL,"priority":"essential_tool","_score":1.0,"_rrf_score":0.0,"_confidence":"HIGH","category":"PolyU ISE Official Live Data"}))
        return ranked[:self.k]
    @staticmethod
    def _normalize_vector_score(raw_score: float) -> float:
        score=float(raw_score); return max(0.0,1.0/(1.0+score)) if score>1.0 else min(max(score,0.0),1.0)
    @staticmethod
    def _doc_key(doc: Document) -> str:
        source=str(doc.metadata.get("source","")); page=str(doc.metadata.get("page",doc.metadata.get("page_start",""))); return f"{source}:{page}:{doc.page_content[:240]}"

class _null_span:
    def __enter__(self): return _NullSpan()
    def __exit__(self,exc_type,exc,tb): return False
class _NullSpan:
    metadata: Dict[str,Any]={}


async def caption_pdf_images_async(pdf_path: str, source_url: str) -> List[Document]:
    if not RAG_IMAGE_CAPTIONING_ENABLED: return []
    captioner=ImageCaptioner(timeout=OLLAMA_REQUEST_TIMEOUT); output=[]; tracer=WhiteBoxTracer(enabled=RAG_TRACING_ENABLED,sink_path=RAG_TRACE_FILE)
    try:
        with tracer.span("ingestion.image_captioning",metadata={"source":source_url}) as sp:
            def _extract():
                extracted=[]
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num,page in enumerate(pdf.pages,1):
                        for image_idx,image in enumerate(page.images or []):
                            if len(extracted)>=RAG_IMAGE_CAPTION_MAX_PER_PDF: return extracted
                            try:
                                bbox=(image["x0"],image["top"],image["x1"],image["bottom"]); pil=page.crop(bbox).to_image(resolution=100).original; import io; buf=io.BytesIO(); pil.save(buf,format="JPEG",quality=85); extracted.append((page_num,image_idx,buf.getvalue()))
                            except Exception as exc: logger.warning("Image extraction failed page=%s image=%s: %s",page_num,image_idx,exc)
                return extracted
            extracted=await asyncio.to_thread(_extract)
            for page_num,image_idx,image_bytes in extracted:
                try:
                    caption=await asyncio.to_thread(captioner.caption_bytes,image_bytes)
                    if caption:
                        output.append(Document(page_content=captioner.to_searchable_text(caption,f"{source_url}#page={page_num}&image={image_idx}",page_num),metadata={"source":source_url,"page":page_num,"image_index":image_idx,"category":"PDF Image / Figure Caption","academic_level":"General","is_image_caption":True}))
                except Exception as exc: logger.warning("Image captioning failed page=%s image=%s: %s",page_num,image_idx,exc)
            sp.metadata.update({"images_extracted":len(extracted),"captions_created":len(output)})
    except Exception as exc: logger.warning("PDF image captioning pipeline failed for %s: %s",source_url,exc)
    return output

async def get_rag_chain_async():
    global bm25_retriever,vector_store
    qdrant_url=os.getenv("QDRANT_URL"); qdrant_api_key=os.getenv("QDRANT_API_KEY"); ollama_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434"); ollama_model=os.getenv("OLLAMA_MODEL",DEFAULT_FAST_MODEL)
    embeddings=OllamaEmbeddings(model="nomic-embed-text",base_url=ollama_url); client=QdrantClient(url=qdrant_url,api_key=qdrant_api_key,timeout=120,check_compatibility=False)
    config_data={}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE,"r",encoding="utf-8") as cfg: config_data=json.load(cfg)
    if not client.collection_exists(COLLECTION_NAME): client.create_collection(collection_name=COLLECTION_NAME,vectors_config=VectorParams(size=768,distance=Distance.COSINE))
    v_store=QdrantVectorStore(client=client,collection_name=COLLECTION_NAME,embedding=embeddings); vector_store=v_store; info=client.get_collection(COLLECTION_NAME); points_count=info.points_count if info.points_count is not None else 0; fingerprint=compute_index_fingerprint(config_data); cached_meta=load_index_meta(); all_docs=[]
    if config_data:
        urls=config_data.get("urls",[]); pdf_paths=config_data.get("pdfs",[]); sem=asyncio.Semaphore(2)
        if urls:
            set_rag_status("scraping",f"Scraping {len(urls)} configured webpages with Jina AI")
            async def safe_web_scrape(u):
                async with sem: return await scrape_webpage_async(u)
            results=await asyncio.gather(*[safe_web_scrape(u) for u in urls],return_exceptions=True)
            for res in results:
                if isinstance(res,list): all_docs.extend(res)
        if pdf_paths:
            headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36","Accept":"application/pdf,*/*"}
            for pdf_url in pdf_paths:
                async with sem:
                    try:
                        temp_pdf=await download_to_tempfile_async(pdf_url,".pdf",headers,timeout=60)
                        if not temp_pdf: continue
                        url_upper=pdf_url.upper()
                        if any(x in url_upper for x in ["45499-LEM","LEM"]): prog="LEM"
                        elif any(x in url_upper for x in ["45499-EEM","EEM"]): prog="EEM"
                        elif any(x in url_upper for x in ["45498-ISE","ISE"]): prog="ISE"
                        elif any(x in url_upper for x in ["45498-PEM","PEM"]): prog="PEM"
                        elif any(x in url_upper for x in ["PIE","45498"]): prog="PIE"
                        elif any(x in url_upper for x in ["AOS","45497"]): prog="AOS"
                        else: prog="General"
                        docs=await asyncio.to_thread(load_prd_by_sections,temp_pdf,pdf_url,prog) if ("PRD" in pdf_url.upper() or "45499" in pdf_url.upper()) else await load_pdf_with_docling_async(temp_pdf,pdf_url)
                        clean_filename=os.path.basename(urllib.parse.urlparse(pdf_url).path) or os.path.basename(pdf_url)
                        for doc in docs:
                            doc.metadata.update({"source":pdf_url,"original_filename":clean_filename,"programme":prog,"academic_level":classify_academic_level(pdf_url,doc.page_content)})
                        image_docs=await caption_pdf_images_async(temp_pdf,pdf_url)
                        for image_doc in image_docs: image_doc.metadata.update({"original_filename":clean_filename,"programme":prog,"academic_level":classify_academic_level(pdf_url,image_doc.page_content)})
                        all_docs.extend(docs); all_docs.extend(image_docs)
                        if os.path.exists(temp_pdf): os.unlink(temp_pdf)
                    except Exception as e: logger.error(f"⚠️ PDF Pipeline Error for {pdf_url}: {e}")
    all_docs.extend(build_index_documents()); all_docs.extend(build_term_index_documents()); splits=[]
    if all_docs:
        logger.info(f"Total raw docs ingested dynamically: {len(all_docs)}")
        ingestion_tracer=WhiteBoxTracer(enabled=RAG_TRACING_ENABLED,sink_path=RAG_TRACE_FILE)
        with ingestion_tracer.span("ingestion.chunking",metadata={"raw_documents":len(all_docs)}) as sp:
            splits=intelligent_chunk_documents(all_docs,target_size=1200,min_size=250,max_size=1700); sp.metadata["chunk_count"]=len(splits)
        with ingestion_tracer.span("ingestion.contextual_metadata",metadata={"chunk_count":len(splits)}) as sp:
            splits=[enrich_document_metadata(doc) for doc in splits]; sp.metadata["enriched_count"]=len(splits)
        with ingestion_tracer.span("ingestion.bm25_index",metadata={"chunk_count":len(splits)}) as sp:
            bm25_retriever=BM25Retriever.from_documents(splits); bm25_retriever.k=10; sp.metadata["bm25_k"]=bm25_retriever.k
    else: bm25_retriever=None
    if splits and (FORCE_REINDEX or points_count<1000 or cached_meta.get("fingerprint")!=fingerprint):
        logger.info(f"📤 Uploading {len(splits)} document chunks to Qdrant...")
        for i in range(0,len(splits),QDRANT_BATCH_SIZE):
            try: v_store.add_documents(splits[i:i+QDRANT_BATCH_SIZE])
            except Exception as e: logger.error(f"❌ Upload Batch Failed [{i}:{i+QDRANT_BATCH_SIZE}]: {e}")
        save_index_meta(fingerprint,client.get_collection(COLLECTION_NAME).points_count); logger.info("✅ Qdrant indexing completed.")
    return build_rag_chain(v_store,bm25_retriever,ollama_model,ollama_url)

def clean_source_info(doc: Document) -> Tuple[str,str]:
    src=str(doc.metadata.get("source","")).strip(); category=str(doc.metadata.get("category","")).strip(); link_text=str(doc.metadata.get("link_text","")).strip(); programme=str(doc.metadata.get("programme","")).strip(); original_file=str(doc.metadata.get("original_filename","")).strip()
    if link_text and not link_text.startswith("http"): title=link_text
    elif category and category not in ["Official PDF","Official Document"]: title=category
    elif programme and programme!="General": title=f"PolyU ISE {programme} Academic Document"
    else: title="PolyU ISE Academic Guidelines"
    if src.startswith(("http://","https://")):
        filename=os.path.basename(urllib.parse.urlparse(src).path)
        if filename and filename.lower().endswith((".pdf",".docx")): title=f"{title} ({filename})"
        return title,src
    if "/tmp/" in src or "tmp" in os.path.basename(src): return f"{title} ({os.path.basename(original_file) if original_file else 'PolyU_Academic_Guide.pdf'})", ""
    return f"{title} ({os.path.basename(src) if src else 'Official Document'})", ""

def format_reference_footer(context_docs: List[Document], min_score: float = 0.0) -> str:
    if not context_docs: return ""
    references=[]; seen=set()
    for doc in context_docs:
        score=float(doc.metadata.get("_score",0.0))
        if score<min_score: continue
        conf=doc.metadata.get("_confidence","MEDIUM"); display_title,clean_url=clean_source_info(doc); dedup=clean_url or display_title
        if not dedup or dedup in seen: continue
        seen.add(dedup); references.append(f"- [{display_title}]({clean_url}) | 🎯 置信度：{score:.4f} ({conf})" if clean_url else f"- 📄 **{display_title}** | 🎯 置信度：{score:.4f} ({conf})")
    return "\n\n---\n### 📚 參考資料與置信度 (References & Confidence Score)\n"+"\n".join(references) if references else ""

def build_rag_chain(v_store,bm25,ollama_model: str,ollama_url: str):
    global vector_store; vector_store=v_store
    llm=ChatOllama(model=ollama_model,base_url=ollama_url,temperature=0.2,top_p=0.9,num_predict=OLLAMA_NUM_PREDICT,num_ctx=OLLAMA_NUM_CTX)
    retriever=ScoreInjectingRetriever(vectorstore=v_store,bm25=bm25,k=RRF_FINAL_K,score_threshold=0.0,rrf_k=RRF_K)
    document_prompt=PromptTemplate.from_template("Document: {category}\nSource: {source}\nContextual retrieval metadata: {retrieval_context}\nContent:\n{page_content}\n\n")
    system_prompt=("你是 Alex，香港理工大學 (PolyU) 工業及系統工程學系 (ISE) 的官方學術諮詢助手。\n\n"
        "⚡ 速度與思考過程優化 (FAST ANSWER RULE)：\n務必直接回答問題，嚴禁輸出任何 <think> 標籤或內部推理過程。\n\n"
        "⚠️ 專有名詞與學制邏輯極重要約束：\n"
        "1. CAR 屬於 GUR 的一部分，絕非相加關係。\n"
        "2. WIE 唯一代表 Work-Integrated Education (校企協作教育 / 實習)，不可解釋為 Women in Engineering。\n"
        "3. 只能根據 Context 回答；若資料不足，清楚說明資料不足，不得虛構課程規則或網址。\n\n"
        "📖 回答要求：使用清晰 Markdown 標題、粗體及條列；使用繁體中文回答，若用戶使用英文則用英文回答。\n\n"
        "Context:\n{context}")
    qa_prompt=ChatPromptTemplate.from_messages([("system",system_prompt),MessagesPlaceholder("chat_history"),("human","{input}")])
    combine_docs_chain=create_stuff_documents_chain(llm,qa_prompt,document_prompt=document_prompt)
    return create_retrieval_chain(retriever,combine_docs_chain),v_store

async def run_rag_query(query_text: str, chat_history: Optional[List[ChatHistoryItem]] = None) -> str:
    global rag_chain
    if rag_chain is None:
        elapsed=round(time.time()-rag_status.get("started_at",time.time()),1); return f"⏳ Alex 正在準備知識庫：{rag_status.get('message','starting')}（已用 {elapsed} 秒）。請稍後再試。"
    normalized_query=compress_long_query(query_text); cached=await get_cached_response_async(normalized_query)
    if cached: logger.info("⚡ Response cache hit for normalized query."); return cached
    tracer=WhiteBoxTracer(enabled=RAG_TRACING_ENABLED,sink_path=RAG_TRACE_FILE); token=_active_tracer.set(tracer)
    try:
        with tracer.span("rag.end_to_end",metadata={"query_length":len(normalized_query),"model":OLLAMA_MODEL,"history_messages":len(chat_history or [])}) as root:
            with tracer.span("query.prepare") as sp: history_messages=normalize_chat_history(chat_history or []); sp.metadata["history_count"]=len(history_messages)
            try:
                with tracer.span("generation.llm") as sp:
                    result=await asyncio.wait_for(rag_chain.ainvoke({"input":normalized_query,"chat_history":history_messages}),timeout=OLLAMA_REQUEST_TIMEOUT); sp.metadata["answer_chars"]=len(str(result.get("answer","")))
                raw_answer=result.get("answer","抱歉，我無法檢索到相關解答。"); clean_answer=sanitize_hallucinations(strip_think_tags(raw_answer)); context_docs=result.get("context",[])
                with tracer.span("citation.formatting",metadata={"context_count":len(context_docs)}) as sp:
                    ref_footer=format_reference_footer(context_docs); sp.metadata["reference_count"]=ref_footer.count("\n- ")
                final_answer=clean_answer+ref_footer; await set_cached_response_async(normalized_query,final_answer); root.metadata["final_answer_chars"]=len(final_answer); return final_answer
            except asyncio.TimeoutError:
                tracer.annotate(error_type="timeout"); return f"⚠️ 本地 AI 模型 ({OLLAMA_MODEL}) 回應逾時。"
            except Exception as e:
                tracer.annotate(error_type=type(e).__name__); logger.exception("RAG query failed"); return f"抱歉，系統運算時發生技術故障：{e}"
    finally: _active_tracer.reset(token)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id=update.effective_chat.id; update_student_session(str(chat_id)); app_url=f"{WEBAPP_URL}/webapp"; keyboard=[[InlineKeyboardButton("🚀 啟動 Academic Advisor Web App",web_app=WebAppInfo(url=app_url))],[InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)",callback_data="faculty_ise")],[InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)",callback_data="faculty_gur")]]
    await update.message.reply_text("👋 歡迎使用香港理工大學 (PolyU) 學術諮詢 AI 助手 (Alex)！\n\n點擊下方按鈕啟動全新的 **Telegram Web App**，或直接在聊天室提問：",reply_markup=InlineKeyboardMarkup(keyboard))

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📚 **Alex Knowledge Base & Advice Scope**\n\nI am trained on official guidelines from the Department of Industrial and Systems Engineering (ISE) at PolyU.\n\n🔄 **Start fresh:** Type `/clear` anytime to reset conversation memory.",parse_mode="Markdown")
async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    success=clear_user_history(str(update.effective_user.id)); await update.message.reply_text("🔄 **Session reset!** Your chat history has been cleared." if success else "⚠️ Could not reset session history.",parse_mode="Markdown")
async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query=update.callback_query; await query.answer(); chat_id=update.effective_chat.id
    if query.data=="faculty_ise":
        update_student_session(str(chat_id),faculty="ISE"); keyboard=[[InlineKeyboardButton("📋 CAR / GUR 學分要求",callback_data="ise_car")],[InlineKeyboardButton("💼 WIE 實習 / 課外活動要求",callback_data="ise_wie")],[InlineKeyboardButton("🎓 Capstone 畢業論文選題",callback_data="ise_capstone")],[InlineKeyboardButton("🔙 返回主選單",callback_data="go_main")]]; await query.edit_message_text("📍 **你已進入 ISE 學術諮詢專區**\n請選擇你想了解的疑問範疇：",reply_markup=InlineKeyboardMarkup(keyboard),parse_mode="Markdown")
    elif query.data in {"ise_wie","ise_car"}:
        update_student_session(str(chat_id)); prompt="請問 ISE 學生 WIE 實習的要求是什麼？有哪些表格可以下載？" if query.data=="ise_wie" else "請問 ISE 學生 CAR 和 GUR 的學分要求是什麼？"; await query.edit_message_text("🔍 正在檢索相關 PolyU 官方資料，請稍候..."); await context.bot.send_chat_action(chat_id=chat_id,action=ChatAction.TYPING); response=await run_rag_query(prompt); await send_chunked_message(update,response,parse_mode="Markdown")
    elif query.data=="go_main":
        update_student_session(str(chat_id),faculty="General"); keyboard=[[InlineKeyboardButton("🚀 啟動 Academic Advisor Web App",web_app=WebAppInfo(url=f"{WEBAPP_URL}/webapp"))],[InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)",callback_data="faculty_ise")],[InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)",callback_data="faculty_gur")]]; await query.edit_message_text("請選擇你所屬的學系或諮詢範疇：",reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    student_text=(update.message.text or "").strip()
    if not student_text: await update.message.reply_text("請輸入問題後再送出。"); return
    chat_id=update.effective_chat.id; update_student_session(str(chat_id)); await context.bot.send_chat_action(chat_id=chat_id,action=ChatAction.TYPING); placeholder=await update.message.reply_text("🤔 Alex 正在思考並查閱 PolyU 學術指引...")
    ai_response=await run_rag_query(student_text)
    try: await placeholder.delete()
    except Exception: pass
    await send_chunked_message(update,ai_response,parse_mode="Markdown")

async def load_rag_in_background_async():
    global rag_chain
    try:
        set_rag_status("loading","Loading and indexing official PolyU resources")
        rag_chain,_=await get_rag_chain_async(); set_rag_status("ready","RAG knowledge base is ready")
    except Exception as e:
        logger.exception("RAG initialization failed"); set_rag_status("error",f"RAG initialization failed: {type(e).__name__}")

async def start_telegram_bot():
    global tg_app
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("⚠️ No valid TELEGRAM_BOT_TOKEN set in .env. Skipping Telegram setup."); return
    custom_request=HTTPXRequest(connection_pool_size=4,read_timeout=20.0,write_timeout=20.0,connect_timeout=20.0,pool_timeout=20.0)
    tg_app=Application.builder().token(TELEGRAM_BOT_TOKEN).request(custom_request).build()
    tg_app.add_handler(CommandHandler("start",start_command)); tg_app.add_handler(CommandHandler("help",help_command)); tg_app.add_handler(CommandHandler("clear",clear_command)); tg_app.add_handler(CallbackQueryHandler(button_click)); tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_message))
    try:
        await tg_app.initialize(); await tg_app.start(); asyncio.create_task(tg_app.updater.start_polling(poll_interval=3.0,drop_pending_updates=True))
    except Exception as e: logger.warning(f"⚠️ Telegram bot startup encountered a network hiccup: {e}. Running in backend-only mode.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sqlite_db(); asyncio.create_task(load_rag_in_background_async()); await start_telegram_bot(); yield
    if tg_app:
        try:
            if tg_app.updater and tg_app.updater.running: await tg_app.updater.stop()
            await tg_app.stop(); await tg_app.shutdown()
        except Exception as e: logger.info(f"ℹ️ Telegram graceful shutdown completed with network notice: {type(e).__name__}")

app=FastAPI(title="PolyU AI Academic Advisor WebApp",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

def review_docling_conversion(file_path: str, output_md_path: str = "review_output.md") -> str:
    if not os.path.exists(file_path) and not file_path.startswith("http"): raise FileNotFoundError(f"Source file not found: {file_path}")
    converter=DocumentConverter(); result=converter.convert(file_path); markdown_content=result.document.export_to_markdown(); open(output_md_path,"w",encoding="utf-8").write(markdown_content); return output_md_path

def compress_long_query(text: str) -> str:
    text=text.strip(); return text if len(text)<=MAX_QUERY_LENGTH else f"{text[:MAX_QUERY_LENGTH-160]}\n\n[Query compressed]"
def normalize_chat_history(history: List[ChatHistoryItem]):
    messages=[]
    for item in history[-MAX_HISTORY_MESSAGES:]:
        content=item.content.strip()
        if not content: continue
        messages.append(AIMessage(content=content) if item.role.lower()=="assistant" else HumanMessage(content=content))
    return messages

def extract_html_tables_as_markdown(html_content: str) -> str:
    soup=BeautifulSoup(html_content,"html.parser"); markdown_tables=[]
    for table in soup.find_all("table"):
        rows=table.find_all("tr");
        if not rows: continue
        md_rows=[]
        for i,row in enumerate(rows):
            cells=[]
            for cell in row.find_all(["th","td"]):
                link=cell.find("a"); cell_text=f"[{link.get_text(strip=True).replace('|','')}]({link.get('href')})" if link and link.get("href") else cell.get_text(strip=True).replace("|","\\|"); cells.append(cell_text)
            if cells: md_rows.append("| " + " | ".join(cells) + " |"); md_rows.append("|"+"---|"*len(cells)) if i==0 else None
        if md_rows: markdown_tables.append("[TABLE_START]\n"+"\n".join(md_rows)+"\n[TABLE_END]")
    return "\n\n".join(markdown_tables)

def sse_payload_with_id(content: str,event_id: int) -> str: return f"id: {event_id}\nretry: 3000\ndata: {json.dumps({'content':content},ensure_ascii=False)}\n\n"
def cache_key_for(query: str) -> str:
    meta=load_index_meta(); fingerprint=meta.get("fingerprint","no-index-meta"); normalized=re.sub(r"\s+"," ",query.strip().lower()); return hashlib.sha256(f"{fingerprint}:{normalized}".encode("utf-8")).hexdigest()
async def get_cached_response_async(query: str) -> Optional[str]:
    async with response_cache_lock: return response_cache.get(cache_key_for(query))
async def set_cached_response_async(query: str,answer: str) -> None:
    async with response_cache_lock:
        if len(response_cache)>=RESPONSE_CACHE_MAX:
            first_key=next(iter(response_cache),None)
            if first_key: response_cache.pop(first_key,None)
        response_cache[cache_key_for(query)]=answer

@app.get("/api/observability/status")
async def observability_status():
    return {"rag_status":rag_status,"tracing_enabled":RAG_TRACING_ENABLED,"rrf_k":RRF_K,"image_captioning_enabled":RAG_IMAGE_CAPTIONING_ENABLED,"collection":COLLECTION_NAME}

@app.get("/api/observability/metrics")
async def observability_metrics():
    if not os.path.exists(RAG_TRACE_FILE): return {"tracing_enabled":RAG_TRACING_ENABLED,"traces":0,"spans":0,"stages":{}}
    spans=[]
    try:
        with open(RAG_TRACE_FILE,"r",encoding="utf-8") as handle:
            for line in handle:
                try: spans.append(json.loads(line))
                except json.JSONDecodeError: pass
    except OSError: pass
    stats={}; trace_ids=set()
    for item in spans:
        trace_ids.add(item.get("trace_id")); name=item.get("name","unknown"); st=stats.setdefault(name,{"count":0,"total_ms":0.0,"errors":0}); st["count"]+=1; st["total_ms"]+=float(item.get("duration_ms") or 0.0); st["errors"]+=1 if item.get("status")=="error" else 0
    for st in stats.values(): st["mean_ms"]=round(st["total_ms"]/st["count"],3) if st["count"] else 0.0; st["total_ms"]=round(st["total_ms"],3)
    return {"tracing_enabled":RAG_TRACING_ENABLED,"trace_file":RAG_TRACE_FILE,"traces":len(trace_ids),"spans":len(spans),"stages":stats}

@app.get("/api/observability/traces/{trace_id}")
async def observability_trace(trace_id: str):
    stages=[]
    if os.path.exists(RAG_TRACE_FILE):
        with open(RAG_TRACE_FILE,"r",encoding="utf-8") as handle:
            for line in handle:
                try:
                    item=json.loads(line)
                    if item.get("trace_id")==trace_id: stages.append(item)
                except json.JSONDecodeError: pass
    return {"trace_id":trace_id,"stages":stages}

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if req.chat_id: update_student_session(req.chat_id)
    return {"status":"success","response":await run_rag_query(req.message.strip())}

@app.post("/api/clear_context")
async def api_clear_context(req: ClearContextRequest):
    session_key=req.session_id or req.browserID
    if session_key: clear_user_history(session_key); active_stream_sessions.discard(session_key)
    return {"status":"success","message":"Context cleared"}

@app.post("/astream")
async def astream(req: AstreamRequest):
    session_key=req.session_id or req.browserID
    if session_key in active_stream_sessions: active_stream_sessions.discard(session_key)
    if session_key: update_student_session(session_key); active_stream_sessions.add(session_key)
    async def event_generator():
        seq=1; query_text=compress_long_query(req.input)
        try:
            if not query_text: yield sse_payload_with_id("請輸入問題後再送出。",seq); return
            cached=await get_cached_response_async(query_text)
            if cached: yield sse_payload_with_id(cached,seq); return
            if rag_chain is None: yield sse_payload_with_id("⏳ Alex 正在準備知識庫，請稍後再試。",seq); return
            tracer=WhiteBoxTracer(enabled=RAG_TRACING_ENABLED,sink_path=RAG_TRACE_FILE); token=_active_tracer.set(tracer)
            try:
                with tracer.span("rag.end_to_end.stream",metadata={"query_length":len(query_text),"model":OLLAMA_MODEL}):
                    history_messages=normalize_chat_history(req.chat_history); yielded=False; chunks=[]; context_docs=[]
                    async with asyncio.timeout(OLLAMA_REQUEST_TIMEOUT):
                        async for chunk in rag_chain.astream({"input":query_text,"chat_history":history_messages}):
                            if isinstance(chunk,dict) and "context" in chunk: context_docs=chunk["context"]
                            answer_chunk=chunk.get("answer") if isinstance(chunk,dict) else None
                            if not answer_chunk: continue
                            cleaned=sanitize_hallucinations(strip_think_tags(str(answer_chunk)))
                            if cleaned: yielded=True; chunks.append(cleaned); yield sse_payload_with_id(cleaned,seq); seq+=1
                    if context_docs:
                        ref_footer=format_reference_footer(context_docs)
                        if ref_footer: chunks.append(ref_footer); yield sse_payload_with_id(ref_footer,seq); seq+=1
                    if not yielded:
                        answer=await run_rag_query(query_text,req.chat_history); yield sse_payload_with_id(answer,seq)
                    elif chunks: await set_cached_response_async(query_text,"".join(chunks))
            finally: _active_tracer.reset(token)
        except asyncio.TimeoutError: yield sse_payload_with_id(f"⚠️ 本地 AI 模型 ({OLLAMA_MODEL}) 回應逾時。",seq)
        except Exception as e: logger.exception("Streaming RAG failed"); yield sse_payload_with_id(f"抱歉，系統運算時發生技術故障：{e}",seq)
        finally:
            if session_key: active_stream_sessions.discard(session_key)
    return StreamingResponse(event_generator(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

@app.get("/webapp",response_class=HTMLResponse)
async def webapp():
    html_content="""<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'><title>PolyU Academic Advisor</title><style>body{font-family:Arial,sans-serif;max-width:900px;margin:auto;padding:24px}textarea{width:100%;box-sizing:border-box}button{padding:10px 16px}</style></head><body><h1>PolyU AI Academic Advisor</h1><p>Telegram/WebApp interface is active.</p><textarea id='userInput' rows='4' placeholder='Ask about PolyU ISE...'></textarea><br><button id='sendButton'>Send</button><button id='clearBtn'>Clear</button><div id='messages'></div><script>let history=[];const input=document.getElementById('userInput');const messages=document.getElementById('messages');document.getElementById('sendButton').onclick=async()=>{const q=input.value.trim();if(!q)return;input.value='';const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});const d=await r.json();const p=document.createElement('p');p.innerHTML='<b>You:</b> '+q+'<br><b>Alex:</b> '+String(d.response).replace(/\n/g,'<br>');messages.prepend(p)};document.getElementById('clearBtn').onclick=async()=>{await fetch('/api/clear_context',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});messages.innerHTML=''};</script></body></html>"""
    return HTMLResponse(content=html_content)

@app.get("/")
async def get_chat_page(): return "<h1>PolyU AI Academic Advisor Backend Active</h1><p>Visit /webapp to access Telegram Mini App interface.</p>"

if __name__ == "__main__": uvicorn.run(app,host="0.0.0.0",port=8000)
