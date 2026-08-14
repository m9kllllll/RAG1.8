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

from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Standard LangChain chain factory functions
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-No-Cache": "true"
    }
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
        "scraper_version": "optimized-v9-docling-hybrid",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def load_index_meta() -> Dict[str, Any]:
    if not os.path.exists(INDEX_META_FILE):
        return {}
    try:
        with open(INDEX_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Could not load index metadata: {e}")
        return {}

def save_index_meta(fingerprint: str, points_count: int) -> None:
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "points_count": points_count, "updated_at": time.time()}, f, indent=2)

def is_valid_pdf(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


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
                    tmp_path = tmp.name
                    total = 0
                    async for chunk in resp.aiter_bytes():
                        if not chunk: continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"Download exceeded limit of {max_bytes} bytes")
                        tmp.write(chunk)

        if suffix == ".pdf" and not is_valid_pdf(tmp_path):
            logger.warning(f"⚠️ Downloaded file for {safe_url} is HTML/Invalid binary. Skipping.")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

        return tmp_path
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as te:
        logger.warning(f"⏱️ Network timeout/connection blocked for {safe_url}: {type(te).__name__}. Skipping.")
    except Exception as e:
        logger.warning(f"⚠️ Download failed for {safe_url}: [{type(e).__name__}] {str(e)}")
        
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return None

async def fetch_via_jina_ai_async(target_url: str) -> Optional[str]:
    sso_url_keywords = ["cas.polyu.edu.hk", "/_sso/", "sso.", "login", "signin", "auth"]
    if any(kw in target_url.lower() for kw in sso_url_keywords):
        logger.warning(f"🛡️ Skipping SSO/Authentication URL to avoid login wall: {target_url}")
        return None

    safe_url = urllib.parse.quote(urllib.parse.unquote(target_url.strip()), safe="%/:=&?~#+!$,;'@()*[]")
    jina_url = f"https://r.jina.ai/{safe_url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-No-Cache": "true"
    }
    try:
        logger.info(f"🤖 Requesting Jina AI parsing for: {safe_url}")
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, verify=False) as client:
            resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                text = resp.text.strip()
                if len(text) > 100:
                    text_lower = text.lower()
                    sso_content_indicators = [
                        "single sign-on", "netid", "polyu netid", 
                        "cas login", "sign in with your netid", "please log in"
                    ]
                    if any(indicator in text_lower for indicator in sso_content_indicators):
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
        "query_terms": [
            "CAR/GUR", "CAR", "GUR", "Cluster-Area Requirements", "General University Requirements",
            "學分要求", "通識", "大學核心", "PolyU ISE CAR credits", "Senior Year Intake GUR", "高年級銜接",
        ],
        "facts": [
            "核心學制關係：CAR (Cluster-Area Requirements) 是 GUR (General University Requirements) 旗下的子項目（包含關係），絕非相互獨立或相加的兩門課程。",
            "四年制新生（Year 1 Entry）GUR 要求：2025/26 學年起入學總 GUR 為 27 學分（含 9 學分 CAR）；2022/23 至 2024/25 學年入學總 GUR 為 30 學分（含 12 學分 CAR）。",
            "高年級銜接生（Senior Year Intake / Articulated Degree）：GUR 要求大幅豁免，通常只需修讀 9 個 GUR 學分（包含 6 個 CAR 學分），具體依入學審查免修結果為準。",
            "修讀彈性與限制：PolyU 並未硬性規定每學期必須修讀多少 CAR/GUR 學分，學生可自主彈性安排，無每學期最低門檻。",
            "畢業時間與排課建議：常規最高修讀上限為每學期 21 學分。學生通常會在大一與大二（前 3 到 4 個學期）將多數 GUR 與 CAR 完成，以便大三大四專心進行 Capstone 畢業論文與 WIE (Work-Integrated Education) 實習。",
            "CAR 語言要求：CAR 課程同時用作滿足英文閱讀寫作 (ER/EW) 及中文閱讀寫作 (CR/CW) 語言要求。",
        ],
        "answer_format": [
            "使用繁體中文回答，結構必須極度清晰。",
            "首先明確澄清：CAR 屬於 GUR 的一部分（包含關係），切勿將兩者學分簡單相加。",
            "區分四年制直入新生 (Year 1 Entry) 與高年級銜接生 (Senior Year Intake) 的不同學分門檻。",
            "說明排課彈性：每學期無硬性門檻，通常可在 3 至 4 個學期內修完。",
            "引導詢問用戶屬於哪一種入學身份（Year 1 Entry 還是 Senior Year Intake），以提供最精準的建議。",
        ],
        "sources": [CAR_GUR_SOURCE_URL, ISE_PROGRAMME_DETAILS_URL, ISE_CONTACT_URL],
    }
}

TERM_INDEX = {}
if os.path.exists("term_index.json"):
    try:
        with open("term_index.json", "r", encoding="utf-8") as f:
            TERM_INDEX = json.load(f)
        logger.info(f"✅ Loaded {len(TERM_INDEX)} terms from term_index.json")
    except Exception as e:
        logger.error(f"❌ Error loading term_index.json: {e}")


def sanitize_hallucinations(text: str) -> str:
    if not text:
        return ""
    pattern = r"Women\s*in\s*Engineering"
    if re.search(pattern, text, re.IGNORECASE):
        logger.warning("🚨 Hallucination caught: 'Women in Engineering' replaced with 'Work-Integrated Education'")
        text = re.sub(pattern, "Work-Integrated Education (校企協作教育)", text, flags=re.IGNORECASE)
    text = text.replace("WomeninEngineering", "Work-Integrated Education")
    return text


def normalize_and_expand_query(query: str) -> str:
    cantonese_map = {
        "有甚麼": "要求 指引",
        "有咩": "要求 指引",
        "點樣": "流程 方法",
        "幾多": "學分 數量",
        "邊啲": "課程 科目",
    }
    for cant, std in cantonese_map.items():
        query = query.replace(cant, std)
    
    if re.search(r'\bWIE\b', query, re.IGNORECASE):
        query += " Work-Integrated Education 校企協作教育 實習 實習要求 實習表格"
        
    if re.search(r'\bCAR\b', query, re.IGNORECASE):
        query += " Cluster-Area Requirements 通識教育 通識學分要求 GUR 包含關係"

    if re.search(r'\bGUR\b', query, re.IGNORECASE):
        query += " General University Requirements 大學核心課程要求"
        
    return query


def is_car_gur_query(query_text: str) -> bool:
    normalized = query_text.lower().replace("／", "/")
    has_car = bool(re.search(r"\bcar\b|cluster-area|cluster area|學群|群組", normalized, re.IGNORECASE))
    has_gur = bool(re.search(r"\bgur\b|general university requirements|大學核心|通識|學分要求", normalized, re.IGNORECASE))
    return has_car or has_gur

def expand_query_with_index(query_text: str) -> str:
    if not is_car_gur_query(query_text):
        return query_text
    index = CURATED_FACT_INDEX["car_gur"]
    return " | ".join([query_text, *index["query_terms"], *index["facts"][:4]])

def build_index_documents() -> List[Document]:
    docs = []
    for key, item in CURATED_FACT_INDEX.items():
        docs.append(Document(
            page_content="\n".join([
                f"Topic: {key}",
                "Search terms: " + ", ".join(item["query_terms"]),
                "Facts:",
                *[f"- {fact}" for fact in item["facts"]],
                "Answer format:",
                *[f"- {fmt}" for fmt in item["answer_format"]],
                "Sources:",
                *[f"- {source}" for source in item["sources"]],
            ]),
            metadata={
                "source": item["sources"][0],
                "category": f"Programmatic Index: {key}",
                "academic_level": "UG",
                "priority": "programmatic_index",
            },
        ))
    return docs

def build_term_index_documents() -> List[Document]:
    docs = []
    if not TERM_INDEX:
        return docs
        
    for key, item in TERM_INDEX.items():
        if isinstance(item, dict):
            content_lines = [
                f"Term / Keyword: {key}",
                f"English Name: {item.get('english', '')}",
                f"Chinese Name: {item.get('chinese', '')}",
                f"Abbreviation: {item.get('abbreviation', key)}",
                f"Category: {item.get('category', 'Glossary')}"
            ]
            if item.get("programme_code"):
                content_lines.append(f"Programme Code: {item['programme_code']}")
            if item.get("jupas_code"):
                content_lines.append(f"JUPAS Code: {item['jupas_code']}")
                
            content = "\n".join(content_lines)
        else:
            content = f"Term: {key}\nDefinition: {item}"
            
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "term_index.json",
                "category": item.get("category", "Glossary Index") if isinstance(item, dict) else "Glossary Index",
                "priority": "programmatic_index",
            }
        ))
    logger.info(f"✅ Built {len(docs)} enriched documents from term_index.json")
    return docs

def extract_course_codes(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z]{2,5}\d{3,5}\b", text.upper())))

def classify_query(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    course_codes = extract_course_codes(query)
    comparison_terms = ["compare", "versus", " vs ", "better", "suitable", "比較", "分別", "哪個"]
    prerequisite_terms = ["prerequisite", "pre-requisite", "先修", "要求"]
    programme_terms = ["programme", "curriculum", "credit", "car", "gur", "wie", "課程", "學分", "實習"]

    if any(term in query_lower for term in comparison_terms) and len(course_codes) >= 2:
        query_type, complexity = "COURSE_COMPARISON", "complex"
    elif course_codes and any(term in query_lower for term in prerequisite_terms):
        query_type, complexity = "PREREQUISITE", "medium"
    elif course_codes:
        query_type, complexity = "COURSE_LOOKUP", "simple"
    elif any(term in query_lower for term in programme_terms):
        query_type, complexity = "PROGRAMME", "medium"
    else:
        query_type, complexity = "GENERAL_POLYU", "medium"

    if complexity == "simple":
        vector_k, bm25_k, final_k = 8, 8, 3
    elif complexity == "complex":
        vector_k, bm25_k, final_k = 15, 15, 5
    else:
        vector_k, bm25_k, final_k = 10, 10, 4

    bm25_weight, vector_weight = (0.65, 0.35) if course_codes else (0.50, 0.50)

    return {
        "query_type": query_type,
        "complexity": complexity,
        "course_codes": course_codes,
        "vector_k": vector_k,
        "bm25_k": bm25_k,
        "final_k": final_k,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
    }

def compress_document_context(doc: Document, query: str, max_chars: int = 2000) -> Document:
    meta = doc.metadata.copy()
    meta.setdefault("source", "PolyU ISE Official Resource")
    meta.setdefault("category", "Official Document")
    meta.setdefault("_score", 0.50)
    meta.setdefault("_confidence", "MEDIUM")
    
    if meta.get("is_table") and meta.get("raw_table"):
        meta["context_compressed"] = False
        raw_html = meta["raw_table"]
        full_content = f"Official Table Data:\n{raw_html}"
        return Document(page_content=full_content, metadata=meta)

    if len(doc.page_content) <= max_chars:
        return Document(page_content=doc.page_content, metadata=meta)
        
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", query)]
    sentences = re.split(r"(?<=[。.!?])\s+|\n+", doc.page_content)
    selected = [s.strip() for s in sentences if s.strip() and any(t in s.lower() for t in terms)]
    content = "\n".join(selected[:15]) if selected else doc.page_content[:max_chars]
    
    if len(content) > max_chars:
        content = content[:max_chars].rsplit(" ", 1)[0]
        
    meta["context_compressed"] = True
    return Document(page_content=content, metadata=meta)

def extract_tables_and_text(text: str) -> List[Dict[str, Any]]:
    blocks = []
    pattern = r'(\[TABLE_START\][\s\S]*?\[TABLE_END\])'
    parts = re.split(pattern, text)
    
    for part in parts:
        if not part.strip():
            continue
        if part.startswith("[TABLE_START]") and part.endswith("[TABLE_END]"):
            blocks.append({"type": "table", "content": part.strip()})
        else:
            blocks.append({"type": "text", "content": part.strip()})
            
    return blocks

def sentence_level_split(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def advanced_multi_strategy_chunker(
    documents: List[Document], 
    target_chunk_size: int = 1200, 
    chunk_overlap: int = 150
) -> List[Document]:
    final_chunks: List[Document] = []

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for doc in documents:
        base_metadata = doc.metadata.copy()
        raw_text = doc.page_content

        blocks = extract_tables_and_text(raw_text)

        for block_idx, block in enumerate(blocks):
            block_content = block["content"]
            
            if block["type"] == "table":
                table_meta = base_metadata.copy()
                table_meta.update({
                    "is_table": True,
                    "chunk_strategy": "table_aware",
                    "block_index": block_idx
                })
                final_chunks.append(Document(page_content=block_content, metadata=table_meta))
                continue

            header_splits = markdown_splitter.split_text(block_content)

            for h_split in header_splits:
                heading_path = " > ".join(
                    [val for key, val in h_split.metadata.items() if key.startswith("Header_")]
                )
                
                heading_metadata = base_metadata.copy()
                heading_metadata.update(h_split.metadata)
                heading_metadata.update({
                    "heading_path": heading_path if heading_path else "Root Section",
                    "is_table": False
                })

                if len(h_split.page_content) > target_chunk_size:
                    rec_splits = recursive_splitter.split_text(h_split.page_content)

                    for rec_text in rec_splits:
                        if len(rec_text) > target_chunk_size * 1.2:
                            sent_splits = sentence_level_split(rec_text, max_chunk_size=target_chunk_size)
                            for sent_text in sent_splits:
                                meta = heading_metadata.copy()
                                meta.update({"chunk_strategy": "heading+recursive+sentence"})
                                final_chunks.append(Document(page_content=sent_text, metadata=meta))
                        else:
                            meta = heading_metadata.copy()
                            meta.update({"chunk_strategy": "heading+recursive"})
                            final_chunks.append(Document(page_content=rec_text, metadata=meta))
                else:
                    meta = heading_metadata.copy()
                    meta.update({"chunk_strategy": "heading_direct"})
                    final_chunks.append(Document(page_content=h_split.page_content, metadata=meta))

    for chunk in final_chunks:
        chunk.metadata["course_codes"] = extract_course_codes(chunk.page_content)

    logger.info(f"⚡ Enhanced Multi-Strategy Processing: Created {len(final_chunks)} structured chunks.")
    return final_chunks

def init_sqlite_db():
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
    logger.info("✅ SQLite database initialized successfully.")

def update_student_session(chat_id: str, faculty: Optional[str] = None):
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
        logger.error(f"❌ SQLite session update failed: {e}")

def clear_user_history(user_id: str) -> bool:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_sessions WHERE student_chat_id = ?", (str(user_id),))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"⚠️ Error clearing SQLite history for user {user_id}: {e}")
        return False

def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return cleaned.strip()

def split_text(text: str, max_length: int = 4000) -> List[str]:
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

async def send_chunked_message(update: Update, text: str, parse_mode: str = "Markdown", reply_to_message_id: Optional[int] = None):
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        msg_reply_id = reply_to_message_id if i == 0 else None
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode, reply_to_message_id=msg_reply_id)
        except Exception:
            await update.message.reply_text(chunk, reply_to_message_id=msg_reply_id)

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

def extract_tables_as_md(page) -> str:
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
        md.insert(1, "|" + "---|"*num_cols)
        blocks.append("\n".join(md))
    if blocks:
        return "\n\n[TABLE_START]\n" + "\n\n".join(blocks) + "\n[TABLE_END]\n"
    return ""

def load_pdf_with_structure(pdf_path: str) -> List[Document]:
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


async def load_pdf_with_docling_async(pdf_path: str, source_url: str) -> List[Document]:
    """Parses PDF using IBM Docling; falls back gracefully if unavailable."""
    if DocumentConverter is not None:
        def _docling_parse():
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Skip OCR for speed on digital PDFs
            pipeline_options.do_table_structure = True

            converter = DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
            )
            result = converter.convert(pdf_path)
            return result.document.export_to_markdown()

        try:
            logger.info(f"📄 Parsing PDF with Docling: {source_url}")
            markdown_text = await asyncio.to_thread(_docling_parse)

            if markdown_text and len(markdown_text.strip()) > 50:
                return [Document(
                    page_content=markdown_text,
                    metadata={
                        "source": source_url,
                        "category": "Docling Parsed PDF",
                        "academic_level": classify_academic_level(source_url, markdown_text)
                    }
                )]
        except Exception as e:
            logger.warning(f"⚠️ Docling parsing failed for {source_url}: {e}. Triggering standard PDF fallback...")

    return await load_pdf_safely_async(pdf_path, source_url)


async def load_pdf_safely_async(pdf_path: str, source_url: str) -> List[Document]:
    docs = []
    try:
        loader = PyPDFLoader(pdf_path)
        docs = await asyncio.to_thread(loader.load)
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"⚠️ PyPDFLoader failed for {source_url}: {e}. Trying lenient PdfReader...")

    def _read_pdf_lenient():
        reader = PdfReader(pdf_path, strict=False)
        extracted = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                extracted.append(Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": source_url}
                ))
        return extracted

    try:
        docs = await asyncio.to_thread(_read_pdf_lenient)
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"⚠️ Lenient PdfReader failed for {source_url}: {e}. Trying pdfplumber...")

    try:
        docs = await asyncio.to_thread(load_pdf_with_structure, pdf_path)
        for doc in docs:
            doc.metadata["source"] = source_url
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"⚠️ pdfplumber failed for {source_url}: {e}. Triggering Jina AI fallback...")

    if source_url.startswith(("http://", "https://")):
        jina_text = await fetch_via_jina_ai_async(source_url)
        if jina_text:
            return [Document(
                page_content=jina_text,
                metadata={"source": source_url, "category": "Jina AI Parsed PDF"}
            )]

    return docs

async def scrape_webpage_async(url: str) -> List[Document]:
    scraped_docs = []
    try:
        logger.info(f"🌐 Scraping webpage with Jina AI: {url}")
        web_text = await fetch_via_jina_ai_async(url)
        if web_text:
            level = classify_academic_level(url, web_text)
            scraped_docs.append(Document(
                page_content=web_text,
                metadata={"source": url, "category": "Official Webpage (Jina AI)", "academic_level": level}
            ))
    except Exception as e:
        logger.error(f"⚠️ Webpage scraping error for {url}: {e}")
    return scraped_docs

def load_prd_by_sections(pdf_path: str, source_url: str, programme: str = "General") -> List[Document]:
    full_text = ""
    page_starts = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_starts.append((len(full_text), i + 1)) 
            text = page.extract_text() or ""
            text += extract_tables_as_md(page)
            full_text += text + "\n\n"

    section_re = re.compile(r'SECTION\s+\d+.*?$', re.MULTILINE | re.IGNORECASE)
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": source_url,
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

        if "SYLLABUS" in section_title.upper() or "SUBJECT" in section_title.upper():
            subject_matches = list(subject_form_re.finditer(section_text))

            if len(subject_matches) > 1:
                first_hit = subject_matches[0].start()
                if first_hit > 10:
                    header_chunk = section_text[:first_hit].strip()
                    documents.append(Document(
                        page_content=header_chunk,
                        metadata={
                            "source": source_url,
                            "original_filename": os.path.basename(source_url),
                            "programme": programme,
                            "section": section_title,
                            "category": "Official Programme Definition Document (PRD)",
                            "chunk_type": "section"
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
                            "source": source_url,
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
                            "source": source_url,
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

        documents.append(Document(
            page_content=section_text,
            metadata={
                "source": source_url,
                "programme": programme,
                "section": section_title,
                "category": "Official PDF",
                "chunk_type": "section",
                "page_start": page_at(sec_start),
                "page_end": page_at(sec_end)
            }
        ))

    return documents


class ScoreInjectingRetriever(BaseRetriever):
    vectorstore: Any = Field(description="The underlying Qdrant vector store")
    bm25: Any = Field(default=None, description="Optional BM25 lexical retriever")
    k: int = Field(default=4)
    score_threshold: float = Field(default=0.45)

    def _perform_hybrid_search(self, raw_query: str) -> List[Document]:
        query = normalize_and_expand_query(raw_query)
        plan = classify_query(query)
        candidate_docs: Dict[str, Dict[str, Any]] = {}
        start = time.perf_counter()

        def vector_lookup():
            return self.vectorstore.similarity_search_with_score(query, k=plan["vector_k"])

        def bm25_lookup():
            if not self.bm25:
                return []
            try:
                return self.bm25.invoke(query)
            except AttributeError:
                return self.bm25.get_relevant_documents(query)

        docs_and_scores = vector_lookup()
        bm25_docs = bm25_lookup()

        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            key = self._doc_key(doc)
            candidate_docs[key] = {
                "doc": doc,
                "vector_score": self._normalize_vector_score(score),
                "bm25_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": None,
            }

        for rank, doc in enumerate(bm25_docs[: plan["bm25_k"]], start=1):
            key = self._doc_key(doc)
            rank_score = 1.0 / rank
            if key not in candidate_docs:
                candidate_docs[key] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": rank_score,
                    "vector_rank": None,
                    "bm25_rank": rank,
                }
            else:
                candidate_docs[key]["bm25_score"] = rank_score
                candidate_docs[key]["bm25_rank"] = rank

        ranked_docs = []
        query_lower = query.lower()
        is_pg_query = any(kw in query_lower for kw in ["master", "msc", "postgraduate", "pgd", "pg"])
        is_wie_query = "wie" in raw_query.lower() or "實習" in raw_query
        query_codes = set(plan["course_codes"])

        for item in candidate_docs.values():
            doc = item["doc"]
            vector_score = item["vector_score"]
            bm25_score = item["bm25_score"]
            effective_score = (plan["vector_weight"] * vector_score) + (plan["bm25_weight"] * bm25_score)
            
            source = str(doc.metadata.get("source", "")).lower()
            content = doc.page_content
            content_lower = content.lower()
            level_meta = str(doc.metadata.get("academic_level", "")).upper()

            if is_wie_query and ("work-integrated" in content_lower or "wie" in content_lower or "實習" in content_lower):
                effective_score += 0.35

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

            if "prd" in source or "programme_def" in source:
                effective_score += 0.15
            elif "student_handbook" in source:
                effective_score -= 0.05

            if "programme structure" in content_lower:
                effective_score += 0.20
            if "compulsory subjects" in content_lower:
                effective_score += 0.25
            if "elective subjects" in content_lower:
                effective_score += 0.15
            if "credit requirements" in content_lower or "graduation requirements" in content_lower:
                effective_score += 0.20
            if doc.metadata.get("priority") == "programmatic_index":
                effective_score += 0.35
            if query_codes and query_codes.intersection(set(doc.metadata.get("course_codes", []))):
                effective_score += 0.30

            effective_score = min(max(effective_score, 0.0), 1.0)

            if effective_score >= self.score_threshold:
                new_meta = doc.metadata.copy()
                new_meta["_score"] = round(effective_score, 3)
                new_meta["_vector_score"] = round(vector_score, 3)
                new_meta["_bm25_score"] = round(bm25_score, 3)
                new_meta["_vector_rank"] = item["vector_rank"]
                new_meta["_bm25_rank"] = item["bm25_rank"]
                new_meta["_query_type"] = plan["query_type"]
                new_meta["_complexity"] = plan["complexity"]
                new_meta["_confidence"] = "HIGH" if effective_score >= 0.80 else "MEDIUM"
                ranked_docs.append(compress_document_context(Document(page_content=doc.page_content, metadata=new_meta), query))

        ranked_docs.sort(key=lambda d: d.metadata.get("_score", 0), reverse=True)
        result_docs = ranked_docs[: plan["final_k"]] if len(ranked_docs) > plan["final_k"] else ranked_docs
        logger.info(
            "🔎 Retrieval plan=%s/%s candidates=%s results=%s elapsed=%.2fs",
            plan["query_type"], plan["complexity"], len(candidate_docs), len(result_docs), time.perf_counter() - start
        )
        return result_docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        ranked_docs = self._perform_hybrid_search(query)
        top_score = ranked_docs[0].metadata.get("_score", 0.0) if ranked_docs else 0.0

        if top_score < self.score_threshold and len(ranked_docs) == 0:
            logger.info(f"🌐 啟動 Essential-Info-Tool 進行 PolyU ISE 官方資料檢索 (Top Score: {top_score:.2f})...")
            
            try:
                official_result = asyncio.run(essential_info_tool.ainvoke({"query": query}))
            except Exception:
                official_result = None
            
            if official_result and "⚠️" not in official_result:
                official_doc = Document(
                    page_content=official_result,
                    metadata={
                        "source": ISE_PROGRAMME_DETAILS_URL,
                        "priority": "essential_tool",
                        "_score": 0.95,
                        "_confidence": "HIGH",
                        "category": "PolyU ISE Official Live Data"
                    }
                )
                ranked_docs.insert(0, official_doc)

        return ranked_docs[: self.k]

    @staticmethod
    def _normalize_vector_score(raw_score: float) -> float:
        score = float(raw_score)
        if score > 1.0:
            return max(0.0, 1.0 / (1.0 + score))
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", doc.metadata.get("page_start", "")))
        return f"{source}:{page}:{doc.page_content[:240]}"


async def get_rag_chain_async():
    global bm25_retriever, vector_store
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_FAST_MODEL)

    logger.info("🔌 Initializing Ollama & Qdrant Client...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120, check_compatibility=False)

    config_data = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as cfg:
            config_data = json.load(cfg)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info(f"✨ Created Qdrant collection: {COLLECTION_NAME}")

    v_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    vector_store = v_store

    info = client.get_collection(COLLECTION_NAME)
    points_count = info.points_count if info.points_count is not None else 0
    fingerprint = compute_index_fingerprint(config_data)
    cached_meta = load_index_meta()

    all_docs = []
    
    if config_data:
        urls = config_data.get("urls", [])
        pdf_paths = config_data.get("pdfs", [])

        sem = asyncio.Semaphore(2)

        if urls:
            set_rag_status("scraping", f"Scraping {len(urls)} configured webpages with Jina AI")
            async def safe_web_scrape(u):
                async with sem:
                    return await scrape_webpage_async(u)
            
            tasks = [safe_web_scrape(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_docs.extend(res)

        if pdf_paths:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*"
            }
            for pdf_url in pdf_paths:
                async with sem:
                    try:
                        temp_pdf = await download_to_tempfile_async(pdf_url, ".pdf", headers, timeout=60)
                        if not temp_pdf:
                            continue

                        url_upper = pdf_url.upper()
                        if any(x in url_upper for x in ["45499-LEM", "LEM"]):
                            prog = "LEM"
                        elif any(x in url_upper for x in ["45499-EEM", "EEM"]):
                            prog = "EEM"
                        elif any(x in url_upper for x in ["45498-ISE", "ISE"]):
                            prog = "ISE"
                        elif any(x in url_upper for x in ["45498-PEM", "PEM"]):
                            prog = "PEM"
                        elif any(x in url_upper for x in ["PIE", "45498"]):
                            prog = "PIE"
                        elif any(x in url_upper for x in ["AOS", "45497"]):
                            prog = "AOS"
                        else:
                            prog = "General"

                        if "PRD" in pdf_url.upper() or "45499" in pdf_url.upper():
                            docs = await asyncio.to_thread(load_prd_by_sections, temp_pdf, pdf_url, prog)
                        else:
                            docs = await load_pdf_with_docling_async(temp_pdf, pdf_url)
                        
                        clean_filename = os.path.basename(urllib.parse.urlparse(pdf_url).path) or os.path.basename(pdf_url)
                        
                        for doc in docs:
                            doc.metadata["source"] = pdf_url
                            doc.metadata["original_filename"] = clean_filename
                            doc.metadata["programme"] = prog
                            doc.metadata["academic_level"] = classify_academic_level(pdf_url, doc.page_content)

                        all_docs.extend(docs)

                        if os.path.exists(temp_pdf):
                            os.unlink(temp_pdf)

                    except Exception as e:
                        logger.error(f"⚠️ PDF Pipeline Error for {pdf_url}: {e}")

    all_docs.extend(build_index_documents())
    all_docs.extend(build_term_index_documents())

    splits = []
    if all_docs:
        logger.info(f"Total raw docs ingested dynamically: {len(all_docs)}")
        splits = advanced_multi_strategy_chunker(documents=all_docs, target_chunk_size=1200, chunk_overlap=150)
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 10
    else:
        bm25_retriever = None

    if splits and (FORCE_REINDEX or points_count < 1000 or cached_meta.get("fingerprint") != fingerprint):
        logger.info(f"📤 Uploading {len(splits)} document chunks to Qdrant...")
        batch_size = QDRANT_BATCH_SIZE
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            try: 
                v_store.add_documents(batch)
            except Exception as e: 
                logger.error(f"❌ Upload Batch Failed [{i}:{i+batch_size}]: {e}")
        logger.info("✅ Qdrant indexing completed.")
        save_index_meta(fingerprint, client.get_collection(COLLECTION_NAME).points_count)

    return build_rag_chain(v_store, bm25_retriever, ollama_model, ollama_url)

def clean_source_info(doc: Document) -> Tuple[str, str]:
    src = str(doc.metadata.get("source", "")).strip()
    category = str(doc.metadata.get("category", "")).strip()
    link_text = str(doc.metadata.get("link_text", "")).strip()
    programme = str(doc.metadata.get("programme", "")).strip()
    original_file = str(doc.metadata.get("original_filename", "")).strip()

    if link_text and not link_text.startswith("http"):
        title = link_text
    elif category and category not in ["Official PDF", "Official Document"]:
        title = category
    elif programme and programme != "General":
        title = f"PolyU ISE {programme} Academic Document"
    else:
        title = "PolyU ISE Academic Guidelines"

    if src.startswith("http://") or src.startswith("https://"):
        filename = os.path.basename(urllib.parse.urlparse(src).path)
        if filename and filename.lower().endswith((".pdf", ".docx")):
            title = f"{title} ({filename})"
        return title, src

    if "/tmp/" in src or "tmp" in os.path.basename(src):
        clean_name = os.path.basename(original_file) if original_file else "PolyU_Academic_Guide.pdf"
        return f"{title} ({clean_name})", ""

    clean_name = os.path.basename(src) if src else "Official Document"
    return f"{title} ({clean_name})", ""


def format_reference_footer(context_docs: List[Document], min_score: float = 0.45) -> str:
    if not context_docs:
        return ""

    references = []
    seen_keys = set()

    for doc in context_docs:
        score = doc.metadata.get("_score", 0.0)
        
        if score < min_score:
            continue

        conf = doc.metadata.get("_confidence", "HIGH" if score >= 0.80 else "MEDIUM")
        display_title, clean_url = clean_source_info(doc)

        dedup_key = clean_url if clean_url else display_title
        if not dedup_key or dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        if clean_url:
            ref_line = f"- [{display_title}]({clean_url}) | 🎯 置信度：{score:.2f} ({conf})"
        else:
            ref_line = f"- 📄 **{display_title}** | 🎯 置信度：{score:.2f} ({conf})"

        references.append(ref_line)

    if not references:
        return ""

    return "\n\n---\n### 📚 參考資料與置信度 (References & Confidence Score)\n" + "\n".join(references)


def build_rag_chain(v_store, bm25, ollama_model: str, ollama_url: str):
    global vector_store
    vector_store = v_store

    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=0.2,
        top_p=0.9,
        num_predict=OLLAMA_NUM_PREDICT,
        num_ctx=OLLAMA_NUM_CTX,
    )
    
    retriever = ScoreInjectingRetriever(vectorstore=v_store, bm25=bm25, k=4, score_threshold=0.45)

    document_prompt = PromptTemplate.from_template(
        "Document: {category}\nSource: {source}\nContent:\n{page_content}\n\n"
    )

    system_prompt = (
        "你是 Alex，香港理工大學 (PolyU) 工業及系統工程學系 (ISE) 的官方學術諮詢助手。\n\n"
        "⚡ 速度與思考過程優化 (FAST ANSWER RULE):\n"
        "務必直接回答問題，嚴禁輸出任何 <think> 標籤或內部推理過程 (Do NOT generate <think> tokens)。立即給出最終答案。\n\n"
        "⚠️ 專有名詞與學制邏輯極重要約束 (CRITICAL ACADEMIC LOGIC RULES):\n"
        "1. **CAR 屬於 GUR 的一部分 (包含關係)**：在 PolyU 學制中，CAR (Cluster-Area Requirements) 本來就是 GUR (General University Requirements) 底下的子項目。**兩者絕不是相加關係**，切勿將 CAR 與 GUR 學分相加作為獨立課業！\n"
        "2. **入學身份與學分差異**：\n"
        "   - **四年制直入新生 (Year 1 Entry)**：2025/26起入學總 GUR 為 27 學分（含 9 學分 CAR）；2022/23–2024/25 年度入學總 GUR 為 30 學分（含 12 學分 CAR）。\n"
        "   - **高年級銜接生 (Senior Year Intake)**：GUR 要求大幅豁免，通常僅需修讀 9 個 GUR 學分（含 6 個 CAR 學分）。\n"
        "3. **修讀彈性與時間**：PolyU **沒有**規定每學期必須修讀最低 CAR/GUR 學分。學生通常能在 3 到 4 個學期內（大一及大二）順利修完所有 CAR/GUR 要求。\n"
        "4. **WIE 專屬定義**：'WIE' 唯一代表 **Work-Integrated Education (校企協作教育 / 實習)**，絕對不可解釋為 'Women in Engineering'！\n\n"
        "📖 詳細學術解答優先原則 (COMPREHENSIVE ANSWER FIRST):\n"
        "1. 針對提問必須先提供完整、結構化的文字說明（包含總學分、各範疇要求、排課彈性）。\n"
        "2. 簡要詢問或引導用戶確認其入學身份（四年制新生 Year 1 Entry 還是高年級銜接生 Senior Year Intake），以給予最精準的規劃提示。\n\n"
        "🎨 結構化排版規則:\n"
        "1. **資訊層次**：使用 Markdown 標題 (###) 將內容分塊（如：### 📋 學制度與學分關係、### 📌 建議排課進度）。\n"
        "2. **重點標示**：使用粗體 (**...**) 強調關鍵學分與重要條款。\n"
        "3. **條列分明**：使用無序列表 (-) 將條件拆解，避免冗長文字牆。\n\n"
        "📊 官方文件與下載連結 (CONDITIONAL TABLES):\n"
        "只有當 Context 中含有明確的 **官方文件名稱與有效下載網址** 時，才在回答末尾附上表格。\n"
        "表格必須使用標準 Markdown 格式，並確保每行正確換行：\n\n"
        "| 文件/表格名稱 | 參考/下載連結 |\n"
        "|---|---|\n"
        "| [文件名稱] | [點擊下載/檢視](URL) |\n\n"
        "回答語氣與嚴謹性：\n"
        "使用親切、專業的繁體中文回答（若用戶使用英文則用英文回答）。切勿憑空撰寫未經證實的課程規則或無效網址。\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(
        llm, 
        qa_prompt, 
        document_prompt=document_prompt
    )
    
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain, v_store

async def run_rag_query(query_text: str, chat_history: Optional[List[ChatHistoryItem]] = None) -> str:
    global rag_chain
    if rag_chain is None:
        elapsed = round(time.time() - rag_status.get("started_at", time.time()), 1)
        return f"⏳ Alex 正在準備知識庫：{rag_status.get('message', 'starting')}（已用 {elapsed} 秒）。請稍後再試。"
    
    normalized_query = compress_long_query(query_text)

    cached = await get_cached_response_async(normalized_query)
    if cached:
        logger.info("⚡ Response cache hit for normalized query.")
        return cached
        
    history_messages = normalize_chat_history(chat_history or [])
    try:
        result = await asyncio.wait_for(
            rag_chain.ainvoke({"input": normalized_query, "chat_history": history_messages}),
            timeout=OLLAMA_REQUEST_TIMEOUT
        )
        
        raw_answer = result.get("answer", "抱歉，我無法檢索到相關解答。")
        clean_answer = sanitize_hallucinations(strip_think_tags(raw_answer))
        
        context_docs = result.get("context", [])
        ref_footer = format_reference_footer(context_docs)
        final_answer = clean_answer + ref_footer
        
        await set_cached_response_async(normalized_query, final_answer)
        return final_answer
    except asyncio.TimeoutError:
        return f"⚠️ 本地 AI 模型 ({OLLAMA_MODEL}) 回應逾時。"
    except Exception as e:
        return f"抱歉，系統運算時發生技術故障：{e}"

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    update_student_session(str(chat_id))
    app_url = f"{WEBAPP_URL}/webapp"
    keyboard = [
        [InlineKeyboardButton("🚀 啟動 Academic Advisor Web App", web_app=WebAppInfo(url=app_url))],
        [InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)", callback_data="faculty_ise")],
        [InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)", callback_data="faculty_gur")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (
        f"👋 歡迎使用香港理工大學 (PolyU) 學術諮詢 AI 助手 (Alex)！\n\n"
        "點擊下方按鈕啟動全新的 **Telegram Web App**，或直接在聊天室提問："
    )
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "📚 **Alex Knowledge Base & Advice Scope**\n\n"
        "I am trained on official guidelines from the Department of Industrial and Systems Engineering (ISE) at PolyU.\n\n"
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
        update_student_session(str(chat_id), faculty="ISE")
        keyboard = [
            [InlineKeyboardButton("📋 CAR / GUR 學分要求", callback_data="ise_car")],
            [InlineKeyboardButton("💼 WIE 實習 / 課外活動要求", callback_data="ise_wie")],
            [InlineKeyboardButton("🎓 Capstone 畢業論文選題", callback_data="ise_capstone")],
            [InlineKeyboardButton("🔙 返回主選單", callback_data="go_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text="📍 **你已進入 ISE 學術諮詢專區**\n請選擇你想了解的疑問範疇：",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    elif query.data == "ise_wie":
        update_student_session(str(chat_id))
        await query.edit_message_text(text="🔍 正在為你檢索 PolyU ISE WIE 實習要求及相關表格，請稍候...")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        response = await run_rag_query("請問 ISE 學生 WIE 實習的要求是什麼？有哪些表格可以下載？")
        try: await query.delete_message()
        except Exception: pass
        await send_chunked_message(update, response, parse_mode="Markdown")
    elif query.data == "ise_car":
        update_student_session(str(chat_id))
        await query.edit_message_text(text="🔍 正在為你檢索 PolyU CAR 要求，請稍候...")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        response = await run_rag_query("請問 ISE 學生 CAR 和 GUR 的學分要求是什麼？")
        try: await query.delete_message()
        except Exception: pass
        await send_chunked_message(update, response, parse_mode="Markdown")
    elif query.data == "go_main":
        update_student_session(str(chat_id), faculty="General")
        app_url = f"{WEBAPP_URL}/webapp"
        keyboard = [
            [InlineKeyboardButton("🚀 啟動 Academic Advisor Web App", web_app=WebAppInfo(url=app_url))],
            [InlineKeyboardButton("工程學院 - 工業及系統工程學系 (ISE)", callback_data="faculty_ise")],
            [InlineKeyboardButton("其他學院 / 通識教育 (GUR/CAR)", callback_data="faculty_gur")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("請選擇你所屬的學系或諮詢範疇：", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    student_text = (update.message.text or "").strip()
    if not student_text:
        await update.message.reply_text("請輸入問題後再送出。")
        return
    chat_id = update.effective_chat.id
    update_student_session(str(chat_id))
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder_msg = await update.message.reply_text("🤔 Alex 正在思考並查閱 PolyU 學術指引...")
    
    ai_response = await run_rag_query(student_text)
    try: await placeholder_msg.delete()
    except Exception: pass
    await send_chunked_message(update, ai_response, parse_mode="Markdown")

async def load_rag_in_background_async():
    global rag_chain, vector_store
    logger.info("🚀 Loading RAG Vector Database & Ollama Model in background...")
    try:
        set_rag_status("loading", "Initializing Qdrant, BM25, and Ollama asynchronously")
        rag_chain, vector_store = await get_rag_chain_async()
        set_rag_status("ready", "RAG Chain is fully loaded and ready")
    except Exception as e:
        set_rag_status("failed", str(e))
        logger.error(f"❌ Failed to load RAG chain: {e}")

async def start_telegram_bot():
    global tg_app
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.warning("⚠️ No valid TELEGRAM_BOT_TOKEN set in .env. Skipping Telegram setup.")
        return

    custom_request = HTTPXRequest(
        connection_pool_size=4,
        read_timeout=20.0,
        write_timeout=20.0,
        connect_timeout=20.0,
        pool_timeout=20.0,
    )

    tg_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(custom_request)
        .build()
    )
    
    tg_app.add_handler(CommandHandler("start", start_command))
    tg_app.add_handler(CommandHandler("help", help_command))
    tg_app.add_handler(CommandHandler("clear", clear_command))
    tg_app.add_handler(CallbackQueryHandler(button_click))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    try:
        await tg_app.initialize()
        await tg_app.start()
        asyncio.create_task(tg_app.updater.start_polling(poll_interval=3.0, drop_pending_updates=True))
    except Exception as e:
        logger.warning(f"⚠️ Telegram bot startup encountered a network hiccup: {e}. Running in backend-only mode.")
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sqlite_db()
    asyncio.create_task(load_rag_in_background_async())
    await start_telegram_bot()
    yield
    if tg_app:
        try:
            if tg_app.updater and tg_app.updater.running:
                await tg_app.updater.stop()
            await tg_app.stop()
            await tg_app.shutdown()
        except Exception as e:
            logger.info(f"ℹ️ Telegram graceful shutdown completed with network notice: {type(e).__name__}")

app = FastAPI(title="PolyU AI Academic Advisor WebApp", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def review_docling_conversion(file_path: str, output_md_path: str = "review_output.md") -> str:
    if not os.path.exists(file_path) and not file_path.startswith("http"):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    converter = DocumentConverter()
    print(f"Converting '{file_path}' using Docling...")
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Markdown output saved successfully to: {output_md_path}")
    return output_md_path


def compress_long_query(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_QUERY_LENGTH:
        return text
    return f"{text[: MAX_QUERY_LENGTH - 160]}\n\n[Query compressed]"

def normalize_chat_history(history: List[ChatHistoryItem]):
    messages = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        content = item.content.strip()
        if not content: continue
        if item.role.lower() == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages

def extract_html_tables_as_markdown(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    markdown_tables = []
    
    for table in tables:
        rows = table.find_all("tr")
        if not rows: continue
        
        md_rows = []
        for i, row in enumerate(rows):
            cells = []
            for cell in row.find_all(["th", "td"]):
                link = cell.find('a')
                if link and link.get('href'):
                    cell_text = f"[{link.get_text(strip=True).replace('|', '')}]({link.get('href')})"
                else:
                    cell_text = cell.get_text(strip=True).replace("|", "\\|")
                cells.append(cell_text)
                
            if not cells: continue
            md_rows.append("| " + " | ".join(cells) + " |")
            
            if i == 0:
                md_rows.append("|" + "---|"*len(cells))
                
        if md_rows:
            markdown_tables.append("[TABLE_START]\n" + "\n".join(md_rows) + "\n[TABLE_END]")
            
    return "\n\n".join(markdown_tables)

def sse_payload_with_id(content: str, event_id: int) -> str:
    return f"id: {event_id}\nretry: 3000\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

def cache_key_for(query: str) -> str:
    meta = load_index_meta()
    fingerprint = meta.get("fingerprint", "no-index-meta")
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    return hashlib.sha256(f"{fingerprint}:{normalized}".encode("utf-8")).hexdigest()

async def get_cached_response_async(query: str) -> Optional[str]:
    async with response_cache_lock:
        return response_cache.get(cache_key_for(query))

async def set_cached_response_async(query: str, answer: str) -> None:
    async with response_cache_lock:
        if len(response_cache) >= RESPONSE_CACHE_MAX:
            first_key = next(iter(response_cache), None)
            if first_key:
                response_cache.pop(first_key, None)
        response_cache[cache_key_for(query)] = answer


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if req.chat_id: update_student_session(req.chat_id)
    answer = await run_rag_query(req.message.strip())
    return {"status": "success", "response": answer}

@app.post("/api/clear_context")
async def api_clear_context(req: ClearContextRequest):
    session_key = req.session_id or req.browserID
    if session_key:
        clear_user_history(session_key)
        active_stream_sessions.discard(session_key)
    return {"status": "success", "message": "Context cleared"}

@app.post("/astream")
async def astream(req: AstreamRequest):
    session_key = req.session_id or req.browserID
    if session_key in active_stream_sessions:
        active_stream_sessions.discard(session_key)

    if session_key:
        update_student_session(session_key)
        active_stream_sessions.add(session_key)

    async def event_generator():
        seq = 1
        query_text = compress_long_query(req.input)
        try:
            if not query_text:
                yield sse_payload_with_id("請輸入問題後再送出。", seq)
                return
            cached = await get_cached_response_async(query_text)
            if cached:
                yield sse_payload_with_id(cached, seq)
                return
            if rag_chain is None:
                yield sse_payload_with_id("⏳ Alex 正在準備知識庫，請稍後再試。", seq)
                return

            history_messages = normalize_chat_history(req.chat_history)
            yielded = False
            chunks = []
            context_docs = []

            async with asyncio.timeout(OLLAMA_REQUEST_TIMEOUT):
                async for chunk in rag_chain.astream({"input": query_text, "chat_history": history_messages}):
                    if isinstance(chunk, dict) and "context" in chunk:
                        context_docs = chunk["context"]

                    answer_chunk = chunk.get("answer") if isinstance(chunk, dict) else None
                    if not answer_chunk: continue
                    cleaned = sanitize_hallucinations(strip_think_tags(str(answer_chunk)))
                    if cleaned:
                        yielded = True
                        chunks.append(cleaned)
                        yield sse_payload_with_id(cleaned, seq)
                        seq += 1

            if context_docs:
                ref_footer = format_reference_footer(context_docs)
                if ref_footer:
                    chunks.append(ref_footer)
                    yield sse_payload_with_id(ref_footer, seq)
                    seq += 1

            if not yielded:
                answer = await run_rag_query(query_text, req.chat_history)
                yield sse_payload_with_id(answer, seq)
            elif chunks:
                await set_cached_response_async(query_text, "".join(chunks))
        except TimeoutError:
            yield sse_payload_with_id("⚠️ 模型回應逾時，請再試一次。", seq)
        except Exception as e:
            yield sse_payload_with_id(f"串流故障：{e}", seq)
        finally:
            if session_key:
                active_stream_sessions.discard(session_key)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )

@app.get("/api/status")
async def api_status():
    return {
        "ready": rag_chain is not None,
        "ollama_model": OLLAMA_MODEL,
        "state": rag_status.get("state"),
        "message": rag_status.get("message"),
    }

@app.get("/webapp", response_class=HTMLResponse)
async def get_web_app():
    html_content = r"""<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>PolyU Academic Advisor</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-color: var(--tg-theme-bg-color, #f4f4f7);
            --text-color: var(--tg-theme-text-color, #1a1a1a);
            --hint-color: var(--tg-theme-hint-color, #8e8e93);
            --button-color: var(--tg-theme-button-color, #007aff);
            --button-text-color: var(--tg-theme-button-text-color, #ffffff);
            --secondary-bg: var(--tg-theme-secondary-bg-color, #ffffff);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color); color: var(--text-color);
            display: flex; flex-direction: column;
            height: 100vh; height: var(--tg-viewport-height, 100dvh);
            overflow: hidden;
        }
        .header {
            padding: 12px 16px; background-color: var(--secondary-bg);
            border-bottom: 1px solid rgba(0, 0, 0, 0.08);
            display: flex; justify-content: space-between; align-items: center;
            flex-shrink: 0;
        }
        .header-info h2 { margin: 0; font-size: 15px; font-weight: 600; }
        .header-info p { margin: 2px 0 0; font-size: 12px; color: var(--hint-color); }
        .clear-context {
            border: 0; border-radius: 8px; padding: 6px 10px;
            background: var(--bg-color); color: var(--hint-color);
            font-size: 11px; cursor: pointer;
        }
        .quick-topics {
            display: flex; gap: 8px; padding: 8px 12px;
            overflow-x: auto; background-color: var(--secondary-bg);
            border-bottom: 1px solid rgba(0,0,0,0.05);
            flex-shrink: 0;
            -webkit-overflow-scrolling: touch;
        }
        .chip {
            background-color: var(--bg-color); color: var(--text-color);
            border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 16px;
            padding: 6px 12px; font-size: 12px; white-space: nowrap; cursor: pointer;
        }
        .chip.disabled {
            opacity: 0.5; pointer-events: none; cursor: not-allowed;
        }
        .chat-container {
            flex: 1; min-height: 0; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
            -webkit-overflow-scrolling: touch;
        }
        .message {
            max-width: 90%; padding: 10px 14px; border-radius: 16px;
            font-size: 14px; line-height: 1.5; word-wrap: break-word;
        }
        .message p { margin: 0 0 8px; }
        .message p:last-child { margin-bottom: 0; }
        .message.user {
            align-self: flex-end; background-color: var(--button-color);
            color: var(--button-text-color); border-bottom-right-radius: 4px;
        }
        .message.bot {
            align-self: flex-start; background-color: var(--secondary-bg);
            color: var(--text-color); border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }
        table {
            width: 100%; border-collapse: collapse; margin: 10px 0;
            font-size: 13px; text-align: left;
        }
        th, td {
            padding: 8px 10px; border: 1px solid rgba(0,0,0,0.1);
        }
        th { background-color: rgba(0,0,0,0.04); font-weight: 600; }
        a { color: #007aff; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
        .input-container {
            padding: 10px 12px; background-color: var(--secondary-bg);
            border-top: 1px solid rgba(0,0,0,0.08);
            display: flex; align-items: center; gap: 8px;
            flex-shrink: 0;
        }
        .input-container textarea {
            flex: 1; border: 1px solid rgba(0, 0, 0, 0.12);
            border-radius: 20px; padding: 10px 14px; font-size: 14px;
            background-color: var(--bg-color); color: var(--text-color);
            outline: none; resize: none; max-height: 96px;
        }
        .input-container textarea:disabled {
            background-color: rgba(0,0,0,0.05); color: var(--hint-color); cursor: not-allowed;
        }
        .input-container button {
            background-color: var(--button-color); color: var(--button-text-color);
            border: none; border-radius: 50%; width: 38px; height: 38px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; font-weight: bold; flex-shrink: 0;
            transition: opacity 0.2s, background-color 0.2s;
        }
        .input-container button:disabled { 
            opacity: 0.4; cursor: not-allowed; background-color: #8e8e93; 
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-info">
            <h2>PolyU Academic Advisor (Alex)</h2>
            <p>Industrial and Systems Engineering (ISE)</p>
        </div>
        <button class="clear-context" id="clearBtn" type="button">Clear Chat</button>
    </div>

    <div class="quick-topics">
        <div class="chip" data-query="CAR/GUR 學分要求是什麼？">📋 CAR/GUR 要求</div>
        <div class="chip" data-query="ISE 的 WIE 實習有甚麼要求與表格下載？">💼 WIE 實習與表格</div>
        <div class="chip" data-query="Capstone 畢業論文如何選題？">🎓 Capstone 選題</div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message bot">👋 Hello! 我係 Alex，PolyU ISE 學術諮詢助手。請點選上方快捷選項或直接在下方輸入問題向我查詢！</div>
    </div>

    <div class="input-container">
        <textarea id="userInput" maxlength="1500" rows="1" placeholder="輸入你的問題..."></textarea>
        <button id="sendButton" type="button" aria-label="Send message">➔</button>
    </div>

    <script>
        var tg = {};
        try {
            if (window.Telegram && window.Telegram.WebApp) {
                tg = window.Telegram.WebApp;
                tg.expand();
            }
        } catch (err) {
            console.warn("Telegram WebApp API initialization failed:", err);
        }

        function newId() {
            if (window.crypto && window.crypto.randomUUID) {
                return window.crypto.randomUUID();
            }
            return Date.now().toString() + "-" + Math.random().toString(16).substring(2);
        }

        var browserId = localStorage.getItem("browserID") || newId();
        localStorage.setItem("browserID", browserId);
        
        var chatId = String((tg.initDataUnsafe && tg.initDataUnsafe.user && tg.initDataUnsafe.user.id) || browserId);
        var conversationId = localStorage.getItem("conversationID") || newId();
        localStorage.setItem("conversationID", conversationId);
        
        var chatHistory = [];
        try {
            var stored = localStorage.getItem("chatHistory");
            chatHistory = stored ? JSON.parse(stored) : [];
            if (!Array.isArray(chatHistory)) chatHistory = [];
        } catch (e) {
            chatHistory = [];
        }
        
        var isSending = false;

        if (window.marked) {
            window.marked.setOptions({ gfm: true, breaks: true });
        }

        function scrollToBottom() {
            var container = document.getElementById("chatContainer");
            if (container) {
                requestAnimationFrame(function() {
                    container.scrollTop = container.scrollHeight;
                });
            }
        }

        function setUIState(sending) {
            isSending = sending;
            var input = document.getElementById("userInput");
            var sendButton = document.getElementById("sendButton");
            var chips = document.querySelectorAll(".chip");

            if (input) input.disabled = sending;
            if (sendButton) sendButton.disabled = sending;
            
            chips.forEach(function(chip) {
                if (sending) chip.classList.add("disabled");
                else chip.classList.remove("disabled");
            });
        }

        function appendMessage(text, sender, isHtml) {
            var container = document.getElementById("chatContainer");
            var msgDiv = document.createElement("div");
            msgDiv.className = "message " + sender;
            if (isHtml) {
                msgDiv.innerHTML = text;
            } else {
                msgDiv.textContent = text;
            }
            container.appendChild(msgDiv);
            scrollToBottom();
            return msgDiv;
        }

        async function sendMessage() {
            if (isSending) return;
            var input = document.getElementById("userInput");
            var text = input ? input.value.trim() : "";
            if (!text) return;

            setUIState(true);

            appendMessage(text, "user", false);
            if (input) {
                input.value = "";
                input.style.height = "auto";
            }

            var botMsg = appendMessage("Thinking...", "bot", false);
            var rawResponse = "";
            var isFirstChunk = true;

            try {
                var response = await fetch("/astream", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    },
                    body: JSON.stringify({
                        input: text,
                        chat_history: chatHistory.slice(-6),
                        browserID: browserId,
                        session_id: chatId,
                        conversation_id: conversationId,
                        message_id: newId()
                    })
                });

                if (!response.ok) throw new Error("HTTP " + response.status);
                if (!response.body) throw new Error("Streaming not supported");

                var reader = response.body.getReader();
                var decoder = new TextDecoder("utf-8");
                var buffer = "";

                while (true) {
                    var res = await reader.read();
                    if (res.done) break;

                    buffer += decoder.decode(res.value, { stream: true });
                    var lines = buffer.split("\n");
                    buffer = lines.pop();

                    for (var i = 0; i < lines.length; i++) {
                        var trimmedLine = lines[i].trim();
                        if (trimmedLine.indexOf("data: ") !== 0) continue;
                        
                        try {
                            var jsonStr = trimmedLine.slice(6);
                            var payload = JSON.parse(jsonStr);
                            if (payload.content) {
                                if (isFirstChunk) {
                                    botMsg.textContent = "";
                                    isFirstChunk = false;
                                }
                                rawResponse += payload.content;
                                if (window.marked) {
                                    botMsg.innerHTML = window.marked.parse(rawResponse);
                                } else {
                                    botMsg.textContent = rawResponse;
                                }
                                scrollToBottom();
                            }
                        } catch (e) {
                            console.warn("SSE JSON Parse warning:", e);
                        }
                    }
                }
            } catch (err) {
                console.error("Fetch Error:", err);
                botMsg.textContent = "⚠️ 連線錯誤，請檢查伺服器或重新嘗試。";
            } finally {
                if (rawResponse) {
                    chatHistory.push({ role: "user", content: text });
                    chatHistory.push({ role: "assistant", content: rawResponse });
                    chatHistory = chatHistory.slice(-6);
                    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
                }
                setUIState(false);
                if (input) {
                    input.focus();
                }
            }
        }

        document.addEventListener("DOMContentLoaded", function() {
            var sendBtn = document.getElementById("sendButton");
            var inputField = document.getElementById("userInput");
            var clearBtn = document.getElementById("clearBtn");

            if (sendBtn) {
                sendBtn.addEventListener("click", function(e) {
                    e.preventDefault();
                    if (!isSending) sendMessage();
                });
            }

            if (inputField) {
                inputField.addEventListener("keydown", function(e) {
                    if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        if (!isSending) sendMessage();
                    }
                });
                inputField.addEventListener("input", function() {
                    inputField.style.height = "auto";
                    inputField.style.height = Math.min(inputField.scrollHeight, 96) + "px";
                });
            }

            if (clearBtn) {
                clearBtn.addEventListener("click", async function() {
                    if (isSending) return;
                    chatHistory = [];
                    conversationId = newId();
                    localStorage.setItem("chatHistory", "[]");
                    localStorage.setItem("conversationID", conversationId);
                    try {
                        await fetch("/api/clear_context", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ browserID: browserId, session_id: chatId })
                        });
                    } catch (e) { console.warn("Context clear error:", e); }
                    appendMessage("✅ Context cleared.", "bot", false);
                });
            }

            var chips = document.querySelectorAll(".chip");
            for (var i = 0; i < chips.length; i++) {
                (function(chip) {
                    chip.addEventListener("click", function() {
                        if (isSending) return;
                        var q = chip.getAttribute("data-query");
                        if (q && inputField) {
                            inputField.value = q;
                            sendMessage();
                        }
                    });
                })(chips[i]);
            }
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.get("/")
async def get_chat_page():
    return "<h1>PolyU AI Academic Advisor Backend Active</h1><p>Visit /webapp to access Telegram Mini App interface.</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)