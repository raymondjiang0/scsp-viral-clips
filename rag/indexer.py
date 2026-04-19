"""
Indexes brand guide PDF + SCSP website into ChromaDB for RAG-augmented clip scoring.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from config import BRAND_GUIDE_PATH, RAG_DB_DIR, SCSP_WEBSITE_URL

_COLLECTION_NAME = "scsp_brand"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 600  # chars
_CHUNK_OVERLAP = 100


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(RAG_DB_DIR))
    return client.get_or_create_collection(_COLLECTION_NAME)


def is_indexed() -> bool:
    col = get_collection()
    return col.count() > 0


def index_all(status_callback=None) -> int:
    col = get_collection()
    embedder = SentenceTransformer(_EMBED_MODEL)
    total = 0

    if status_callback:
        status_callback("Indexing brand guide PDF…")
    total += _index_pdf(col, embedder)

    if status_callback:
        status_callback("Scraping SCSP website…")
    total += _index_website(col, embedder)

    if status_callback:
        status_callback(f"Indexed {total} chunks into RAG.")

    return total


def _index_pdf(col: chromadb.Collection, embedder: SentenceTransformer) -> int:
    if not BRAND_GUIDE_PATH.exists():
        return 0

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return 0

    doc = fitz.open(str(BRAND_GUIDE_PATH))
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    chunks = _chunk_text(full_text, source="brand_guide")
    return _upsert_chunks(col, embedder, chunks)


def _index_website(col: chromadb.Collection, embedder: SentenceTransformer) -> int:
    pages = [
        SCSP_WEBSITE_URL,
        f"{SCSP_WEBSITE_URL}/about",
        f"{SCSP_WEBSITE_URL}/reports",
        f"{SCSP_WEBSITE_URL}/events",
    ]
    all_chunks = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in pages:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            all_chunks.extend(_chunk_text(text, source=url))
        except Exception:
            continue

    return _upsert_chunks(col, embedder, all_chunks)


def add_custom_content(text: str, label: str = "custom") -> int:
    col = get_collection()
    embedder = SentenceTransformer(_EMBED_MODEL)
    chunks = _chunk_text(text, source=label)
    return _upsert_chunks(col, embedder, chunks)


def add_performing_clip(clip_description: str, why_worked: str) -> None:
    col = get_collection()
    embedder = SentenceTransformer(_EMBED_MODEL)
    text = f"HIGH PERFORMING CLIP: {clip_description}\nWHY IT WORKED: {why_worked}"
    chunks = _chunk_text(text, source="feedback")
    _upsert_chunks(col, embedder, chunks)


def _chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            doc_id = hashlib.md5(f"{source}:{start}:{chunk[:50]}".encode()).hexdigest()
            chunks.append({"id": doc_id, "text": chunk, "source": source})
        start = end - _CHUNK_OVERLAP
    return chunks


def _upsert_chunks(col: chromadb.Collection, embedder: SentenceTransformer, chunks: list[dict]) -> int:
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return len(chunks)
