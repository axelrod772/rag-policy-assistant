"""
Ingestion with RecursiveCharacterTextSplitter.
Chunk size 1000, overlap 200 to maintain context across segments.
"""
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults: chunk 1000, overlap 200 (Senior-level chunking)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_text(t: str) -> str:
    """Remove excessive whitespace and strip."""
    return re.sub(r"\s+", " ", (t or "")).strip()


def read_pdf(path: Path) -> List[Dict]:
    """Extract text from each page of a PDF file and return as list of dicts."""
    reader = PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        chunks.append({
            "text": text,
            "metadata": {"source": str(path), "page": i + 1},
        })
    return chunks


def read_txt(path: Path) -> List[Dict]:
    """Read and clean text from a TXT or MD file, returned as a single chunk."""
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [{"text": text, "metadata": {"source": str(path), "page": 1}}]


def get_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Return RecursiveCharacterTextSplitter for advanced chunking."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def ingest_paths(
    paths: List[Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Ingest PDF, TXT, and MD files using RecursiveCharacterTextSplitter.
    Returns list of {"text": str, "metadata": dict}.
    """
    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Dict] = []

    for p in paths:
        if p.suffix.lower() == ".pdf":
            pages = read_pdf(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            pages = read_txt(p)
        else:
            continue

        for page in pages:
            parts = splitter.split_text(page["text"])
            for idx, part in enumerate(parts):
                if len(part) < 50:
                    continue
                meta = dict(page["metadata"])
                meta["chunk_id"] = idx
                all_chunks.append({"text": part, "metadata": meta})

    return all_chunks
