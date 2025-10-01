from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader
import re

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
            "metadata": {
                "source": str(path), 
                "page": i + 1
            }
        })
    return chunks

def read_txt(path: Path) -> List[Dict]:
    """Read and clean text from a TXT or MD file, returned as a single chunk."""
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [{
        "text": text, 
        "metadata": {
            "source": str(path), 
            "page": 1
        }
    }]

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Chunk the input text with specified size and overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += step
    return chunks

def ingest_paths(paths: List[Path], chunk_size=900, overlap=150) -> List[Dict]:
    """Ingest PDF, TXT, and MD files, split text into chunks, and add metadata."""
    all_chunks: List[Dict] = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            pages = read_pdf(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            pages = read_txt(p)
        else:
            continue
        for page in pages:
            parts = chunk_text(page["text"], chunk_size, overlap)
            for idx, part in enumerate(parts):
                if len(part) < 50:
                    continue
                meta = dict(page["metadata"])
                meta["chunk_id"] = idx
                all_chunks.append({
                    "text": part, 
                    "metadata": meta
                })
    return all_chunks
