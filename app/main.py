"""
Production FastAPI app: RAG chain with /query (answer + source documents).
"""
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.retrieval import HybridRetriever
from src.generate import get_generator, build_prompt, generate_answer
from src.reranker import CEReRanker

ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = ROOT / "artifacts" / "index"

app = FastAPI(title="RAG Policy Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_retriever() -> HybridRetriever:
    if not (IDX_DIR / "faiss.index").exists():
        raise RuntimeError("Index not found. Run: python -m scripts.build_index")
    retr = HybridRetriever()
    retr.load(str(IDX_DIR))
    return retr


def get_generator_dep():
    return get_generator()


# --- Request/Response models ---


class QueryIn(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    max_new_tokens: int = 160
    temperature: float = 0.2
    use_reranker: bool = True
    candidates: int = 20


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class QueryOut(BaseModel):
    answer: str
    source_documents: list[SourceDocument]
    latency_ms: int


# --- Endpoints ---


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryOut)
def query(
    payload: QueryIn,
    retr: HybridRetriever = Depends(get_retriever),
    gen=Depends(get_generator_dep),
):
    """
    RAG query: returns answer and source documents used.
    Uses hybrid retrieval (FAISS + BM25, 50/50) and optional reranker.
    """
    start = time.time()
    q = (payload.query or "").strip()
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short.")

    results = retr.search(q, k=payload.top_k, alpha=payload.alpha)
    if payload.use_reranker:
        candidates = retr.search_candidates(
            q, n=payload.candidates, alpha=payload.alpha
        )
        ce = CEReRanker()
        reranked = ce.rerank(q, [(s, t) for (s, t, m) in candidates])
        out_results = []
        for score, text in reranked[: payload.top_k]:
            for s, t, m in candidates:
                if t == text:
                    out_results.append((score, t, m))
                    break
        results = out_results

    prompt = build_prompt(q, results)
    answer = generate_answer(
        gen, prompt,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )
    latency_ms = int((time.time() - start) * 1000)
    source_documents = [
        SourceDocument(content=t[:2000], metadata=m)
        for (_, t, m) in results
    ]
    return QueryOut(
        answer=answer,
        source_documents=source_documents,
        latency_ms=latency_ms,
    )


# Backward compatibility: /ask with same behavior, response includes "contexts"
class AskIn(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    max_new_tokens: int = 160
    temperature: float = 0.2
    use_reranker: bool = True
    candidates: int = 20


class AskOut(BaseModel):
    answer: str
    contexts: list[dict]
    latency_ms: int


@app.post("/ask", response_model=AskOut)
def ask(
    payload: AskIn,
    retr: HybridRetriever = Depends(get_retriever),
    gen=Depends(get_generator_dep),
):
    """Legacy /ask endpoint: same as /query but returns 'contexts' with score/text/metadata."""
    start = time.time()
    q = (payload.query or "").strip()
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short.")

    results = retr.search(q, k=payload.top_k, alpha=payload.alpha)
    if payload.use_reranker:
        candidates = retr.search_candidates(
            q, n=payload.candidates, alpha=payload.alpha
        )
        ce = CEReRanker()
        reranked = ce.rerank(q, [(s, t) for (s, t, m) in candidates])
        out_results = []
        for score, text in reranked[: payload.top_k]:
            for s, t, m in candidates:
                if t == text:
                    out_results.append((score, t, m))
                    break
        results = out_results

    prompt = build_prompt(q, results)
    answer = generate_answer(
        gen, prompt,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )
    latency_ms = int((time.time() - start) * 1000)
    ctxs = [
        {"score": float(s), "text": t[:500], "metadata": m}
        for (s, t, m) in results
    ]
    return AskOut(answer=answer, contexts=ctxs, latency_ms=latency_ms)
