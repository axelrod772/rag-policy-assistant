from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import time
import python_multipart # Suppress Starlette deprecation warning

from src.retrieval import HybridRetriever
from src.generate import get_generator, build_prompt

ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = ROOT / "artifacts" / "index"

app = FastAPI(title="RAG Policy Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_retriever():
    if not (IDX_DIR / "faiss.index").exists():
        raise RuntimeError("Index not found. Run: python -m scripts.build_index")
    retr = HybridRetriever()
    retr.load(str(IDX_DIR))
    return retr

def get_generator_dep():
    return get_generator("google/flan-t5-small")

class AskIn(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.6
    max_new_tokens: int = 160
    temperature: float = 0.2

class AskOut(BaseModel):
    answer: str
    contexts: List[dict]
    latency_ms: int

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi import Depends

@app.post("/ask", response_model=AskOut)
def ask(
    payload: AskIn,
    retr: HybridRetriever = Depends(get_retriever),
    gen=Depends(get_generator_dep)
):
    start = time.time()
    q = (payload.query or "").strip()
    if len(q) < 5:
        raise HTTPException(status_code=400, detail="Query too short.")
    results = retr.search(q, k=payload.top_k, alpha=payload.alpha)
    prompt = build_prompt(q, results)
    do_sample = payload.temperature > 0.0
    out = gen(prompt, max_new_tokens=payload.max_new_tokens, temperature=payload.temperature, clean_up_tokenization_spaces=True, do_sample=do_sample)
    answer = out[0]["generated_text"]
    latency = int((time.time() - start) * 1000)
    ctxs = [{"score": float(s), "text": t[:500], "metadata": m} for (s, t, m) in results]
    return {"answer": answer, "contexts": ctxs, "latency_ms": latency}
