"""
Hybrid retrieval: LangChain EnsembleRetriever with FAISS (semantic) + BM25 (lexical).
50/50 weight to capture both meaning and specific keywords in policy documents.
"""
from typing import List, Dict, Tuple
import os
import json

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss

# Default 50/50 weight: semantic + keyword
FAISS_WEIGHT = 0.5
BM25_WEIGHT = 0.5

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _BM25RetrieverWrapper(BaseRetriever):
    """Wrap rank_bm25 in a LangChain BaseRetriever for EnsembleRetriever."""

    texts: List[str]
    metas: List[Dict]
    bm25: BM25Okapi
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_idx = np.argsort(-scores)[: self.k]
        out = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            out.append(
                Document(
                    page_content=self.texts[i],
                    metadata=self.metas[i],
                )
            )
        return out


class HybridRetriever:
    """
    Hybrid retriever: FAISS (semantic) + BM25 (keyword) with configurable weights.
    Can build/save/load index and search returning (score, text, metadata).
    """

    def __init__(self, embed_model_name: str = EMBED_MODEL):
        self.model_name = embed_model_name
        self.embed = SentenceTransformer(embed_model_name)
        self.faiss_index = None
        self.bm25 = None
        self.texts: List[str] = []
        self.metas: List[Dict] = []

    def build(self, chunks: List[Dict]):
        self.texts = [c["text"] for c in chunks]
        self.metas = [c["metadata"] for c in chunks]
        if not self.texts:
            raise ValueError("No texts to index.")

        embs = self.embed.encode(
            self.texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        d = embs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embs)

        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"texts": self.texts, "metas": self.metas, "model": self.model_name},
                f,
            )

    def load(self, path: str):
        self.faiss_index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metas = data["metas"]
        self.embed = SentenceTransformer(data["model"])
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def get_ensemble_retriever(
        self,
        k: int = 10,
        faiss_weight: float = FAISS_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ) -> BaseRetriever:
        """
        Return a LangChain-compatible retriever that uses 50/50 hybrid (FAISS + BM25).
        """
        alpha = faiss_weight  # FAISS weight
        parent = self

        class _HybridRetrieverWrapper(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun | None = None,
            ) -> List[Document]:
                hits = parent.search(query, k=k, alpha=alpha)
                return [
                    Document(page_content=text, metadata=meta)
                    for _, text, meta in hits
                ]

        return _HybridRetrieverWrapper()

    def search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
    ) -> List[Tuple[float, str, Dict]]:
        """
        Hybrid search with alpha = weight for FAISS (1-alpha = BM25).
        Default alpha=0.5 for 50/50. Returns [(score, text, metadata), ...].
        """
        qv = self.embed.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        n = min(k * 4, len(self.texts))
        dense_scores, dense_idx = self.faiss_index.search(qv, n)
        dense_scores, dense_idx = dense_scores[0], dense_idx[0]

        if dense_scores.size:
            dmin, dmax = float(dense_scores.min()), float(dense_scores.max())
            if dmax - dmin > 1e-9:
                dense_scores = (dense_scores - dmin) / (dmax - dmin)

        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_idx = np.argsort(-bm25_scores)[:n]
        bm25_top_scores = bm25_scores[bm25_idx]
        if bm25_top_scores.size:
            bmin, bmax = float(bm25_top_scores.min()), float(bm25_top_scores.max())
            if bmax - bmin > 1e-9:
                bm25_top_scores = (bm25_top_scores - bmin) / (bmax - bmin)

        scores: Dict[int, float] = {}
        for s, i in zip(dense_scores, dense_idx):
            scores[int(i)] = scores.get(int(i), 0.0) + alpha * float(s)
        for s, i in zip(bm25_top_scores, bm25_idx):
            scores[int(i)] = scores.get(int(i), 0.0) + (1 - alpha) * float(s)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(sc, self.texts[i], self.metas[i]) for i, sc in ranked]

    def search_candidates(self, query: str, n: int = 20, alpha: float = 0.5):
        """Return top-n (score, text, meta) for reranking."""
        return self.search(query, k=n, alpha=alpha)
