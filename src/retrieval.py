from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
import os
import json

class HybridRetriever:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
                {"texts": self.texts, "metas": self.metas, "model": self.model_name}, f
            )

    def load(self, path: str):
        self.faiss_index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts, self.metas = data["texts"], data["metas"]
        self.embed = SentenceTransformer(data["model"])
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self, query: str, k: int = 5, alpha: float = 0.6
    ) -> List[Tuple[float, str, Dict]]:
        qv = self.embed.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        dense_scores, dense_idx = self.faiss_index.search(qv, min(k * 4, len(self.texts)))
        dense_scores, dense_idx = dense_scores[0], dense_idx[0]
        if dense_scores.size:
            dmin, dmax = float(dense_scores.min()), float(dense_scores.max())
            if dmax - dmin > 1e-9:
                dense_scores = (dense_scores - dmin) / (dmax - dmin)
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_idx = np.argsort(-bm25_scores)[:min(k * 4, len(self.texts))]
        bm25_top_scores = bm25_scores[bm25_idx]
        if bm25_top_scores.size:
            bmin, bmax = float(bm25_top_scores.min()), float(bm25_top_scores.max())
            if bmax - bmin > 1e-9:
                bm25_top_scores = (bm25_top_scores - bmin) / (bmax - bmin)
        scores = {}
        for s, i in zip(dense_scores, dense_idx):
            scores[int(i)] = max(scores.get(int(i), 0.0), alpha * float(s))
        for s, i in zip(bm25_top_scores, bm25_idx):
            scores[int(i)] = scores.get(int(i), 0.0) + (1 - alpha) * float(s)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(sc, self.texts[i], self.metas[i]) for i, sc in ranked]
