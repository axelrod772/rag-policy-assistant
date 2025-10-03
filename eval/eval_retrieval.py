from pathlib import Path
import json
from typing import List, Dict, Tuple
from src.retrieval import HybridRetriever
from src.reranker import CEReRanker

ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = ROOT / "artifacts" / "index"
QA = ROOT / "notebooks" / "qa_seed.jsonl"

def load_qa() -> List[Dict]:
    lines = QA.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]

def keyword_hit(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)

def recall_at_k(retr, qa, alpha=0.6, k=3, use_reranker=False, pool=20):
    hits = 0
    reranker = CEReRanker() if use_reranker else None
    for ex in qa:
        if use_reranker:
            cands = retr.search(ex["question"], k=pool, alpha=alpha)
            pairs = [(s, t) for (s, t, m) in cands]
            reranked = reranker.rerank(ex["question"], pairs)
            top = reranked[:k]
            texts = [t for _, t in top]
            hit = any(keyword_hit(t, ex["keywords"]) for t in texts)
        else:
            res = retr.search(ex["question"], k=k, alpha=alpha)
            hit = any(keyword_hit(r[1], ex["keywords"]) for r in res)
        hits += 1 if hit else 0
    return hits / max(1, len(qa))

def main():
    retr = HybridRetriever()
    retr.load(str(IDX_DIR))
    qa = load_qa()
    for alpha in [0.0, 0.6, 1.0]:
        base = recall_at_k(retr, qa, alpha=alpha, k=3, use_reranker=False)
        rer  = recall_at_k(retr, qa, alpha=alpha, k=3, use_reranker=True, pool=20)
        print(f"alpha={alpha:.1f} Recall@3 base={base:.2f} reranked={rer:.2f}")
    print("Done.")

if __name__ == "__main__":
    main()
