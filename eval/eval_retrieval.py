from pathlib import Path
import json
from typing import List, Dict
from src.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = ROOT / "artifacts" / "index"
QA = ROOT / "notebooks" / "qa_seed.jsonl"

def load_qa() -> List[Dict]:
    lines = QA.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]

def keyword_hit(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)

def main():
    retr = HybridRetriever()
    retr.load(str(IDX_DIR))
    qa = load_qa()
    for alpha in [0.0, 0.6, 1.0]:
        hits_at_3 = 0
        for ex in qa:
            res = retr.search(ex["question"], k=3, alpha=alpha)
            hit = any(keyword_hit(r[1], ex["keywords"]) for r in res)
            hits_at_3 += 1 if hit else 0
        recall = hits_at_3 / max(1, len(qa))
        print(f"alpha={alpha:.1f} Recall@3={recall:.2f} over {len(qa)} queries")

if __name__ == "__main__":
    main()
