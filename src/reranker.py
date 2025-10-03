from typing import List, Tuple
from sentence_transformers import CrossEncoder

class CEReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        pairs = [(query, c[1]) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(scores, [c[1] for c in candidates]), key=lambda x: x[0], reverse=True)
        # Repackage as (score, text)
        return [(float(s), t) for s, t in ranked]
