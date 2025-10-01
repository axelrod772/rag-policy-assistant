from pathlib import Path
from src.retrieval import HybridRetriever
from src.ingest import ingest_paths

def test_build_and_search():
    root = Path(__file__).resolve().parent.parent
    data = root / "data" / "policy_sample.txt"
    chunks = ingest_paths([data], chunk_size=200, overlap=50)
    retr = HybridRetriever()
    retr.build(chunks)
    out = retr.search("Who regulates insurance in India?", k=3, alpha=0.6)
    assert len(out) >= 1
    assert isinstance(out[0][1], str)
