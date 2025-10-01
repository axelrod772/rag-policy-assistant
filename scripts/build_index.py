from pathlib import Path
from src.ingest import ingest_paths
from src.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IDX_DIR = ROOT / "artifacts" / "index"

def main():
    paths = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in [".pdf", ".txt", ".md"]])
    if not paths:
        print("No files in data/. Add PDFs or .txt and rerun.")
        return
    chunks = ingest_paths(paths)
    print(f"Ingested chunks: {len(chunks)}")
    retr = HybridRetriever()
    retr.build(chunks)
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    retr.save(str(IDX_DIR))
    print(f"Saved index to {IDX_DIR}")

if __name__ == "__main__":
    main()
