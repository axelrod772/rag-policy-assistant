import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Ensure FAISS index exists before tests
import pytest
from pathlib import Path
import subprocess

@pytest.fixture(scope="session", autouse=True)
def ensure_faiss_index():
	root = Path(__file__).resolve().parent.parent
	idx_dir = root / "artifacts" / "index"
	faiss_index = idx_dir / "faiss.index"
	meta_json = idx_dir / "meta.json"
	if not (faiss_index.exists() and meta_json.exists()):
		subprocess.run(["python", "scripts/build_index.py"], check=True)