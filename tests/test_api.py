from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_ask_short_query():
    r = client.post("/ask", json={"query": "hi"})
    assert r.status_code == 400

def test_ask_ok():
    r = client.post(
        "/ask",
        json={"query": "Who regulates insurance in India?", "top_k": 3, "alpha": 0.6},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "contexts" in data
