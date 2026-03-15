"""
Backward-compatible app entry: uvicorn src.app:app.
For production use: uvicorn app.main:app
"""
from app.main import app

__all__ = ["app"]
