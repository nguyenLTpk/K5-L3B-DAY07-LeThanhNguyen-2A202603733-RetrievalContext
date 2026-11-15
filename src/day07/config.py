# -*- coding: utf-8 -*-
"""Đường dẫn, model và đọc .env (không in secret)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Repo Day 6 (chỉ đọc). Có thể ghi đè bằng biến môi trường DAY6_DIR.
DAY6_DIR = Path(os.environ.get(
    "DAY6_DIR", ROOT.parent / "K5-L3B-DAY06-LeThanhNguyen-2A202603733-DataIngestion"))
E5_MODEL_DIR = Path(os.environ.get("E5_MODEL_DIR", DAY6_DIR / "models" / "multilingual-e5-small"))

# Bản sao chỉ-đọc của vector store Day 6 (copy từ DAY6_DIR/results/chroma_db, xem scripts/import_day6.py)
CHROMA_DIR = DATA / "day6_store" / "chroma_db"
DAY6_COLLECTION = "day06_b_e5small"   # chiến lược B (theo cấu trúc) đã chọn ở Day 6

LLM_MODEL = os.environ.get("DAY07_LLM_MODEL", "gemini-3.5-flash")


def load_env(path: Path = ROOT / ".env") -> None:
    """Nạp KEY=VALUE từ .env vào os.environ nếu chưa có. Không log giá trị."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def gemini_key() -> str:
    load_env()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Thiếu GEMINI_API_KEY (đặt trong .env, file này đã được .gitignore)")
    return key
