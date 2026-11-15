# -*- coding: utf-8 -*-
"""Nhập đầu ra Day 6 vào Day 7 (chỉ đọc repo Day 6).

1. Copy vector store Chroma Day 6 (results/chroma_db) -> data/day6_store/chroma_db.
   Day 7 mở bản sao để không ghi vào repo Day 6 (PersistentClient có thể ghi sqlite).
2. Xuất collection B (theo cấu trúc, 178 chunk) sang data/chunks.json theo định dạng Day 7:
   id, text, source_id, source, page, metadata. Vector KHÔNG bị tạo lại: dense retrieval dùng
   chính vector đã lưu trong Chroma Day 6.
3. Ghi data/day6_import.json: sha256 các file, số chunk, model embedding, đường dẫn gốc.

Chạy: python scripts/import_day6.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day07.config import CHROMA_DIR, DATA, DAY6_COLLECTION, DAY6_DIR  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    src = DAY6_DIR / "results" / "chroma_db"
    if not src.exists():
        raise SystemExit(f"Không thấy vector store Day 6 tại {src}")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    shutil.copytree(src, CHROMA_DIR)

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(DAY6_COLLECTION)
    got = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        chunks.append({"id": cid, "text": doc, "source_id": meta["document_id"], "source": meta["source"],
                       "page": int(meta["page"]), "metadata": meta})
    # thứ tự đọc tự nhiên: văn bản -> trang -> vị trí ký tự (dùng cho mở rộng chunk lân cận)
    chunks.sort(key=lambda c: (c["source_id"], c["page"], c["metadata"].get("start_char", 0)))
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    run_meta = json.loads((DAY6_DIR / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    info = {
        "day6_dir": str(DAY6_DIR.name),
        "collection": DAY6_COLLECTION,
        "n_chunks": len(chunks),
        "documents": sorted({c["source_id"] for c in chunks}),
        "embedding": run_meta["embedding"],
        "vector_store": run_meta["vector_store"],
        "day6_run_at_utc": run_meta["run_at_utc"],
        "chroma_sqlite_sha256": sha256(src / "chroma.sqlite3"),
        "chunks_json_sha256": sha256(DATA / "chunks.json"),
    }
    (DATA / "day6_import.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("collection", "n_chunks", "documents")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
