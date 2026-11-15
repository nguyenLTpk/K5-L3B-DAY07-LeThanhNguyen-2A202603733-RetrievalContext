# -*- coding: utf-8 -*-
"""Index toàn bộ 104 trang báo cáo CCDR: chunk text + chunk trực quan -> e5-small -> Chroma.

1. Text: block text từng trang (PyMuPDF), bỏ header/footer lặp ("Việt Nam: Báo cáo về Khí hậu và Phát triển",
   số trang in), gộp block liên tiếp thành chunk <= 900 ký tự trong cùng trang.
   ID: ccdr:pNNN:tK. metadata: page (trang PDF), printed_page (= page - 10 với trang nội dung), kind="text".
2. Trực quan: mỗi phần tử informative trong advanced/data/visual/pNNN.json -> 1 chunk
   "[Hình 12 — chart] tiêu đề / Mô tả / Nội dung trích xuất / Đơn vị / Ghi chú".
   ID: ccdr:pNNN:vK. metadata: kind (table/chart/map/diagram/...), label, box_2d (vùng trên trang), crop.
   Phần tử informative=false (ảnh trang trí) chỉ ghi vào thống kê, không index.
3. Embedding: intfloat/multilingual-e5-small (giống Day 6), 'passage: ' + 512 token đầu (chunk dài bị cắt ở phần
   embedding; BM25 và nội dung gửi model vẫn dùng toàn văn). Chroma cosine, collection day07_ccdr_e5small.
Ghi: advanced/data/chunks.json, advanced/data/index_stats.json, advanced/data/store/.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

from day07.embedder import get_embedder  # noqa: E402

DATA = ROOT / "data"
STORE = DATA / "store"
COLLECTION = "day07_ccdr_e5small"
SOURCE = "data/raw/ccdr_vietnam_2022_vi.pdf"
HEADER = re.compile(r"^(\d+\s*)?\|?\s*Việt Nam: Báo cáo về Khí hậu và Phát triển\s*\|?\s*(\d+)?$")
MAX_CHARS = 900


def printed(page: int) -> int | None:
    return page - 10 if page >= 11 else None


def text_chunks() -> list[dict]:
    pages = json.loads((DATA / "page_text.json").read_text(encoding="utf-8"))
    out = []
    for p in pages:
        pno = p["page"]
        blocks = []
        for b in p["blocks"]:
            t = " ".join(b["text"].split())
            if not t or HEADER.match(t) or re.fullmatch(r"\d{1,3}", t):
                continue
            blocks.append(t)
        cur, k = "", 0
        for t in blocks:
            if cur and len(cur) + 1 + len(t) > MAX_CHARS:
                out.append((pno, k, cur))
                k += 1
                cur = ""
            if len(t) > MAX_CHARS and not cur:  # block quá dài: tách theo câu
                sents = re.split(r"(?<=[.;:])\s+", t)
                for s in sents:
                    if cur and len(cur) + 1 + len(s) > MAX_CHARS:
                        out.append((pno, k, cur))
                        k += 1
                        cur = ""
                    cur = f"{cur} {s}".strip()
                continue
            cur = f"{cur}\n{t}".strip()
        if cur:
            out.append((pno, k, cur))
    return [{"id": f"ccdr:p{pno:03d}:t{k}", "text": txt, "source_id": "ccdr-vn-2022", "source": SOURCE, "page": pno,
             "metadata": {"page": pno, "printed_page": printed(pno), "kind": "text", "label": "", "location": f"p{pno}"}}
            for pno, k, txt in out]


def visual_chunks() -> tuple[list[dict], dict]:
    out, stats = [], Counter()
    for f in sorted((DATA / "visual").glob("p*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        pno = rec["page"]
        for i, e in enumerate(rec["elements"]):
            stats[f"{e.get('kind')}_{'info' if e.get('informative') else 'decor'}"] += 1
            if not e.get("informative"):
                continue
            label = (e.get("label") or "").strip()
            head = f"[{label + ' — ' if label else ''}{e.get('kind')}] {e.get('title', '').strip()} (trang in {printed(pno)})"
            body = [head, f"Mô tả: {e.get('description', '').strip()}"]
            if e.get("transcription"):
                body.append(f"Nội dung trích xuất: {e['transcription'].strip()}")
            if e.get("units"):
                body.append(f"Đơn vị: {e['units']}")
            if e.get("notes"):
                body.append(f"Ghi chú: {e['notes']}")
            out.append({"id": f"ccdr:p{pno:03d}:v{i}", "text": "\n".join(body), "source_id": "ccdr-vn-2022", "source": SOURCE,
                        "page": pno, "metadata": {"page": pno, "printed_page": printed(pno), "kind": e.get("kind"),
                                                  "label": label, "title": e.get("title", ""), "box_2d": json.dumps(e.get("box_2d")),
                                                  "crop": e.get("crop") or "", "location": f"p{pno}", "extracted_by": rec.get("model")}})
    return out, dict(stats)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    tchunks = text_chunks()
    vchunks, vstats = visual_chunks()
    chunks = sorted(tchunks + vchunks, key=lambda c: (c["page"], c["id"].split(":")[2][0] != "t", c["id"]))
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    emb = get_embedder()
    texts = [c["text"] for c in chunks]
    ntok = [emb.n_tokens(t) for t in texts]
    vecs = emb.embed_documents(texts)
    if STORE.exists():
        shutil.rmtree(STORE)
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(STORE), settings=Settings(anonymized_telemetry=False))
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)
    for i in range(0, len(chunks), 128):
        part = chunks[i:i + 128]
        col.add(ids=[c["id"] for c in part], embeddings=vecs[i:i + 128], documents=[c["text"] for c in part],
                metadatas=[{k: (v if v is not None else -1) for k, v in c["metadata"].items()} for c in part])
    pages_with_text = {c["page"] for c in tchunks}
    pages_with_visual = {c["page"] for c in vchunks}
    stats = {"pages": 104, "text_chunks": len(tchunks), "visual_chunks": len(vchunks), "total_chunks": len(chunks),
             "pages_with_text_chunks": len(pages_with_text), "pages_with_visual_chunks": len(pages_with_visual),
             "pages_without_any_chunk": sorted(set(range(1, 105)) - pages_with_text - pages_with_visual),
             "visual_elements_by_kind": vstats, "embedding_model": emb.model_name, "dimensions": emb.dimensions,
             "chunks_over_512_tokens_truncated_for_embedding": sum(n >= 512 for n in ntok),
             "max_tokens": max(ntok), "collection": COLLECTION, "build_s": round(time.perf_counter() - t0, 1)}
    (DATA / "index_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
