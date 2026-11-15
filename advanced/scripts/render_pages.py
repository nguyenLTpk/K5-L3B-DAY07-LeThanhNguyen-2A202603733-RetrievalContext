# -*- coding: utf-8 -*-
"""Render mọi trang PDF thành PNG (advanced/data/pages/p001.png ...) và xuất text từng trang.

- PNG 144 DPI dùng cho vision model, crop vùng hình/bảng và kiểm chứng thủ công.
- Text từng trang (PyMuPDF, thứ tự block đọc) -> advanced/data/page_text.json.
Chạy: python -I advanced/scripts/render_pages.py [--dpi 144]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "ccdr_vietnam_2022_vi.pdf"
PAGES = ROOT / "data" / "pages"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=144)
    args = ap.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    texts = []
    for i, page in enumerate(doc):
        pno = i + 1
        page.get_pixmap(dpi=args.dpi).save(PAGES / f"p{pno:03d}.png")
        blocks = page.get_text("blocks", sort=True)
        texts.append({"page": pno, "printed_page": None,
                      "blocks": [{"bbox": [round(v, 1) for v in b[:4]], "text": b[4].strip()} for b in blocks if b[4].strip() and b[6] == 0]})
    (ROOT / "data" / "page_text.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8")
    print("rendered", doc.page_count, "pages at", args.dpi, "dpi")


if __name__ == "__main__":
    main()
