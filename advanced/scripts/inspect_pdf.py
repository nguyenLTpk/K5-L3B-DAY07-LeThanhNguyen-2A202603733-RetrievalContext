# -*- coding: utf-8 -*-
"""Thống kê từng trang PDF: số ký tự text, số ảnh nhúng, số đối tượng vector (gợi ý biểu đồ/bảng),
các nhãn Hình/Bảng/Biểu đồ/Hộp xuất hiện trong text. Ghi advanced/data/page_stats.json.

Chạy: python -I advanced/scripts/inspect_pdf.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "ccdr_vietnam_2022_vi.pdf"

LABEL = re.compile(r"(Hình|Bảng|Biểu đồ|Hộp|Bản đồ)\s+([A-Z]?\d+(?:\.\d+)?[a-z]?)\s*[:.]?\s*([^\n]{0,120})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    doc = fitz.open(PDF)
    stats = []
    for i, page in enumerate(doc):
        text = page.get_text()
        labels = [f"{m.group(1)} {m.group(2)}: {m.group(3).strip()}" for m in LABEL.finditer(text)]
        stats.append({"page": i + 1, "chars": len(text.strip()), "images": len(page.get_images(full=True)),
                      "drawings": len(page.get_drawings()), "labels": labels[:8],
                      "head": " ".join(text.split())[:160]})
    (ROOT / "data" / "page_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pages", doc.page_count, "size", doc[0].rect)
    for s in stats:
        print(f"p{s['page']:03d} chars={s['chars']:5d} img={s['images']:2d} draw={s['drawings']:4d} | "
              f"{'; '.join(s['labels'])[:150]} || {s['head'][:90]}")


if __name__ == "__main__":
    main()
