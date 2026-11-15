# -*- coding: utf-8 -*-
"""Kiểm tra benchmark nâng cao (không thay việc đối chiếu bằng mắt với ảnh trang gốc).

- Đếm câu theo split/nhóm; kiểm tra đủ ngưỡng ADVANCED.md (test >= 30, >= 24 có đáp án, >= 6 thiếu
  evidence; bảng 6, ảnh 5, biểu đồ 5, multi-hop 4, text 4; dev >= 15 + 3).
- Với evidence dạng trích dẫn văn bản, kiểm tra chuỗi có trong text layer của đúng trang
  (bỏ '...' ở giữa). Với bảng/biểu đồ/ảnh: báo các số trong 'content' có/không có trong text layer
  (nếu không có -> nội dung chỉ nằm trong ảnh/raster, phải kiểm bằng mắt — đã làm khi soạn).
- Liệt kê trang evidence dùng chung giữa dev và test.
- Ghi sha256 vào advanced/benchmark/benchmark_lock.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
from day07.textnorm import norm  # noqa: E402

MIN_TEST = {"table": 6, "image": 5, "chart": 5, "multi_hop": 4, "text": 4}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "benchmark" / "benchmark.json"
    qs = json.loads(path.read_text(encoding="utf-8"))
    pages = {p["page"]: " ".join(b["text"] for b in p["blocks"])
             for p in json.loads((ROOT / "data" / "page_text.json").read_text(encoding="utf-8"))}
    report = {"counts": {}, "quote_checks": [], "problems": []}
    for split in ("dev", "test"):
        c = Counter(q["group"] for q in qs if q["split"] == split)
        report["counts"][split] = dict(c) | {"total": sum(c.values()), "answerable": sum(v for k, v in c.items() if k != "unanswerable")}
    t = report["counts"]["test"]
    if t["total"] < 30 or t["answerable"] < 24 or t.get("unanswerable", 0) < 6:
        report["problems"].append("test chưa đủ số câu")
    for g, m in MIN_TEST.items():
        if t.get(g, 0) < m:
            report["problems"].append(f"test nhóm {g} < {m}")
    d = report["counts"]["dev"]
    if d["answerable"] < 15 or d.get("unanswerable", 0) < 3:
        report["problems"].append("dev chưa đủ số câu")

    ev_pages = {"dev": set(), "test": set()}
    for q in qs:
        if not q["answerable"]:
            continue
        for gi, grp in enumerate(q["evidence"]):
            for alt in grp["alternatives"]:
                ev_pages[q["split"]].add(alt["page"])
                text = norm(pages.get(alt["page"], ""))
                content = alt["content"]
                parts = [p for p in re.split(r"\s*\.\.\.\s*|\s*\|\s*", content) if len(p.strip()) >= 3]
                found = {p: norm(p) in text for p in parts}
                # trích dẫn nguyên văn: evidence là đoạn văn / hộp; bảng-hình: giá trị hoặc mô tả vùng
                kind = "quote" if alt["element"].startswith(("đoạn", "Hộp")) else "values"
                report["quote_checks"].append({"id": q["id"], "group": gi, "page": alt["page"], "element": alt["element"],
                                               "kind": kind, "found_in_text_layer": found})
                if kind == "quote" and not all(found.values()):
                    report["problems"].append(f"{q['id']}: trích dẫn không thấy trong text layer trang {alt['page']}: "
                                              f"{[p for p, ok in found.items() if not ok]}")
    report["pages_shared_dev_test"] = sorted(ev_pages["dev"] & ev_pages["test"])
    report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "benchmark" / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "problems", "pages_shared_dev_test", "sha256")}, ensure_ascii=False, indent=1))
    for c in report["quote_checks"]:
        if c["kind"] == "values":
            miss = [p for p, ok in c["found_in_text_layer"].items() if not ok]
            if miss:
                print(f"  {c['id']} p{c['page']} {c['element']}: chỉ có trong ảnh/raster hoặc khác định dạng -> {miss}")


if __name__ == "__main__":
    main()
