# -*- coding: utf-8 -*-
"""Tạo benchmark Day 7 = 20 câu Day 6 (đổi schema) + câu bổ sung Day 7, rồi kiểm chứng.

Schema câu có đáp án: id, split, type, answerable=true, question, expected_answer,
  gold_groups: [[{source_id, page, article, evidence}, ...(thay thế)], ...(nhóm bắt buộc)],
  gold: danh sách phẳng mọi evidence (tương thích starter), filter, origin.
Câu thiếu evidence: answerable=false, expected_behavior, absence_reason.

Kiểm chứng (ghi ra data/benchmark_verification.json):
- mỗi evidence nằm trọn trong >= 1 chunk của index (đúng source_id + page) sau chuẩn hóa;
- với câu thiếu evidence: in các từ khóa kiểm tra và số chunk chứa chúng (đọc thủ công);
- dev/test không trùng Điều (với câu có đáp án);
- sha256 benchmark để chốt trước khi chạy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day07.config import DATA, DAY6_DIR  # noqa: E402
from day07.textnorm import contains, norm  # noqa: E402

ABSENCE_KEYWORDS = {
    "d23": ["học bổng khuyến khích", "đồng", "triệu"],
    "d24": ["nghỉ hè", "tuần"],
    "d25": ["trọng số", "thi kết thúc"],
    "d26": ["vay", "tín dụng"],
    "t29": ["khiếu nại", "thời hạn khiếu nại", "ngày kể từ"],
    "t30": ["giao thông", "trừ điểm"],
    "t31": ["lệ phí", "chuyển trường"],
    "t32": ["hiến máu", "cộng điểm"],
}


def convert_day6(q: dict) -> dict:
    g = {"source_id": q["gold_source_id"], "page": q["gold_page"], "article": q.get("gold_article"),
         "evidence": q["evidence_quote"]}
    return {"id": q["id"], "split": q["split"], "type": q["type"], "answerable": True,
            "question": q["question"], "expected_answer": q["expected_answer"],
            "gold_groups": [[g]], "filter": q.get("filter"), "origin": "day6"}


def main() -> None:
    day6 = json.loads((DAY6_DIR / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    extra = json.loads((DATA / "benchmark_day7_additions.json").read_text(encoding="utf-8"))
    questions = [convert_day6(q) for q in day6]
    for q in extra:
        q = dict(q)
        q["origin"] = "day7"
        questions.append(q)
    for q in questions:
        if q["answerable"]:
            q["gold"] = [g for grp in q["gold_groups"] for g in grp]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "ID trùng"

    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    report = {"evidence": [], "absence": [], "problems": []}
    for q in questions:
        if q["answerable"]:
            for gi, grp in enumerate(q["gold_groups"]):
                for g in grp:
                    where = [c["id"] for c in chunks if c["source_id"] == g["source_id"] and c["page"] == g["page"]
                             and contains(c["text"], g["evidence"])]
                    elsewhere = [c["id"] for c in chunks if contains(c["text"], g["evidence"])
                                 and not (c["source_id"] == g["source_id"] and c["page"] == g["page"])]
                    report["evidence"].append({"id": q["id"], "group": gi, "evidence": g["evidence"],
                                               "chunks_containing": where, "same_text_elsewhere": elsewhere})
                    if not where:
                        report["problems"].append(f"{q['id']} nhóm {gi}: evidence không nằm trọn trong chunk nào của trang gold")
        else:
            kws = ABSENCE_KEYWORDS.get(q["id"], [])
            report["absence"].append({"id": q["id"], "question": q["question"], "keywords": {
                k: [c["id"] for c in chunks if norm(k) in norm(c["text"])] for k in kws}})

    def arts(split):
        return {(g["source_id"], g.get("article")) for q in questions if q["answerable"] and q["split"] == split
                for g in q["gold"]}
    overlap = sorted(arts("dev") & arts("test"))
    report["dev_test_article_overlap"] = overlap
    if overlap:
        report["problems"].append(f"Điều trùng giữa dev và test: {overlap}")

    counts = {}
    for q in questions:
        key = f"{q['split']}_{'answerable' if q['answerable'] else 'unanswerable'}"
        counts[key] = counts.get(key, 0) + 1
    report["counts"] = counts

    out = DATA / "benchmark.json"
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report["benchmark_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (DATA / "benchmark_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"counts": counts, "problems": report["problems"], "sha256": report["benchmark_sha256"]},
                     ensure_ascii=False, indent=1))
    for a in report["absence"]:
        print(a["id"], {k: len(v) for k, v in a["keywords"].items()})


if __name__ == "__main__":
    main()
