# -*- coding: utf-8 -*-
"""Ghi nhãn chấm thủ công (đã đọc từng câu trả lời + từng citation) thành reviews.json theo định dạng
starter/agent_metrics.py, kèm lý do; rồi chạy agent_metrics cho từng cấu hình.

Người chấm: Claude (trợ lý AI) đọc trace, đối chiếu đáp án + evidence gold và nội dung từng chunk được trích;
học viên rà soát lại. Quy ước:
- answer_grade: 2 đúng và đủ ý được hỏi; 1 đúng một phần hoặc có ý không có trong context; 0 sai/từ chối.
- citation_support: mỗi citation (theo thứ tự trong trace) có trong context VÀ hỗ trợ ý được gắn.
- abstention_correct: status=abstain VÀ nội dung không thêm khẳng định vô căn cứ (định nghĩa của starter).
Mặc định: câu trả lời đúng/đủ và citation hỗ trợ; mọi ngoại lệ ghi trong OVERRIDES với lý do.

Chạy: python scripts/write_reviews.py --split dev
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from day07.metrics import groups_hit  # noqa: E402
from day07.retrievers import chunk_by_id  # noqa: E402

# (split, config, query_id) -> nhãn ngoại lệ
OVERRIDES = {
    ("dev", "A0_dense", "d22"): {
        "answer_grade": 1, "citation_false": ["tt10-2016:u08:st02"],
        "reason": "Kết luận đúng (đủ tiêu chuẩn Giỏi) nhưng ý '3,3 thuộc loại Giỏi (3,2–3,6)' không có trong context: "
                  "chunk tt08-2021:p12:st02 bị loại vì vượt ngân sách 4000 ký tự; model tự điền ('theo quy chế của nhiều cơ sở') "
                  "và gắn citation tt10-2016:u08:st02 không chứa thang điểm."},
    ("test", "A0_dense", "t27"): {
        "answer_grade": 1, "citation_false": ["tt10-2016:u01:st00", "tt16-2015:u07:st01"],
        "reason": "Ý 1 đúng (năm cuối không được chuyển ngành, có evidence p17). Ý 2 sai: khẳng định 'quy chế không quy định' việc xử lý kết quả "
                  "rèn luyện khi chuyển trường, trong khi TT16 Điều 10 khoản 9 quy định được bảo lưu; context chỉ phủ 1/2 nhóm evidence. "
                  "Hai citation tt10-2016:u01:st00, tt16-2015:u07:st01 gắn cho khẳng định sai."},
    ("test", "A3_hybrid_strict", "t27"): {
        "answer_grade": 1, "citation_false": ["tt08-2021:p17:st02", "tt08-2021:p18:st00"],
        "reason": "Ý 1 đúng. Ý 2 sai: 'không có quy định' (TT16 Điều 10 khoản 9 có). Trace: model tự suy filter sai "
                  "(quy_che_cong_tac_sinh_vien) cho 3 lượt tìm 'kết quả rèn luyện'; lượt TT16 cuối không đưa được chunk khoản 9 vào context "
                  "(1/2 nhóm). p17:st02, p18:st00 (Quy chế đào tạo) gắn cho khẳng định sai."},
}
UNANSWERABLE_NOTE = {
    "answered": "status=answered: nội dung nói tài liệu không quy định (không bịa số) nhưng hệ thống không trả trạng thái từ chối "
                "-> không tính abstention đúng theo định nghĩa starter (abstention_correct=false).",
    "abstain": "status=abstain, nội dung nói rõ tài liệu không quy định thông tin được hỏi, không thêm số liệu -> từ chối đúng.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    qs = {q["id"]: q for q in json.loads((ROOT / "data" / "benchmark.json").read_text(encoding="utf-8"))}
    by_id = chunk_by_id()
    base = ROOT / "results" / "agent" / args.split
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in qs.values() if q["split"] == args.split]
    (base / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    configs = args.configs or sorted(p.name for p in base.iterdir() if (p / "traces.json").exists())
    for cfg in configs:
        traces = json.loads((base / cfg / "traces.json").read_text(encoding="utf-8"))
        reviews = []
        for t in traces:
            q = qs[t["query_id"]]
            ov = OVERRIDES.get((args.split, cfg, t["query_id"]), {})
            if t["status"] == "error":
                continue
            r = {"query_id": t["query_id"], "config": cfg}
            false_c = set(ov.get("citation_false", []))
            r["citation_support"] = [c not in false_c for c in t["citations"]]
            r["citation_gold_hit"] = [bool(q["answerable"] and groups_hit(by_id[c], q)) for c in t["citations"]]
            if q["answerable"]:
                r["answer_grade"] = 0 if t["status"] == "abstain" else ov.get("answer_grade", 2)
                r["reason"] = ov.get("reason") or (
                    "Từ chối câu có đáp án -> 0." if t["status"] == "abstain" else
                    f"Đúng và đủ so với đáp án: {q['expected_answer']} Citation đã đọc, hỗ trợ ý được gắn.")
            else:
                r["abstention_correct"] = ov.get("abstention_correct", t["status"] == "abstain")
                r["reason"] = ov.get("reason") or UNANSWERABLE_NOTE[t["status"]]
            reviews.append(r)
        (base / cfg / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
        traces_min = [{"query_id": t["query_id"], "status": t["status"], "events": t["events"], "citations": t["citations"]} for t in traces]
        (base / cfg / "traces_min.json").write_text(json.dumps(traces_min, ensure_ascii=False), encoding="utf-8")
        out = subprocess.run([sys.executable, str(ROOT / "starter" / "agent_metrics.py"), "--cases", str(base / "cases.json"),
                              "--traces", str(base / cfg / "traces_min.json"), "--reviews", str(base / cfg / "reviews.json"),
                              "--split", args.split, "--output", str(base / cfg / "summary_labels.json")],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0:
            raise SystemExit(f"{cfg}: {out.stderr}")
        s = json.loads((base / cfg / "summary_labels.json").read_text(encoding="utf-8"))
        g = s["grounded_answer"]
        print(f"{cfg}: grounded {g['0']}/{g['1']}/{g['2']} mean={g['mean']:.3f} | citation {s['citation_validity']['n']}/{s['citation_validity']['N']} "
              f"| ans_with_cit {s['answer_citation_coverage']['n']}/{s['answer_citation_coverage']['N']} | abstention {s['abstention']['n']}/{s['abstention']['N']} "
              f"| false_abstain {s['false_abstention']['n']}/{s['false_abstention']['N']} | tool_use {s['tool_use']['n']}/{s['tool_use']['N']} | errors {s['provider_errors']}")


if __name__ == "__main__":
    main()
