# -*- coding: utf-8 -*-
"""Nhãn đạt/không đạt (thủ công) cho benchmark nâng cao + tổng hợp theo nhóm.

Người chấm: Claude (trợ lý AI) đọc từng câu trả lời + citation, đối chiếu với đáp án/evidence đã soạn từ ẢNH TRANG GỐC
(không dùng nội dung model trích xuất làm chuẩn); học viên rà soát. Tiêu chí (ADVANCED.md):
- có đáp án: đạt khi đúng và đủ (đơn vị/điều kiện/phép tính), citation xác định trang + vùng evidence, evidence hỗ trợ toàn bộ;
  đúng một phần / sai citation / từ chối -> không đạt;
- thiếu evidence: đạt khi từ chối hoặc nêu rõ không đủ bằng chứng, không thêm khẳng định vô căn cứ;
- lỗi hệ thống/timeout -> không đạt, vẫn ở mẫu số.
Mặc định "đạt"; mọi câu không đạt ghi lý do trong FAILS. Câu đạt có lý do ngắn trong NOTES nếu cần lưu ý.

Chạy: python -X utf8 advanced/scripts/write_labels.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAILS = {
    ("dev", "B0_text", "ad03"): "Trả lời 'tác động Cao' và chỉ ghi 'Khẩn cấp' — sai: ô đúng là khẩn cấp Cao, tác động Trung bình (text layer làm mất cấu trúc ô).",
    ("dev", "B0_text", "ad05"): "Trả lời báo cáo không có số cụ thể — sai: Hình 13 ghi CPS 246, ADS 63 (nhãn chỉ có trên ảnh).",
    ("dev", "B0_text", "ad06"): "Cho rằng 'lúa→ngô' âm nhiều nhất (≈ −150 đến −200) — sai: lúa→ngô cao nhất (≈ 390), lúa→nuôi tôm thấp nhất.",
    ("dev", "B0_text", "ad07"): "Trả lời 'Thiệt hại về cơ sở vật chất' 27% — sai hàng; đúng: Thiếu nguồn cung nguyên liệu đầu vào 33%.",
    ("dev", "B0_text", "ad10"): "Nói ma trận không liệt kê hành động này — sai (có ở ô khẩn cấp Trung bình × tác động Cao).",
    ("dev", "B0_text", "ad11"): "Nói không có số TWh điện than năm 2040 trong Hình 13 — thiếu nhóm evidence 1 (26 TWh).",
    ("dev", "B1_visual", "ad03"): "Nêu đúng tác động Trung bình nhưng không nêu mức khẩn cấp (Cao) — thiếu một điều kiện.",
    ("dev", "B1_visual", "ad06"): "Chunk trích xuất Hình 16 ghép sai cột-nhãn; trả lời cao nhất là 'AWD/SRI khu tưới tiêu hạn chế ≈ 400' — sai (đúng: lúa→ngô).",
    ("dev", "B1_visual", "ad10"): "Đặt hành động vào ô khẩn cấp Trung bình × tác động Trung bình (theo chunk trích xuất sai) và mô tả Bảng 6 của ô đó — sai ô.",
    ("dev", "B2_visual_view", "ad10"): "view_page đọc đúng (Trung bình × Cao) nhưng câu trả lời theo chunk trích xuất sai (Trung bình × Trung bình) — sai ô.",
    ("dev", "B4_view_prefer", "ad10"): "Lần này view_page đọc ô Trung bình × Trung bình (vision không ổn định giữa các lần) và agent tin view_page — sai ô.",
    ("dev", "B4_view_prefer", "ad11"): "Điện than ADS năm 2040 trả lời 124 TWh — đó là giá trị của Khí; đúng là 26 TWh (view_page đọc nhầm series, quy tắc ưu tiên view_page khiến bỏ qua chunk trích xuất đúng).",
    ("test", "B2_visual_view", "at07"): "Đủ ba loại thiên tai nhưng ý 2 chỉ nêu 'duyên hải miền Trung', thiếu dải ven biển miền Bắc, và mô tả sai màu "
                                        "('đỏ đậm' — bản đồ bão lũ màu xanh lá) -> đúng một phần, không đạt.",
    ("test", "B2_visual_view", "at09"): "Trả lời TP. Hồ Chí Minh và Cần Thơ — sai; trên ảnh gốc vùng đậm nhất là dải miền Trung quanh Đà Nẵng "
                                        "(view_page đọc sai bản đồ raster độ phân giải thấp).",
    ("test", "B2_visual_view", "at17"): "Ý 2 đúng (19,69, Phụ lục 1) nhưng ý 1 sai: trả lời 'khoảng năm 2010' (đó là lúc LUCF chuyển ngược về âm); "
                                        "đúng là ngay sau 2000. Agent không gọi view_page, tin mô tả trích xuất của Hình 4.",
    ("test", "B0_text", "at07"): "Đủ ba loại thiên tai nhưng ý 2 sai: nêu ven biển + ĐBSCL + vùng núi phía Bắc (suy từ văn bản, không đọc bản đồ bão lũ).",
    ("test", "B0_text", "at08"): "Nói tài liệu không mô tả chú giải — chú giải 'Percentage exposure' chỉ có trên ảnh raster.",
    ("test", "B0_text", "at09"): "Không nêu được thành phố (nói ĐBSCL có nguy cơ cao) — đúng là Đà Nẵng.",
    ("test", "B0_text", "at13"): "Từ chối ('Hình 4 không cung cấp con số') — câu có đáp án trên ảnh (≈ −70).",
    ("test", "B0_text", "at14"): "Trả lời Công nghiệp −9,4% — sai (−9,4 là mốc trục); đúng là Vận tải −6,3%.",
    ("test", "B0_text", "at16"): "Khí đốt 25% công suất — sai (25% là năng lượng tái tạo; đúng 13%), ý 17% đúng.",
    ("test", "B0_text", "at17"): "Ý 2 đúng (19,69) nhưng ý 1 nói báo cáo không có năm chuyển đổi — thiếu nhóm evidence Hình 4.",
}
NOTES = {
    ("test", "B2_visual_view", "at13"): "GIÁP RANH: model đọc ≈ −60 tại năm 2000 (ảnh gốc ≈ −72). Nằm trong khoảng chấp nhận −60…−75 đã ghi trong benchmark "
                                        "trước khi chạy và nêu đúng là giá trị âm (hấp thụ ròng) -> đạt theo tiêu chí đã chốt. Nếu chấm chặt hơn tiêu chí, câu này không đạt.",
    ("test", "B2_visual_view", "at21"): "Đủ 3 biện pháp; citation ccdr:p057:v1 là phần tử 'Hộp 7' (box) trang PDF 57 có box_2d.",
    ("dev", "B2_visual_view", "ad06"): "Đúng tên hai biện pháp (giá trị ≈ −180 của nuôi tôm lệch so với ≈ −80 trên ảnh, tiêu chí không bắt buộc giá trị).",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    qs = {q["id"]: q for q in json.loads((ROOT / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))}
    base = ROOT / "results" / args.split
    configs = args.configs or sorted(p.name for p in base.iterdir() if (p / "traces.json").exists())
    table = {}
    for cfg in configs:
        traces = json.loads((base / cfg / "traces.json").read_text(encoding="utf-8"))
        labels, by_group = [], defaultdict(lambda: [0, 0])
        for t in traces:
            q = qs[t["query_id"]]
            key = (args.split, cfg, t["query_id"])
            if t["status"] == "error":
                passed, reason = False, f"Lỗi hệ thống: {t['error']}"
            elif key in FAILS:
                passed, reason = False, FAILS[key]
            elif q["answerable"]:
                passed, reason = True, NOTES.get(key, f"Đúng và đủ theo tiêu chí '{q.get('accept', '')}'; citation {t['citations']} chỉ đúng trang/phần tử evidence.")
            else:
                passed, reason = True, NOTES.get(key, f"Nêu rõ báo cáo không có thông tin được hỏi, không bịa số (status={t['status']}).")
            labels.append({"query_id": t["query_id"], "group": q["group"], "pass": passed, "status": t["status"],
                           "citations": t["citations"], "reason": reason})
            by_group[q["group"]][0] += passed
            by_group[q["group"]][1] += 1
        n_pass, n = sum(l["pass"] for l in labels), len(labels)
        summary = {"config": cfg, "split": args.split, "pass": f"{n_pass}/{n}", "pass_rate": round(100 * n_pass / n, 1),
                   "by_group": {g: f"{a}/{b}" for g, (a, b) in sorted(by_group.items())},
                   "errors": sum(t["status"] == "error" for t in traces), "grader": "Claude (AI), học viên rà soát"}
        (base / cfg / "labels.json").write_text(json.dumps({"summary": summary, "labels": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
        table[cfg] = summary
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
