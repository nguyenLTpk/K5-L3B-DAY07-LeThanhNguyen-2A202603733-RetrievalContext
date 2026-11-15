# Benchmark Day 7 — phần cơ bản (corpus quy chế Day 6)

File: [benchmark.json](benchmark.json) (tạo bằng `python scripts/build_benchmark.py`) ·
Kiểm chứng: [benchmark_verification.json](benchmark_verification.json) ·
**sha256 lúc chốt:** `ba441760942200ec9f0f345cb959178f9c87d3697db63d1da81714d21871dc00` (chốt 2026-11-15, trước lần chạy retrieval/agent đầu tiên của Day 7).

## Thành phần

| Split | Có đáp án | Thiếu bằng chứng | Tổng |
|---|---:|---:|---:|
| dev | 17 (15 câu Day 6 + 2 multi-hop mới) | 4 (mới) | 21 |
| test | 13 (5 câu Day 6 + 6 câu mới + 2 multi-hop mới) | 4 (mới) | 17 |

- **20 câu Day 6** (q01–q20) giữ nguyên câu hỏi, đáp án, evidence, filter; chỉ đổi schema sang Day 7
  (`gold_groups` = danh sách nhóm evidence bắt buộc, mỗi nhóm gồm các evidence thay thế; `gold` = danh sách phẳng).
- **18 câu mới** ([benchmark_day7_additions.json](benchmark_day7_additions.json)) do Claude soạn sau khi đọc text đã chuẩn hóa
  của Day 6; học viên rà soát.
  - Multi-hop: d21, d22 (dev) và t27, t28 (test). Mỗi câu ghi rõ các nhóm evidence bắt buộc (2–3 nhóm, khác Điều/văn bản).
  - Thiếu bằng chứng: d23–d26, t29–t32. Mỗi câu gần chủ đề và có `absence_reason` + từ khóa đã dò trong toàn bộ 178 chunk
    (kết quả dò ở `benchmark_verification.json`, đã đọc ngữ cảnh từng lần xuất hiện).
- **Tách dev/test theo Điều**: không câu test có đáp án nào dùng Điều đã dùng ở dev (script kiểm tra: `dev_test_article_overlap = []`).
- **Rủi ro rò rỉ đã biết**: 5 câu test Day 6 (q16–q20) đã được chạy *một lần* với dense retrieval ở Day 6 (kết quả công khai trong repo Day 6)
  nhưng không được dùng để chỉnh gì ở Day 6 lẫn Day 7. Một chunk có thể chứa 2 Điều liền nhau (q17/q06 cùng trang p15; q18/q09 cùng đơn vị u05).

## Kiểm chứng tự động

`build_benchmark.py` xác nhận mọi evidence nằm trọn trong ít nhất một chunk của index (đúng `source_id` + `page`) sau chuẩn hóa
(chữ thường, bỏ dấu câu) — nghĩa là mọi câu có đáp án đều *có thể* hit. Không đưa gold vào agent hay retriever: `run_agent.py`
chỉ truyền `question` và `filter` (do người dùng chọn trước, có ở q05, q09, q12, q18, q19).

## Định nghĩa metric

- **Hit@k / Evidence Hit@3**: có ≥ 1 nhóm evidence được phủ trong top-k (chunk đúng nguồn + trang + chứa evidence). Với câu 1 nhóm = Day 6.
- **Full@k**: mọi nhóm evidence được phủ trong top-k (chỉ khác Hit@k ở câu multi-hop).
- **Context Evidence Rate**: context cuối gửi model phủ *mọi* nhóm evidence / số câu có đáp án.
- Câu thiếu bằng chứng không vào mẫu số retrieval/context.
