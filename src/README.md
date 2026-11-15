# Code Day 7 — cách chạy

Python 3.11 (đã chạy 3.11.9, Windows 11, CPU). Trên Windows nên bật UTF-8: `set PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8=1`).

```bash
pip install -r requirements.txt
# .env ở thư mục gốc (đã .gitignore):  GEMINI_API_KEY=...
```

## Phần cơ bản (corpus quy chế Day 6)

| Bước | Lệnh | Kết quả |
|---|---|---|
| 0. Nhập Day 6 | `python scripts/import_day6.py` | Copy Chroma Day 6 → `data/day6_store/`, xuất `data/chunks.json` (178 chunk), `data/day6_import.json`. Cần repo Day 6 cạnh repo này hoặc `DAY6_DIR=<đường dẫn>`; model e5-small đọc từ `DAY6_DIR/models/`. |
| 1. Benchmark | `python scripts/build_benchmark.py` | `data/benchmark.json` (38 câu) + `data/benchmark_verification.json` |
| 2. Retrieval | `python scripts/eval_retrieval.py --split dev --methods dense bm25 hybrid rerank multiquery` (thêm `--use-filter` cho filter người dùng) | `results/retrieval/<split>/*.json`, `summary.json` |
| 3. Agent | `python scripts/run_agent.py --split dev --configs A0_dense A1_hybrid A2_rerank_adj A3_hybrid_strict` | `results/agent/<split>/<config>/traces.json`, `summary_auto.json` |
| 4. Nhãn + metric | `python scripts/write_reviews.py --split dev` | `reviews.json` (nhãn thủ công + lý do), `summary_labels.json` (qua `starter/agent_metrics.py`) |
| Test | như bước 2–4 với `--split test` | chỉ chạy sau khi chốt cấu hình (`results/selection_basic.json`) |
| Unit test | `python -X utf8 -m unittest discover -s tests` | 25 test (mock, không gọi mạng) |

Module (`src/day07/`): `config.py` (đường dẫn, .env) · `embedder.py` (e5-small ONNX, tương thích index Day 6) ·
`retrievers.py` (dense / BM25 / hybrid RRF / rerank LLM / multi-query) · `agent.py` (Gemini function calling,
context budget, kiểm citation, strict abstain) · `metrics.py` (Hit@k, Full@k, Context Evidence Rate) ·
`llm.py` (REST client, retry, đếm token) · `pricing.py` · `configs.py` (cấu hình A0–A3 + giả thuyết).

## Thử thách nâng cao (báo cáo CCDR Việt Nam 2022, 104 trang)

```bash
python advanced/scripts/fetch_pdf.py          # tải PDF, kiểm sha256
python -I advanced/scripts/inspect_pdf.py     # thống kê trang
python -I advanced/scripts/render_pages.py    # PNG 144 DPI + text layer
python -I advanced/scripts/extract_visuals.py # vision: mô tả/chép bảng-biểu đồ-bản đồ (cache advanced/data/visual/)
python -I advanced/scripts/build_index.py     # chunk text + visual -> e5-small -> Chroma
python -I advanced/scripts/verify_benchmark.py
python -I advanced/scripts/run_agent.py --split dev --configs B0_text B1_visual B2_visual_view B3_rerank_view
```

Chi tiết: [advanced/README.md](../advanced/README.md).

## Lưu ý quota

Key dùng trong lab thuộc **Gemini API Free Tier**: `gemini-3.5-flash` chỉ 20 request/ngày, các model Flash-Lite giới hạn
theo phút/ngày. `llm.py` retry có backoff cho 429/5xx và báo lỗi ngay nếu hết quota ngày (không giả kết quả). Chi phí
trong báo cáo là ước tính theo bảng giá trả phí từ số token thực đếm được; chi phí thực tế đã trả là $0.
