# Thử thách nâng cao — Báo cáo Quốc gia về Khí hậu và Phát triển cho Việt Nam (CCDR, 2022)

## Tài liệu

| Mục | Giá trị |
|---|---|
| Tên | Việt Nam: Báo cáo Quốc gia về Khí hậu và Phát triển (bản tiếng Việt), Ngân hàng Thế giới, tháng 7/2022 |
| Nguồn PDF | https://www.ifc.org/content/dam/ifc/doc/mgrt/ccdr-full-report-vie-1107.pdf (bản tiếng Anh: http://hdl.handle.net/10986/37618, CC BY 3.0 IGO) |
| sha256 | `cb0c7f23a88ceadcbb5ad837e31caa73a7cc201db4140aed45820dcbcf666a18` (3,9 MB) |
| Số trang | 104 trang PDF (trang in = trang PDF − 10 với trang nội dung) |
| Quyền sử dụng | © 2022 IBRD/World Bank. Trang 2: "toàn bộ hoặc một phần báo cáo này có thể được sao chép lại cho các mục đích phi thương mại miễn là có ghi nhận đầy đủ về tác phẩm". Repo không lưu PDF/ảnh trang; tải lại bằng `scripts/fetch_pdf.py`. |
| Nội dung trực quan | 6 bảng đánh số + bảng phụ lục, 22 hình (biểu đồ cột/đường/tròn/treemap/scatter, bản đồ raster, sơ đồ), 9 hộp, ảnh bìa chương (trang trí) |

## Pipeline

1. `fetch_pdf.py` → `inspect_pdf.py` (thống kê ký tự/ảnh/đối tượng vector từng trang: `data/page_stats.json`)
   → `render_pages.py` (PNG 144 DPI + text layer từng block: `data/page_text.json`).
2. `extract_visuals.py`: vision model đọc ảnh trang + text layer, trả JSON từng phần tử trực quan (nhãn, loại, box_2d,
   mô tả, chép bảng markdown/giá trị biểu đồ/nhãn bản đồ). Cache `data/visual/pNNN.json`, crop `data/crops/`.
   - Chạy cho 48 trang có ảnh nhúng, ≥ 8 đối tượng vector hoặc chú thích Hình/Bảng/Phụ lục (`data/visual_page_selection.json`);
     56 trang còn lại chỉ có chữ → nội dung đã đủ trong text layer.
   - Model: 18 trang đầu bằng `gemini-3.5-flash` (đến khi hết quota free tier 20 request/ngày), phần còn lại `gemini-3.1-flash-lite`
     (ghi trong trường `model` của từng file).
3. `build_index.py`: chunk text (≤ 900 ký tự, bỏ header/footer) + 1 chunk/phần tử trực quan informative → e5-small (giống Day 6) →
   Chroma `day07_ccdr_e5small`. Thống kê: `data/index_stats.json`.
4. Agent (`src/day07/adv_agent.py`): `search_report` (hybrid dense+BM25 RRF, tùy chọn rerank) + `view_page` (đọc lại ảnh gốc trang
   bằng vision model lúc trả lời) + `final_answer` với citation `ccdr:pNNN:tK` (đoạn văn), `ccdr:pNNN:vK` (bảng/hình, có nhãn + box_2d)
   hoặc `ccdr:pNNN:page` (trang đã xem).

## Benchmark

[benchmark/benchmark.json](benchmark/benchmark.json) — soạn bằng cách xem **ảnh trang gốc** (không dùng nội dung model trích xuất),
chốt trước khi chạy index/agent. sha256 và kiểm tra: [benchmark/verification.json](benchmark/verification.json).

| Split | Bảng | Ảnh/sơ đồ | Biểu đồ | Multi-hop | Text | Thiếu evidence | Tổng |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 3 | 2 | 4 | 3 | 3 | 3 | 18 |
| test | 6 | 5 | 5 | 4 | 4 | 6 | 30 |

Dev/test không dùng chung trang evidence nào. Mỗi câu có: nhóm chính, câu hỏi, đáp án (hoặc hành vi từ chối + cách đã kiểm tra),
evidence theo trang PDF/trang in/phần tử/vùng (hàng-cột, series, vùng bản đồ), đơn vị, phép tính và tiêu chí chấm (`accept`).

## Chạy và chấm

`run_agent.py --split dev|test --configs ...` → `results/<split>/<config>/traces.json`. Mỗi câu test được chấm **đạt/không đạt**
thủ công theo `accept` và đối chiếu ảnh trang gốc: `results/test/<config>/labels.json`.
