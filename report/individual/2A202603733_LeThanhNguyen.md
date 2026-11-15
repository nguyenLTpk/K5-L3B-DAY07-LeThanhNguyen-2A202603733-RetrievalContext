# Báo cáo cá nhân — Day 7: Agentic RAG

**Họ và tên:** Lê Thành Nguyên · **MSSV:** 2A202603733 · **Lớp:** K5-L3B
**Corpus phần cơ bản:** quy chế sinh viên Day 6 (TT08/2021, TT10/2016, TT16/2015) · **Thử thách nâng cao:** Báo cáo CCDR Việt Nam 2022 (104 trang)
**Cách chạy:** [src/README.md](../../src/README.md) · **Kết quả:** [results/](../../results/), [advanced/results/](../../advanced/results/)

> **Tóm tắt:** phần cơ bản — chọn A3_hybrid_strict; test: grounded 1,92/2, citation hợp lệ 17/19, từ chối đúng 4/4, từ chối nhầm 0/13
> (baseline A0: 1,92/2, 21/23, 1/4, 0/13). Thử thách nâng cao — B2_visual_view đạt **27/30 = 90,0%** test (không làm tròn), 0 lỗi hệ thống.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Dùng lại nguyên corpus và index Day 6 (repo `K5-L3B-DAY06-LeThanhNguyen-2A202603733-DataIngestion`): 3 văn bản quy phạm pháp luật
    (không thuộc đối tượng bảo hộ quyền tác giả — khoản 2 Điều 15 Luật SHTT), 51 đơn vị nội dung (20 trang PDF TT08 + 31 đơn vị HTML),
    **178 chunk** chiến lược B (theo Chương/Điều/khoản, ≤ 800 ký tự) — chiến lược đã chọn ở Day 6.
  - `scripts/import_day6.py` copy vector store Day 6 sang `data/day6_store/` (Day 7 không ghi vào repo Day 6) và xuất `data/chunks.json`
    (`id, text, source_id, source, page, metadata`); sha256 trong [data/day6_import.json](../../data/day6_import.json).
- **Vector store, embedding model, phiên bản và số chiều:**
  - ChromaDB 1.5.9 `PersistentClient`, collection `day06_b_e5small`, cosine/HNSW — **dùng nguyên vector đã lưu ở Day 6**, không embed lại.
  - Query embedding: `intfloat/multilingual-e5-small` revision `614241f6`, ONNX (sha256 `ca456c06…` kiểm tra khi nạp), 384 chiều,
    mean pooling + L2, tiền tố `query: ` — giống hệt Day 6 nên tương thích index. Kiểm tra hồi quy: dense top-3 Day 7 trùng Day 6.
- **Agent model, cách chạy và phân chia dev/test:**
  - LLM điều phối: Gemini `gemini-3.5-flash-lite` (REST `generateContent`, function calling, mode ANY), không thinking. Rerank/multi-query
    cũng dùng model này. Key thuộc **Gemini API Free Tier** (500 request/ngày/model với Flash-Lite, 20/ngày với 3.5 Flash).
  - Benchmark [data/benchmark.json](../../data/benchmark.json) (mô tả: [data/BENCHMARK_DAY7.md](../../data/BENCHMARK_DAY7.md)), chốt sha256
    `ba441760…` trước lần chạy đầu: **dev 21 câu (17 có đáp án, 4 thiếu evidence), test 17 câu (13 có đáp án, 4 thiếu evidence)**.
    20 câu Day 6 giữ nguyên; 18 câu mới gồm 4 câu multi-hop (2–3 nhóm evidence, khác Điều/văn bản) và 8 câu thiếu evidence.
    Dev/test tách theo Điều (0 Điều trùng).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: `gold_groups` (nhóm bắt buộc × evidence thay thế), mỗi evidence được script xác nhận nằm trọn trong ≥ 1 chunk đúng nguồn + trang.
  - Câu thiếu evidence: chủ đề gần (học bổng, nghỉ hè, trọng số điểm thi, vay vốn, thời hạn khiếu nại, trừ điểm rèn luyện, lệ phí, hiến máu);
    ghi `absence_reason` + từ khóa đã dò trên cả 178 chunk và đọc ngữ cảnh từng lần xuất hiện.
  - Chấm agent: grounded 0/1/2, citation hợp lệ (có trong context **và** hỗ trợ ý được gắn — đọc từng chunk), abstention đúng = `status=abstain`
    và không bịa (đúng định nghĩa `starter/agent_metrics.py`). Nhãn + lý do từng câu: `results/agent/<split>/<config>/reviews.json`.

## Retrieval và thử nghiệm

**5 phương pháp** trên cùng corpus/index/câu hỏi/nhãn (retrieval-only, query gốc, luôn xuất top-10; [results/retrieval/](../../results/retrieval/)):

| Phương pháp | Cách tìm |
|---|---|
| dense | e5-small → Chroma Day 6 (cosine), `where regulation=` khi có filter người dùng |
| bm25 | BM25 (k1 1,5, b 0,75) trên âm tiết tiếng Việt giữ dấu + bigram âm tiết |
| hybrid | RRF (k=60) của dense top-20 và BM25 top-20 |
| rerank | hybrid top-15 → Gemini chấm 0–3 từng đoạn → xếp lại |
| multiquery | Gemini viết 3 truy vấn (không trả lời) + query gốc → hybrid mỗi truy vấn → RRF; truy vấn viết lại lưu trong trace |

Retrieval-only trên **dev** (17 câu có đáp án; `bm25_unigram` = BM25 không bigram để đo ảnh hưởng tách từ):

| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | Full@3 | Full@5 | MRR@10 | Trung vị/câu | Chi phí ước tính |
|---|---|---|---|---|---|---|---|---|
| dense (Day 6) | không | 12/17 (71%) | 15/17 (88%) | 14/17 | 16/17 | 0,814 | 0,04 s | $0 |
| dense (Day 6) | người dùng | 12/17 (71%) | 16/17 (94%) | 15/17 | 16/17 | 0,828 | 0,05 s | $0 |
| bm25 (+ bigram) | không | 16/17 (94%) | 16/17 (94%) | 14/17 | 15/17 | 0,949 | 0,006 s | $0 |
| bm25_unigram | không | 14/17 (82%) | 16/17 (94%) | 14/17 | 15/17 | 0,887 | 0,004 s | $0 |
| hybrid RRF | không | 15/17 (88%) | 16/17 (94%) | 15/17 | 16/17 | 0,924 | 0,05 s | $0 |
| hybrid RRF | người dùng | 15/17 (88%) | 17/17 (100%) | 16/17 | 16/17 | 0,931 | 0,05 s | $0 |
| rerank LLM | không | **17/17 (100%)** | 17/17 (100%) | 16/17 | 16/17 | **1,000** | 1,9 s | $0,050 / 21 câu |
| multiquery | không | 14/17 (82%) | 17/17 (100%) | 16/17 | 16/17 | 0,902 | 1,0 s | $0,004 / 21 câu |

Nhận xét dev: BM25 mạnh hơn dense trên corpus quy phạm vì câu hỏi trùng thuật ngữ/con số với văn bản; bigram âm tiết tăng Hit@1 14→16.
Hybrid giữ Hit@3 của BM25 và ổn định hơn ở câu diễn đạt lại; rerank đưa đoạn đúng lên hạng 1 ở mọi câu nhưng thêm ~1,9 s và 1 lời gọi LLM.
Full@3 < Hit@3 do các câu multi-hop cần 2–3 nhóm evidence không cùng nằm trong top-3 của **một** truy vấn — đây là lý do dùng agent nhiều lượt.

Retrieval-only trên **test** (chạy sau khi chốt; bảng đủ 5 phương pháp ở phần Test):

| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | Full@3 | MRR@10 |
|---|---|---|---|---|---|
| dense (Day 6) | không / người dùng | 11/13 (85%) | 13/13 (100%) | 13/13 | 0,923 |
| bm25 | không | 12/13 (92%) | 13/13 (100%) | 13/13 | 0,962 |
| hybrid RRF | không / người dùng | 12/13 (92%) | 13/13 (100%) | 13/13 | 0,962 |

**Cấu hình agent** (cùng system prompt, cùng LLM; [src/day07/configs.py](../../src/day07/configs.py)):

| Cấu hình | Tool / top-k / lượt / context budget | Giả thuyết / thay đổi |
|---|---|---|
| A0_dense (baseline) | `search_regulations` = dense Day 6, top-3, ≤ 4 lượt, 4.000 ký tự | Như starter; dự đoán hụt câu diễn đạt lại (q14) |
| A1_hybrid (phương pháp 2 + cải thiện 1) | hybrid, top-4, ≤ 6 lượt, 6.000 ký tự | BM25 bắt thuật ngữ/con số; ngân sách lớn hơn giữ đủ evidence multi-hop |
| A2_rerank_adj (phương pháp 3 + cải thiện 2) | rerank LLM trên 15 candidate, top-4, + tool `read_adjacent`, 6.000 ký tự | Đưa đoạn trực tiếp lên đầu; đọc chunk liền kề khi Điều bị tách |
| A3_hybrid_strict (cải thiện 3, thêm sau lỗi dev) | A1 + trường bắt buộc `tai_lieu_neu_truc_tiep` trong `final_answer` (false → abstain) | Sửa lỗi model viết "không quy định" nhưng `status=answered` |

Agent trên **dev** (21 câu; nhãn thủ công `results/agent/dev/<config>/reviews.json`, tổng hợp bằng `starter/agent_metrics.py`):

| Cấu hình | Hit@1 / Hit@3 (lần search đầu) | Context Evidence Rate | Grounded 0/1/2 (TB /2) | Citation hợp lệ | Từ chối đúng | Từ chối nhầm | Search/câu | Trung vị | Token in/out | Chi phí ước tính |
|---|---|---|---|---|---|---|---|---|---|---|
| A0_dense | 13/17 / 15/17 | 16/17 (94%) | 0/1/16 (1,94) | 21/22 | 2/4 | 0/17 | 2,24 | 57 s* | 120.687 / 4.161 | $0,047 |
| A1_hybrid | 14/17 / 16/17 | 17/17 (100%) | 0/0/17 (2,00) | 25/25 | 2/4 | 0/17 | 2,71 | 4 s* | 184.199 / 4.768 | $0,067 |
| A2_rerank_adj | 17/17 / 17/17 | 17/17 (100%) | 0/0/17 (2,00) | 25/25 | 2/4 | 0/17 | 2,86 | 98 s* | 350.962 / 31.099 | $0,183 |
| **A3_hybrid_strict** | 15/17 / 16/17 | 17/17 (100%) | 0/0/17 (2,00) | 22/22 | **4/4** | 0/17 | 3,19 | 61 s* | 229.133 / 5.428 | $0,082 |

Tool use 21/21 ở mọi cấu hình; mọi câu trả lời `answered` đều có citation; 0 lỗi hệ thống. *Thời gian phụ thuộc hàng đợi/429 của free tier (các cấu hình chạy song song), không phản ánh tốc độ phương pháp.

**Hai (ba) lỗi dev, nguyên nhân, cách thử:**

1. **A0 · d22 (multi-hop 3 nhóm) — grounded 1/2.** Lượt search thứ 4 của A0 *có* trả `tt08-2021:p12:st02` (thang 3,2–3,6 = Giỏi) nhưng context đã
   3.769/4.000 ký tự nên chunk bị đánh dấu `over_budget` và không gửi model; model tự điền "3,2–3,6 theo quy chế của nhiều cơ sở" và gắn citation
   `tt10-2016:u08:st02` (không chứa thang điểm) → citation 21/22. Nguyên nhân: **ngân sách context**, không phải retrieval. Sửa: A1 tăng budget
   6.000 + hybrid → d22 có đủ 3 nhóm evidence, grounded 2/2; Context Evidence Rate 16/17 → 17/17.
2. **A0/A1/A2 · d24, d25 (thiếu evidence) — abstention 2/4.** Nội dung trả lời đúng ("quy chế không quy định số tuần nghỉ hè / tỷ lệ tối thiểu
   điểm thi"), không bịa số, nhưng `status=answered` kèm citation tới đoạn cùng chủ đề (Điều 6 TT08, Điều 9 TT08). Nguyên nhân: model coi
   "tài liệu không quy định X" là một câu trả lời có nguồn; hướng dẫn trong prompt chưa đủ (đã thêm từ smoke test 3 câu, vẫn sai). Sửa: A3 bắt
   khai báo `tai_lieu_neu_truc_tiep` → abstention **4/4**, không từ chối nhầm câu có đáp án (0/17).
3. (retrieval-only) **q09 không filter**: dense, BM25 đều không có Điều 9 TT16 trong top-3 vì "xuất sắc/tốt" trùng ở TT10 Điều 7 và TT08 Điều 10;
   có filter người dùng `regulation` thì dense 16/17, hybrid 17/17. **q14** (diễn đạt lại "vi phạm tiếp" ≠ "tái phạm"): dense hụt, BM25/hybrid tìm thấy;
   ở agent, A2 dùng `read_adjacent` lấy phần "b) Cảnh cáo" bị tách khỏi câu dẫn.

**Chọn cấu hình** (tiêu chí ghi trước khi chạy dev, [results/selection_basic.json](../../results/selection_basic.json)): (1) grounded TB → (2) abstention
đúng rồi từ chối nhầm → (3) Context Evidence Rate → (4) chi phí/thời gian; chênh ≤ 1 câu là hòa. A1, A2, A3 hòa ở (1) = 2,00; A3 thắng (2) 4/4 so với 2/4.
**Chọn A3_hybrid_strict**, baseline test A0_dense. A2 bị loại vì tốn ~2,2 lần mà không hơn ở tiêu chí nào. Lưu ý trung thực: tiêu chí (2) ghi trước
không nói rõ tính theo status hay nội dung; nếu tính theo nội dung, A1/A2/A3 cùng 4/4 và A1 rẻ hơn A3 ~18% → sẽ chọn A1.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool: `search_regulations(query, regulation?)`, `read_adjacent(chunk_id, before|after)` (chỉ A2), `final_answer(status, answer, citations[, tai_lieu_neu_truc_tiep])`.
    Function calling mode ANY: mỗi lượt model buộc gọi 1 tool.
  - Filter người dùng (q05, q09, q12, q18, q19) luôn được hệ thống áp dụng; `regulation` do model tự suy ra ghi riêng `inferred_filter` trong trace.
  - Dừng: `final_answer` hợp lệ; hết `max_tool_calls` → chỉ cho phép `final_answer`; citation sai → trả lỗi cho model sửa 1 lần, sai tiếp → abstain.
  - Context: mỗi chunk gửi 1 lần (dedup), tổng ký tự ≤ budget, chunk vượt budget chỉ báo ID (`over_budget` trong trace).
  - Citation phải là ID đã gửi model; ID hợp lệ chưa đủ — đã đọc từng chunk được trích để chấm hỗ trợ.
  - Lỗi model/vector store: retry có backoff cho 429/5xx; hết quota ngày → báo lỗi, ghi `status=error`, không giả kết quả.

### Test (sau khi chốt)

Chạy 2026-11-15 sau 18:56 (giờ VN), đúng cấu hình đã chốt, mỗi cấu hình **một lần**. Ghi chú quy trình: lượt chạy test đầu (03:00) bị dừng giữa
chừng vì `gemini-3.5-flash-lite` chạm giới hạn free tier 500 request/ngày; lượt đó không ghi file kết quả nào (không xem được câu trả lời test),
cấu hình không đổi giữa hai lượt.

**Retrieval-only trên test** (rerank/multiquery chạy sau khi chốt):

| Phương pháp | Hit@1 | Evidence Hit@3 | Full@3 | MRR@10 | Trung vị/câu | Chi phí ước tính |
|---|---|---|---|---|---|---|
| dense (Day 6) | 11/13 (85%) | 13/13 (100%) | 13/13 | 0,923 | 0,05 s | $0 |
| bm25 | 12/13 (92%) | 13/13 (100%) | 13/13 | 0,962 | 0,008 s | $0 |
| hybrid RRF | 12/13 (92%) | 13/13 (100%) | 13/13 | 0,962 | 0,05 s | $0 |
| rerank LLM | **13/13 (100%)** | 13/13 (100%) | 13/13 | **1,000** | 2,4 s | $0,040 / 17 câu |
| multiquery | 10/13 (77%) | 13/13 (100%) | 12/13 | 0,885 | 1,5 s | $0,003 / 17 câu |

**Agent trên test** — baseline vs cấu hình đã chọn (nhãn đọc **toàn bộ 17 câu** mỗi cấu hình: `results/agent/test/<config>/reviews.json`):

| Cấu hình | Hit@1 / Hit@3 (lần search đầu) | Context Evidence Rate | Context 0/1/2 | Grounded 0/1/2 (TB /2) | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Search/câu | Trung vị | Token in/out | Chi phí ước tính |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_dense (baseline) | 11/13 / 13/13 | 12/13 (92%) | 0/1/12 | 0/1/12 (1,92) | 21/23 | 16/16 | 1/4 | 0/13 | 2,41 | 7 s | 103.285 / 4.009 | $0,041 |
| **A3_hybrid_strict** (chọn) | 12/13 / 13/13 | 12/13 (92%) | 0/1/12 | 0/1/12 (1,92) | 17/19 | 13/13 | **4/4** | 0/13 | 3,12 | 47 s | 177.306 / 4.668 | $0,065 |

Tool use 17/17, 0 lỗi hệ thống ở cả hai. Context 0/1/2: 2 = context có đủ mọi nhóm evidence + nguồn (12 câu); 1 = t27 có evidence ý 1, thiếu ý 2.

- **Kết luận test:** A3 giữ chất lượng trả lời như baseline (1,92/2) và sửa đúng lỗi đã thấy ở dev — từ chối đúng **4/4 so với 1/4** (A0 viết
  "không quy định" nhưng `status=answered` ở t29–t31), không từ chối nhầm. Đổi lại A3 tốn thêm ~58% chi phí (nhiều lượt tìm hơn ở câu thiếu evidence).
  Retrieval trên test bão hòa (Hit@3 13/13 mọi phương pháp), nên test không tách được lợi ích hybrid so với dense như dev đã cho thấy.
- **Lỗi test chung của hai cấu hình — t27 (multi-hop, grounded 1):** ý 1 đúng; ý 2 khẳng định sai "quy chế không quy định" kết quả rèn luyện khi chuyển trường
  (TT16 Điều 10 khoản 9 có quy định bảo lưu). Trace A3: model **tự suy filter sai** `quy_che_cong_tac_sinh_vien` cho 3 lượt tìm "kết quả rèn luyện",
  lượt tìm đúng TT16 cuối cùng không đưa được chunk khoản 9 vào context. Citation gắn cho khẳng định sai bị chấm không hỗ trợ (A0 21/23, A3 17/19).
  Hướng sửa (chưa thử, vì đã chốt): không cho model tự đặt filter khi câu hỏi không nêu văn bản, hoặc bắt buộc tìm lại không filter trước khi kết luận "không có".

**Ví dụ trình bày** (trace trong `results/agent/test/A3_hybrid_strict/traces.json`):

| Query ID | Loại | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t24 | có đáp án | 1 search hybrid → `tt16-2015:u07:st01` hạng đầu | "…thông báo cho người học biết trước 20 ngày trước khi ban hành quyết định chính thức [tt16-2015:u07:st01]" | grounded 2; citation chứa nguyên văn evidence |
| t31 | thiếu evidence | 6 search ("lệ phí chuyển trường", …) | `status=abstain`: quy chế không quy định lệ phí chuyển trường | abstention đúng; A0 cùng nội dung nhưng `answered` → không tính |
| t27 | lỗi | 6 search, 3 lượt bị filter sai văn bản | đúng ý 1, sai ý 2 ("không có quy định") | grounded 1; 2 citation không hỗ trợ |

## Thử thách nâng cao

Chi tiết: [advanced/README.md](../../advanced/README.md).

- **Tài liệu, nguồn, quyền sử dụng, số trang, phiên bản:** "Việt Nam: Báo cáo Quốc gia về Khí hậu và Phát triển" (Ngân hàng Thế giới, 7/2022), bản
  tiếng Việt, **104 trang PDF**, sha256 `cb0c7f23…`; nguồn https://www.ifc.org/content/dam/ifc/doc/mgrt/ccdr-full-report-vie-1107.pdf (bản tiếng Anh
  CC BY 3.0 IGO: hdl.handle.net/10986/37618). Trang 2 cho phép sao chép phi thương mại khi ghi nguồn; repo không lưu PDF/ảnh trang (tải bằng
  `advanced/scripts/fetch_pdf.py`).
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Text: text layer PyMuPDF theo block, bỏ header/footer, chunk ≤ 900 ký tự trong trang → **426 chunk text** (98 trang có text).
  - Trực quan: vision model đọc ảnh trang (144 DPI) + text layer, trả JSON từng phần tử (nhãn, loại, `box_2d`, mô tả, chép bảng markdown /
    giá trị biểu đồ / nhãn-chú giải bản đồ) → **40 chunk trực quan** (16 bảng, 16 biểu đồ, 4 bản đồ, 1 sơ đồ, 1 infographic, 2 hộp; 8 ảnh trang trí
    không index). Đủ 22/22 Hình, 6/6 Bảng và các bảng Phụ lục 1, 3, 4. Vision chạy trên 48 trang có ảnh/đồ họa vector/chú thích Hình-Bảng; 56 trang
    chỉ có chữ dùng text layer. 6 trang không có chunk: 21, 47, 75, 89, 103 (trang trắng/chỉ số trang) và 104 (bìa sau).
  - Model trích xuất: 18 trang đầu bằng `gemini-3.5-flash` (đến khi hết quota 20/ngày), 39 trang còn lại `gemini-3.1-flash-lite`.
  - Embedding e5-small (như Day 6) → Chroma `day07_ccdr_e5small`; 11 chunk bảng dài bị cắt ở 512 token **khi embed** (BM25 và nội dung gửi model vẫn đủ).
  - **Kiểm tra trích xuất với bản gốc** (mẫu 6 phần tử dev): 5 đúng (Hình 1, 6, 8, 13, bảng Phụ lục 3); **Hình 16 sai** — ghép nhầm cột-nhãn
    (cho "lúa→ngô ≈ 0" và "AWD/SRI tưới hạn chế ≈ 400", bản gốc ngược lại). Ma trận Phụ lục 4 (ô gộp) cũng bị chép lệch ô (lộ ra ở ad10).
- **Agent:** `search_report` (hybrid dense+BM25 RRF trên text + visual) · `view_page` (gửi ảnh gốc 200 DPI của một trang cho vision model đọc
  đúng phần cần hỏi; kết quả là mục context trích dẫn được `ccdr:pNNN:page`) · `final_answer` với citation `ccdr:pNNN:tK` / `ccdr:pNNN:vK`
  (có nhãn Bảng/Hình + `box_2d`) / `ccdr:pNNN:page`. Controller + vision: `gemini-3.1-flash-lite` (free tier).
- **Benchmark** ([advanced/benchmark/benchmark.json](../../advanced/benchmark/benchmark.json), sha256 `917d1e11…`): soạn bằng cách **xem ảnh trang gốc**
  trước khi chạy index/agent. Dev 18 (bảng 3, ảnh 2, biểu đồ 4, multi-hop 3, text 3, thiếu evidence 3); test **30** (bảng 6, ảnh/sơ đồ 5, biểu đồ 5,
  multi-hop 4, text 4, thiếu evidence 6). Dev/test không chung trang evidence. Evidence ghi trang PDF + trang in + phần tử + vùng (hàng/cột, series,
  vùng bản đồ), đơn vị, phép tính, tiêu chí `accept`; câu thiếu evidence ghi cách đã kiểm tra trong 104 trang (vd. "giá carbon" bị loại khỏi nhóm
  thiếu evidence vì tr. 49 có 12–120 USD/tCO2e).
- **Thử nghiệm trên dev** (chấm đạt/không đạt, [advanced/results/dev/*/labels.json](../../advanced/results/dev/)):

| Cấu hình | Thay đổi | Đạt dev | Bảng | Ảnh | Biểu đồ | Multi-hop | Text | Thiếu ev. | view_page | Chi phí ước tính |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_text (baseline) | chỉ chunk text layer | 12/18 (67%) | 2/3 | 2/2 | 1/4 | 1/3 | 3/3 | 3/3 | 0 | $0,097 |
| B1_visual | + chunk trực quan | 15/18 (83%) | 2/3 | 2/2 | 3/4 | 2/3 | 3/3 | 3/3 | 0 | $0,061 |
| **B2_visual_view** | + `view_page` ảnh gốc | **17/18 (94%)** | 3/3 | 2/2 | 4/4 | 2/3 | 3/3 | 3/3 | 12 | $0,063 |
| B4_view_prefer | B2 + "tin view_page khi mâu thuẫn" | 16/18 (89%) | 3/3 | 2/2 | 4/4 | 1/3 | 3/3 | 3/3 | 19 | $0,078 |

  Retrieval mức trang trên dev (không LLM, [advanced/results/dev/retrieval_summary.json](../../advanced/results/dev/retrieval_summary.json)): Hit@5 15/15
  cho mọi phương pháp/phạm vi; hybrid Hit@1 13/15 (text) → 14/15 (text+visual) — **tìm đúng trang dễ, đọc đúng giá trị mới khó**.
  Lỗi dev: B0 hụt các biểu đồ raster/nhãn chỉ có trên ảnh (ad05, ad07, ad11) và ma trận ô gộp (ad03, ad10); B1 sửa biểu đồ nhưng thừa hưởng lỗi trích
  xuất (Hình 16, Phụ lục 4); B2 sửa bằng đọc lại ảnh gốc, còn ad10: `view_page` đọc **đúng** ô nhưng controller theo chunk trích xuất sai. B4 thử ưu tiên
  `view_page` → ad10 vẫn sai (lần này vision đọc nhầm) và thêm lỗi ad11 (đọc 124 TWh của Khí thay vì 26 TWh của Than): vision không ổn định giữa
  các lần nên ưu tiên tuyệt đối không giúp. **Chốt B2_visual_view** ([selection_advanced.json](../../advanced/results/selection_advanced.json));
  lưu ý: tiêu chí chọn ghi sau khi chạy B0–B2 (không ghi trước như phần cơ bản), chọn theo tỷ lệ đạt dev cao nhất.

- **Test** (B2_visual_view, chạy một lần trên toàn bộ 30 câu; nhãn từng câu kèm lý do: [advanced/results/test/B2_visual_view/labels.json](../../advanced/results/test/B2_visual_view/labels.json)):

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | Bảng 1–4, Phụ lục 1 (2 trang): đọc đúng hàng/cột, không nhầm 844 với 677 |
| Ảnh, sơ đồ | 3/5 | 60% | at07: chỉ nêu duyên hải miền Trung, thiếu ven biển miền Bắc, mô tả sai màu; at09: bản đồ raster Hình 11 đọc sai (TP.HCM/Cần Thơ thay vì Đà Nẵng) |
| Biểu đồ | 5/5 | 100% | at13 **giáp ranh**: đọc ≈ −60 (ảnh ≈ −72), trong khoảng chấp nhận −60…−75 đã chốt trước |
| Multi-hop | 3/4 | 75% | at17: "chuyển sang dương khoảng 2010" (đúng ≈ 2001) — không gọi view_page, tin mô tả trích xuất của Hình 4 |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 6/6 | 100% | Nêu rõ báo cáo không có thông tin (5/6 `status=answered` nhưng nội dung từ chối — tiêu chí ADVANCED.md tính theo nội dung) |
| **Toàn bộ** | **27/30** | **90,0%** | 0 lỗi hệ thống/timeout; 17 lần view_page; trung vị 85 s/câu (free tier, hàng đợi); chi phí ước tính $0,093 (thực trả $0) |

- **Đạt ngưỡng ≥ 90% đúng ở mức tối thiểu (27/30)** và phụ thuộc câu giáp ranh at13: nếu người chấm không chấp nhận ≈ −60 (dù nằm trong khoảng ghi
  trước), kết quả là 26/30 = 86,7% < 90%. Đề nghị giảng viên/học viên xem lại at07, at09, at13, at17 trong `labels.json`.
- Điểm yếu còn lại tập trung ở bản đồ raster độ phân giải thấp (Hình 9, 11) và biểu đồ vùng chồng (Hình 4): vision Flash-Lite đọc màu/vị trí kém, và kết quả
  đọc không ổn định giữa các lần chạy (đã thấy ở dev B2 vs B4).
- **So sánh baseline trên test** (chạy sau B2, không dùng để chỉnh): B0_text (chỉ text layer) đạt **23/30 = 76,7%** — bảng 6/6, ảnh 2/5, biểu đồ 2/5,
  multi-hop 3/4, text 4/4, thiếu evidence 6/6. Pipeline trực quan + view_page thêm +4 câu, toàn bộ ở biểu đồ/ảnh (at08, at13, at14, at16) — đúng giả thuyết
  dev. Retrieval mức trang trên test (không LLM): hybrid Hit@1 16/24 (chỉ text) → 21/24 (text + chunk trực quan), Hit@5 24/24 cả hai.

## Giới hạn và ghi chú trung thực

- Corpus nhỏ (178 chunk) và benchmark nhỏ: test 13 câu có đáp án — mỗi câu ≈ 7,7%. Retrieval-only trên test đạt Hit@3 13/13 với mọi phương pháp
  không LLM: test dễ hơn dev, không đủ phân biệt phương pháp; kết luận so sánh chủ yếu dựa trên dev.
- 5 câu test Day 6 đã được chạy dense một lần ở Day 6 (không chỉnh theo); câu hỏi mới do Claude soạn từ text đã chuẩn hóa nên có thể gần cách
  diễn đạt văn bản.
- Nhãn chấm do **Claude (trợ lý AI) chấm** sau khi đọc trace + từng chunk được trích, có lý do từng câu; học viên cần rà soát. Không chấm mù.
- Thời gian đo trên API free tier, có hàng đợi/429: trung vị dao động mạnh giữa các lần chạy (A1 4,3 s, A0 57 s) — không dùng để kết luận tốc độ phương pháp.
- Chi phí là **ước tính** theo bảng giá trả phí × token thật; chi phí đã trả thực tế $0 (free tier).
- **Công cụ AI:** Claude (Anthropic) viết code, soạn câu hỏi, chấm nhãn và viết báo cáo; Gemini là LLM của agent. Kiểm tra bằng script xác minh
  evidence, unit test (25 test, mock), đọc lại trace/chunk, đối chiếu ảnh trang gốc (phần nâng cao).
