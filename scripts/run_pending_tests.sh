#!/usr/bin/env bash
# Chạy các bước TEST còn lại (cấu hình đã chốt trên dev — results/selection_basic.json, advanced/results/selection_advanced.json).
# Cần quota Gemini free tier đầy (reset 0h giờ Thái Bình Dương = 15:00 giờ VN). Mỗi lệnh chạy MỘT lần.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

# 1. Phần cơ bản — retrieval-only các phương pháp dùng LLM trên test (gemini-3.5-flash-lite)
py -3.11 scripts/eval_retrieval.py --split test --methods rerank multiquery

# 2. Phần cơ bản — agent baseline + cấu hình đã chọn trên test
py -3.11 scripts/run_agent.py --split test --configs A0_dense A3_hybrid_strict --workers 2

# 3. Nâng cao — cấu hình đã chọn trên toàn bộ 30 câu test (gemini-3.1-flash-lite)
py -3.11 -X utf8 advanced/scripts/run_agent.py --split test --configs B2_visual_view --workers 2

# 4. (tùy quota) Nâng cao — baseline text-only trên test để so sánh
py -3.11 -X utf8 advanced/scripts/run_agent.py --split test --configs B0_text --workers 2 || true

echo "Xong. Tiếp theo: chấm nhãn (scripts/write_reviews.py --split test, advanced/scripts/write_labels.py --split test)."
