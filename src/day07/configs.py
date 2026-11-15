# -*- coding: utf-8 -*-
"""Cấu hình agent (chốt trước khi chạy dev; xem docs/EXPERIMENTS.md cho giả thuyết).

Cùng corpus, index Day 6, benchmark, system prompt và LLM điều phối cho mọi cấu hình; chỉ đổi
phần ghi trong cột 'thay đổi'.
"""
from __future__ import annotations

AGENT_CONFIGS = {
    # Baseline: dense retrieval trên vector store Day 6, top-3, như khung starter.
    "A0_dense": {"retriever": "dense", "top_k": 3, "max_tool_calls": 4, "budget_chars": 4000, "allow_adjacent": False,
                 "llm": "gemini-3.5-flash-lite",
                 "change": "baseline",
                 "hypothesis": "Dense top-3 đủ cho câu trực tiếp; dự đoán hụt ở câu diễn đạt lại (q14) và câu filter khi không có filter."},
    # Cải thiện 1: đổi tool sang hybrid RRF, top-4, nhiều lượt và ngân sách lớn hơn.
    "A1_hybrid": {"retriever": "hybrid", "top_k": 4, "max_tool_calls": 6, "budget_chars": 6000, "allow_adjacent": False,
                  "llm": "gemini-3.5-flash-lite",
                  "change": "tool hybrid (dense+BM25, RRF) + top_k 4 + 6 lượt + budget 6000",
                  "hypothesis": "BM25 bắt thuật ngữ/con số mà dense bỏ lỡ; dev retrieval hybrid Hit@3 16/17 > dense 15/17. Dự đoán tăng Context Evidence Rate và grounded answer, chi phí token tăng nhẹ."},
    # Cải thiện 2: hybrid + rerank bằng LLM + tool đọc chunk liền kề.
    "A2_rerank_adj": {"retriever": "rerank", "top_k": 4, "max_tool_calls": 6, "budget_chars": 6000, "allow_adjacent": True,
                      "llm": "gemini-3.5-flash-lite",
                      "change": "A1 + rerank LLM trên 15 candidate + tool read_adjacent",
                      "hypothesis": "Rerank đưa đoạn trả lời trực tiếp lên đầu (dev retrieval Hit@1 17/17); read_adjacent sửa lỗi Điều bị tách chunk (q14 Day 6). Dự đoán tăng grounded answer ở câu điều kiện/multi-hop, đổi lại thêm ~1 lời gọi LLM mỗi lần search."},
    # Cải thiện 3 (thêm sau khi đọc lỗi dev d24/d25): A1 + model tự khai báo tài liệu có nêu trực tiếp thông tin không.
    "A3_hybrid_strict": {"retriever": "hybrid", "top_k": 4, "max_tool_calls": 6, "budget_chars": 6000, "allow_adjacent": False,
                         "strict_abstain": True, "llm": "gemini-3.5-flash-lite",
                         "change": "A1 + trường bắt buộc tai_lieu_neu_truc_tiep trong final_answer (false -> abstain)",
                         "hypothesis": "Lỗi dev d24/d25: model viết 'không quy định' nhưng status=answered. Bắt model khai báo rõ sẽ đưa status abstain về 4/4, không tăng từ chối nhầm câu có đáp án."},
}


def selection_criteria() -> list[str]:
    return [
        "1. Grounded answer trung bình /2 trên câu có đáp án của dev (cao hơn tốt hơn).",
        "2. Abstention đúng trên câu thiếu evidence của dev, rồi số câu có đáp án bị từ chối nhầm (ít hơn tốt hơn).",
        "3. Context Evidence Rate trên dev.",
        "4. Chi phí và thời gian trung vị mỗi câu (thấp hơn tốt hơn).",
        "Chênh lệch <= 1 câu ở một tiêu chí coi là hòa và xét tiêu chí kế tiếp.",
    ]
