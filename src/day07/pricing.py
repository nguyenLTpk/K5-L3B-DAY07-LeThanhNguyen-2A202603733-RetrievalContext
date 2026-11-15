# -*- coding: utf-8 -*-
"""Ước tính chi phí theo bảng giá Gemini API (standard tier, USD / 1M token), tra ngày 2026-11-15:
https://ai.google.dev/gemini-api/docs/pricing

- gemini-3.5-flash-lite: input 0.30, output 2.50 (output gồm thinking token).
- gemini-3.5-flash (được route sang gemini-3.6-flash): dùng giá niêm yết 1.50 / 7.50
  (trang pricing ghi giá khuyến mãi 0.75 / 3.75 tới 2026-12-31; ta lấy mức cao để không báo thấp chi phí).
Đây là ước tính theo token đếm từ usageMetadata NẾU trả phí. Key dùng trong lab thuộc Gemini API Free Tier
(xác nhận qua lỗi 429 'generate_content_free_tier_requests'), nên chi phí thực tế đã trả là $0.
"""
from __future__ import annotations

PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # không tìm được giá riêng của 3.1 Flash-Lite lúc chạy: tạm dùng giá 3.5 Flash-Lite (ước tính, có thể cao hơn thực tế)
    "gemini-3.1-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 7.50),
    "gemini-3.6-flash": (1.50, 7.50),
}


def cost_usd(model: str, usage) -> float:
    if usage is None:
        return 0.0
    pin, pout = PRICES.get(model, (1.50, 7.50))
    u = usage if isinstance(usage, dict) else usage.as_dict()
    return (u["prompt_tokens"] * pin + (u["output_tokens"] + u["thinking_tokens"]) * pout) / 1e6
