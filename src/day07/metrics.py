# -*- coding: utf-8 -*-
"""Metric retrieval/context. Định nghĩa hit giống Day 6:
chunk hit một nhóm evidence khi đúng source_id + đúng page VÀ chứa trọn một evidence của nhóm
(chuỗi con sau chuẩn hóa chữ thường/bỏ dấu câu). Gold chỉ dùng ở đây, không vào agent/retriever.

- Hit@k: có >= 1 nhóm evidence được phủ trong top-k (câu 1 nhóm: như Day 6).
- Full@k: TẤT CẢ nhóm evidence được phủ trong top-k (khác Hit@k chỉ ở câu multi-hop).
- MRR@10: 1/rank của chunk hit đầu tiên.
- Context Evidence Rate: context cuối gửi model phủ TẤT CẢ nhóm evidence.
Câu thiếu evidence không vào mẫu số.
"""
from __future__ import annotations

from typing import Any

from .textnorm import contains


def groups_hit(chunk: dict[str, Any], q: dict[str, Any]) -> set[int]:
    out = set()
    for gi, grp in enumerate(q.get("gold_groups") or []):
        for g in grp:
            if chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and contains(chunk["text"], g["evidence"]):
                out.add(gi)
                break
    return out


def covered(chunks: list[dict[str, Any]], q: dict[str, Any]) -> set[int]:
    s: set[int] = set()
    for c in chunks:
        s |= groups_hit(c, q)
    return s


def ranking_metrics(results: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    n_groups = len(q["gold_groups"])
    per = [sorted(groups_hit(c, q)) for c in results]
    first = next((i + 1 for i, g in enumerate(per) if g), None)

    def at(k: int) -> set[int]:
        s: set[int] = set()
        for g in per[:k]:
            s |= set(g)
        return s
    return {
        "hit1": bool(at(1)), "hit3": bool(at(3)), "hit5": bool(at(5)),
        "full3": len(at(3)) == n_groups, "full5": len(at(5)) == n_groups, "full10": len(at(10)) == n_groups,
        "rr10": (1.0 / first) if first and first <= 10 else 0.0, "first_hit_rank": first, "groups_per_result": per,
    }


def context_full(context_chunks: list[dict[str, Any]], q: dict[str, Any]) -> bool:
    return len(covered(context_chunks, q)) == len(q["gold_groups"])
