# -*- coding: utf-8 -*-
"""Chuẩn hóa text dùng chung cho chấm hit và BM25."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+", re.UNICODE)


def norm(s: str) -> str:
    """NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng (giữ dấu tiếng Việt)."""
    s = unicodedata.normalize("NFC", s).casefold()
    return " ".join(_WORD.findall(s))


def tokens(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", s).casefold())


def contains(chunk_text: str, evidence: str) -> bool:
    return norm(evidence) in norm(chunk_text)
