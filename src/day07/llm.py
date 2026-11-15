# -*- coding: utf-8 -*-
"""Client Gemini REST (generateContent) có function calling, retry và đếm token.

- Không dùng SDK để giảm phụ thuộc; API key lấy từ .env, gửi qua header x-goog-api-key.
- Retry có backoff cho 429/500/503/504 và lỗi mạng; lỗi khác ném ra ngoài (không giả kết quả).
- Mọi lần gọi được cộng vào `usage` (prompt/output/thinking token, số lần gọi, thời gian).
"""
from __future__ import annotations

import base64
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import LLM_MODEL, gemini_key

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, meta: dict[str, Any], seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += int(meta.get("promptTokenCount", 0) or 0)
        self.output_tokens += int(meta.get("candidatesTokenCount", 0) or 0)
        self.thinking_tokens += int(meta.get("thoughtsTokenCount", 0) or 0)
        self.cached_tokens += int(meta.get("cachedContentTokenCount", 0) or 0)
        self.seconds += seconds

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "thinking_tokens": self.thinking_tokens, "cached_tokens": self.cached_tokens,
                "llm_seconds": round(self.seconds, 2), "retries": self.retries, "errors": list(self.errors)}

    def merge(self, other: "Usage") -> None:
        for k in ("calls", "prompt_tokens", "output_tokens", "thinking_tokens", "cached_tokens", "retries"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.seconds += other.seconds
        self.errors.extend(other.errors)


class Gemini:
    def __init__(self, model: str = LLM_MODEL, thinking_level: str | None = "low",
                 timeout: int = 180, max_retries: int = 6, min_interval: float = 0.0) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._key = gemini_key()
        self._lock = threading.Lock()
        self._last = 0.0
        self.session = requests.Session()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def generate(self, contents: list[dict[str, Any]], *, system: str | None = None,
                 tools: list[dict[str, Any]] | None = None, tool_mode: str | None = None,
                 allowed: list[str] | None = None, json_schema: dict[str, Any] | None = None,
                 usage: Usage | None = None, max_output_tokens: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
            if tool_mode:
                cfg: dict[str, Any] = {"mode": tool_mode}
                if allowed:
                    cfg["allowedFunctionNames"] = allowed
                body["toolConfig"] = {"functionCallingConfig": cfg}
        gen: dict[str, Any] = {}
        if self.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        if json_schema is not None:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = json_schema
        if max_output_tokens:
            gen["maxOutputTokens"] = max_output_tokens
        if gen:
            body["generationConfig"] = gen

        url = API.format(model=self.model)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            t0 = time.perf_counter()
            try:
                r = self.session.post(url, headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
                                      data=json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"network: {type(e).__name__}"
                if usage:
                    usage.retries += 1
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            dt = time.perf_counter() - t0
            if r.status_code == 200:
                data = r.json()
                if usage is not None:
                    usage.add(data.get("usageMetadata", {}), dt)
                if not data.get("candidates"):
                    raise LLMError(f"Không có candidate: {json.dumps(data.get('promptFeedback', data))[:300]}")
                return data
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 504):
                if usage:
                    usage.retries += 1
                delay = min(90, 4 * 2 ** attempt + random.random() * 2)
                try:  # tôn trọng RetryInfo nếu có
                    for d in r.json().get("error", {}).get("details", []):
                        if d.get("@type", "").endswith("RetryInfo") and d.get("retryDelay", "").endswith("s"):
                            delay = max(delay, float(d["retryDelay"][:-1]) + 1)
                except Exception:
                    pass
                if delay > 300:  # hết quota theo ngày (free tier): báo lỗi ngay, không ngủ nhiều giờ
                    last_err = f"QUOTA_EXHAUSTED (retry sau {int(delay)}s): {r.text[:200]}"
                    break
                time.sleep(delay)
                continue
            break
        if usage is not None:
            usage.errors.append(last_err[:200])
        raise LLMError(last_err)

    # ---- tiện ích ----
    @staticmethod
    def parts(resp: dict[str, Any]) -> list[dict[str, Any]]:
        return resp["candidates"][0].get("content", {}).get("parts", []) or []

    @staticmethod
    def text(resp: dict[str, Any]) -> str:
        return "".join(p.get("text", "") for p in Gemini.parts(resp) if not p.get("thought"))

    @staticmethod
    def function_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
        return [p["functionCall"] for p in Gemini.parts(resp) if "functionCall" in p]

    def json_call(self, prompt: str, schema: dict[str, Any], *, system: str | None = None,
                  images: list[bytes] | None = None, usage: Usage | None = None) -> Any:
        parts: list[dict[str, Any]] = []
        for img in images or []:
            parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(img).decode()}})
        parts.append({"text": prompt})
        resp = self.generate([{"role": "user", "parts": parts}], system=system, json_schema=schema, usage=usage)
        txt = self.text(resp)
        try:
            return json.loads(txt)
        except json.JSONDecodeError as e:
            raise LLMError(f"JSON không hợp lệ từ model: {txt[:200]}") from e
