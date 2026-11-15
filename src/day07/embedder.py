# -*- coding: utf-8 -*-
"""Query embedding tương thích index Day 6: intfloat/multilingual-e5-small (ONNX, 384 chiều).

Giữ nguyên cách làm của Day 6 (src/embeddings.py): tokenizer.json của model, tối đa 512 token,
mean pooling theo attention mask, chuẩn hóa L2, tiền tố "query: " cho câu hỏi.
Model được đọc từ repo Day 6 (không copy 490 MB); sai checksum -> dừng, không thay model khác.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import E5_MODEL_DIR

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
ONNX_SHA256 = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
DIMENSIONS = 384


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class E5Embedder:
    def __init__(self, model_dir: Path = E5_MODEL_DIR, max_length: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path, tok_path = model_dir / "onnx_model.onnx", model_dir / "onnx_tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            raise FileNotFoundError(f"Thiếu model e5-small tại {model_dir} (đặt DAY6_DIR hoặc E5_MODEL_DIR)")
        if _sha256(onnx_path) != ONNX_SHA256:
            raise ValueError("Checksum ONNX không khớp revision Day 6 — query embedding sẽ không tương thích index")
        self.model_name = f"{MODEL_ID}@{MODEL_REVISION[:8]}"
        self.dimensions = DIMENSIONS
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = self.session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask,
                                                           "token_type_ids": np.zeros_like(ids)})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(np.float32)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([f"query: {question}"])[0].tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Dùng cho index thử thách nâng cao (tiền tố 'passage: ' như Day 6)."""
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._encode([f"passage: {t}" for t in texts[i:i + batch_size]]).tolist())
        return out

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(f"passage: {text}").ids)


@lru_cache(maxsize=1)
def get_embedder() -> E5Embedder:
    return E5Embedder()
