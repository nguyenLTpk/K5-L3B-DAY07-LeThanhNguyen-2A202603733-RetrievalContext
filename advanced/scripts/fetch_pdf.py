# -*- coding: utf-8 -*-
"""Tải bản PDF tiếng Việt của báo cáo CCDR Việt Nam (Ngân hàng Thế giới, 7/2022) và kiểm tra sha256.

Nguồn: https://www.ifc.org/content/dam/ifc/doc/mgrt/ccdr-full-report-vie-1107.pdf
Bản quyền © 2022 IBRD/World Bank — trang 2 cho phép sao chép toàn bộ/một phần cho mục đích phi thương mại
khi ghi nhận đầy đủ tác phẩm. Bản tiếng Anh trên Open Knowledge Repository: http://hdl.handle.net/10986/37618
(CC BY 3.0 IGO). Repo không lưu PDF; chạy script này để tái tạo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests

URL = "https://www.ifc.org/content/dam/ifc/doc/mgrt/ccdr-full-report-vie-1107.pdf"
SHA256 = "cb0c7f23a88ceadcbb5ad837e31caa73a7cc201db4140aed45820dcbcf666a18"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ccdr_vietnam_2022_vi.pdf"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        r = requests.get(URL, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        OUT.write_bytes(r.content)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 không khớp ({digest}); bản PDF có thể đã thay đổi — benchmark gắn với phiên bản {SHA256[:12]}")
    print("OK", OUT, digest)


if __name__ == "__main__":
    main()
