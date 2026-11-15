# -*- coding: utf-8 -*-
"""Chạy agent thử thách nâng cao trên benchmark CCDR (dev hoặc test).

Chạy: python -I advanced/scripts/run_agent.py --split dev --configs B0_text B1_visual B2_visual_view
Output: advanced/results/<split>/<config>/traces.json + summary_auto.json
Chỉ gửi câu hỏi cho agent; đáp án/evidence gold chỉ dùng ở bước đo page-hit (tự động) và chấm (thủ công).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

from day07 import retrievers as R  # noqa: E402
from day07.adv_agent import CCDR, AdvAgent  # noqa: E402
from day07.llm import Gemini  # noqa: E402

CONFIGS = {
    "B0_text": {"retriever": "hybrid", "scope_lock": "text", "view": False, "llm": "gemini-3.1-flash-lite",
                "hypothesis": "Baseline chỉ text layer PDF: trả lời được câu text và bảng/biểu đồ có nhãn số dạng vector; hụt ở ảnh raster, bản đồ, màu chú giải."},
    "B1_visual": {"retriever": "hybrid", "scope_lock": None, "view": False, "llm": "gemini-3.1-flash-lite",
                  "hypothesis": "Thêm chunk trực quan (vision mô tả/chép bảng-biểu đồ lúc index) -> tăng câu bảng/biểu đồ/ảnh."},
    "B2_visual_view": {"retriever": "hybrid", "scope_lock": None, "view": True, "llm": "gemini-3.1-flash-lite",
                       "vision": "gemini-3.1-flash-lite",
                       "hypothesis": "Thêm view_page đọc ảnh gốc lúc trả lời -> sửa lỗi trích xuất lúc index, tăng độ chính xác số liệu biểu đồ; tốn thêm lời gọi vision."},
    "B3_rerank_view": {"retriever": "rerank", "scope_lock": None, "view": True, "llm": "gemini-3.1-flash-lite", "vision": "gemini-3.1-flash-lite",
                       "hypothesis": "B2 + rerank LLM trên 15 candidate -> đưa đúng Bảng/Hình lên đầu khi nhiều trang cùng chủ đề; thêm 1 lời gọi LLM mỗi lần search."},
    # Thêm sau khi đọc lỗi dev ad10 của B2: view_page đọc đúng nhưng controller tin chunk trích xuất sai.
    "B4_view_prefer": {"retriever": "hybrid", "scope_lock": None, "view": True, "prefer_view": True, "llm": "gemini-3.1-flash-lite",
                       "vision": "gemini-3.1-flash-lite",
                       "hypothesis": "B2 + quy tắc: khi view_page (ảnh gốc) khác chunk trích xuất lúc index thì tin view_page -> sửa ad10, không làm sai câu khác."},
}
# Ghi chú: key dùng Gemini API Free Tier; gemini-3.5-flash chỉ 20 request/ngày nên không dùng cho agent/vision.


def make(name: str) -> AdvAgent:
    c = CONFIGS[name]
    llm = Gemini(c["llm"], thinking_level="low" if "lite" not in c["llm"] else None, timeout=240)
    ret = R.build(c["retriever"], llm=Gemini("gemini-3.1-flash-lite", thinking_level=None), corpus=CCDR)
    vision = (Gemini(c["vision"], thinking_level=None if "lite" in c["vision"] else "low", timeout=240)
              if c.get("view") else None)
    return AdvAgent(llm, ret, name=name, vision=vision, allow_view_page=c.get("view", False), scope_lock=c.get("scope_lock"),
                    prefer_view=c.get("prefer_view", False))


def page_hits(t: dict, q: dict) -> dict:
    """Tự động, mức trang: nhóm evidence nào có ít nhất một mục context nằm trên trang gold."""
    if not q["answerable"]:
        return {}
    ctx_pages = {int(cid.split(":")[1][1:]) for cid in t["context_ids"]}
    cit_pages = {int(cid.split(":")[1][1:]) for cid in t["citations"]}
    groups = [{a["page"] for a in g["alternatives"]} for g in q["evidence"]]
    first = next((e for e in t["events"] if e["event"] == "search" and e.get("results")), None)
    first_pages = [r["page"] for r in first["results"]] if first else []
    return {"context_all_groups": all(g & ctx_pages for g in groups),
            "citation_all_groups": all(g & cit_pages for g in groups),
            "first_search_hit1": bool(first_pages[:1] and any(first_pages[0] in g for g in groups)),
            "first_search_hit3": any(p in g for p in first_pages[:3] for g in groups)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    qs = [q for q in json.loads((ROOT / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
          if q["split"] == args.split and (not args.only or q["id"] in args.only)]
    for name in args.configs:
        local = threading.local()

        def work(q):
            if not hasattr(local, "agent"):
                local.agent = make(name)
            return local.agent.run(q["question"], query_id=q["id"])

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            traces = list(ex.map(work, qs))
        qmap = {q["id"]: q for q in qs}
        for t in traces:
            t["group"] = qmap[t["query_id"]]["group"]
            t["auto_page_hits"] = page_hits(t, qmap[t["query_id"]])
        out = ROOT / "results" / args.split / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = [t for t in traces if t["status"] != "error"]
        ans = [t for t in ok if qmap[t["query_id"]]["answerable"]]
        s = {"config": name, "detail": CONFIGS[name], "split": args.split, "n": len(traces), "errors": len(traces) - len(ok),
             "status_counts": dict(Counter(t["status"] for t in traces)),
             "context_page_coverage_all_groups": f"{sum(t['auto_page_hits']['context_all_groups'] for t in ans)}/{len(ans)}",
             "citation_page_coverage_all_groups": f"{sum(t['auto_page_hits']['citation_all_groups'] for t in ans)}/{len(ans)}",
             "first_search_page_hit1": f"{sum(t['auto_page_hits']['first_search_hit1'] for t in ans)}/{len(ans)}",
             "first_search_page_hit3": f"{sum(t['auto_page_hits']['first_search_hit3'] for t in ans)}/{len(ans)}",
             "abstain_on_unanswerable": f"{sum(t['status'] == 'abstain' for t in ok if not qmap[t['query_id']]['answerable'])}/{sum(not qmap[t['query_id']]['answerable'] for t in ok)}",
             "abstain_on_answerable": f"{sum(t['status'] == 'abstain' for t in ans)}/{len(ans)}",
             "search_calls_distribution": dict(sorted(Counter(t["n_search"] for t in ok).items())),
             "view_page_calls": sum(t["n_view_page"] for t in ok),
             "median_s": round(statistics.median(t["elapsed_s"] for t in ok), 1) if ok else None,
             "wall_s": round(time.perf_counter() - t0, 1),
             "cost_usd": round(sum(t["cost_usd"] for t in traces), 4),
             "errors_detail": [t["error"] for t in traces if t["error"]]}
        if not args.only:
            (out / "summary_auto.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: s[k] for k in ("config", "errors", "status_counts", "context_page_coverage_all_groups",
                                              "first_search_page_hit3", "abstain_on_unanswerable", "abstain_on_answerable",
                                              "view_page_calls", "median_s", "cost_usd")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
