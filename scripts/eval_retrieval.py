# -*- coding: utf-8 -*-
"""So sánh các phương pháp retrieval (không agent) trên cùng corpus, index Day 6, câu hỏi và nhãn.

Chạy:  python scripts/eval_retrieval.py --split dev --methods dense bm25 hybrid rerank multiquery
Output: results/retrieval/<split>/<method>[_<tag>].json (trace từng câu, top-10) và summary.json.

- Query = câu hỏi gốc của benchmark. `filter` chỉ áp dụng khi câu có filter do người dùng chọn trước
  (giống Day 6 chiến lược C) và khi bật --use-filter; ghi rõ trong output.
- Luôn xuất top-10 để tính Hit@1, Evidence Hit@3, Full@3/5, MRR@10 với cùng định nghĩa cho mọi phương pháp.
- Câu thiếu evidence chạy để đo thời gian nhưng không vào mẫu số metric.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day07 import retrievers as R  # noqa: E402
from day07.config import DATA, RESULTS  # noqa: E402
from day07.llm import Gemini  # noqa: E402
from day07.metrics import ranking_metrics  # noqa: E402
from day07.pricing import cost_usd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("--use-filter", action="store_true", help="áp dụng filter người dùng cung cấp (nếu câu có)")
    ap.add_argument("--no-bigrams", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--llm-model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()

    qs = [q for q in json.loads((DATA / "benchmark.json").read_text(encoding="utf-8")) if q["split"] == args.split]
    outdir = RESULTS / "retrieval" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    llm = Gemini(args.llm_model, thinking_level=None)

    for method in args.methods:
        t_build = time.perf_counter()
        ret = R.build(method, llm=llm, bigrams=not args.no_bigrams)
        build_s = time.perf_counter() - t_build
        traces, times = [], []
        for q in qs:
            flt = q.get("filter") if args.use_filter else None
            t0 = time.perf_counter()
            res = ret.search(q["question"], filters=flt, top_k=10)
            dt = time.perf_counter() - t0
            times.append(dt)
            tr = {"query_id": q["id"], "question": q["question"], "type": q["type"], "answerable": q["answerable"],
                  "filter": flt, "elapsed_s": round(dt, 3),
                  "results": [{k: r.get(k) for k in ("rank", "id", "source_id", "page", "score", "component_ranks",
                                                     "rerank_score", "pre_rank")} | {"article": r["metadata"].get("articles_covered"),
                                                                                       "text": r["text"]} for r in res]}
            if getattr(ret, "last_queries", None):
                tr["rewritten_queries"] = list(ret.last_queries)
            if q["answerable"]:
                tr["metrics"] = ranking_metrics(res, q)
            traces.append(tr)
        ans = [t for t in traces if t["answerable"]]
        n = len(ans)

        def cnt(key):
            return sum(bool(t["metrics"][key]) for t in ans)
        usage = getattr(ret, "usage", None)
        key = method + (f"_{args.tag}" if args.tag else "") + ("_filter" if args.use_filter else "")
        s = {"method": method, "split": args.split, "filter": "user-provided" if args.use_filter else "none",
             "bm25_bigrams": not args.no_bigrams, "n_answerable": n,
             "hit1": f"{cnt('hit1')}/{n}", "hit3": f"{cnt('hit3')}/{n}", "hit5": f"{cnt('hit5')}/{n}",
             "full3": f"{cnt('full3')}/{n}", "full5": f"{cnt('full5')}/{n}", "full10": f"{cnt('full10')}/{n}",
             "mrr10": round(sum(t["metrics"]["rr10"] for t in ans) / n, 3),
             "median_query_s": round(statistics.median(times), 3), "total_query_s": round(sum(times), 2),
             "build_s": round(build_s, 2),
             "llm_usage": usage.as_dict() if usage else None,
             "cost_usd": round(cost_usd(args.llm_model, usage), 5) if usage else 0.0,
             "misses_hit3": [t["query_id"] for t in ans if not t["metrics"]["hit3"]]}
        summary[key] = s
        (outdir / f"{key}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: s[k] for k in ("method", "filter", "hit1", "hit3", "full3", "full5", "mrr10", "median_query_s", "cost_usd", "misses_hit3")}, ensure_ascii=False))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
