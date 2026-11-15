# -*- coding: utf-8 -*-
"""Retrieval-only cho benchmark nâng cao, đo ở MỨC TRANG (evidence ghi theo trang + phần tử, không theo chunk).

Hit@k: có >= 1 nhóm evidence có trang gold xuất hiện trong top-k. Full@k: mọi nhóm đều có trang trong top-k.
Hai phạm vi index: text (chỉ chunk text layer) và any (text + chunk trực quan). Không dùng LLM.
Chạy: python -X utf8 advanced/scripts/eval_retrieval.py --split dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
from day07 import retrievers as R  # noqa: E402
from day07.adv_agent import CCDR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    args = ap.parse_args()
    qs = [q for q in json.loads((ROOT / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
          if q["split"] == args.split and q["answerable"]]
    dense, bm25 = R.DenseRetriever(CCDR), R.BM25Retriever(CCDR)
    hybrid = R.HybridRetriever(dense, bm25)
    rows, traces = {}, []
    for name, ret in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid)):
        for scope, flt in (("text", {"kind": "text"}), ("any", None)):
            key = f"{name}_{scope}"
            c = {"hit1": 0, "hit3": 0, "hit5": 0, "full5": 0, "full10": 0}
            by_group = {}
            for q in qs:
                res = ret.search(q["question"], filters=flt, top_k=10)
                pages = [r["page"] for r in res]
                groups = [{a["page"] for a in g["alternatives"]} for g in q["evidence"]]
                hit = lambda k: any(p in g for p in pages[:k] for g in groups)  # noqa: E731
                full = lambda k: all(any(p in g for p in pages[:k]) for g in groups)  # noqa: E731
                for m, v in (("hit1", hit(1)), ("hit3", hit(3)), ("hit5", hit(5)), ("full5", full(5)), ("full10", full(10))):
                    c[m] += v
                bg = by_group.setdefault(q["group"], [0, 0])
                bg[0] += hit(5)
                bg[1] += 1
                traces.append({"config": key, "query_id": q["id"], "group": q["group"], "top10": [
                    {"id": r["id"], "page": r["page"], "kind": r["metadata"].get("kind"), "label": r["metadata"].get("label"), "score": r["score"]}
                    for r in res]})
            n = len(qs)
            rows[key] = {m: f"{v}/{n}" for m, v in c.items()} | {"hit5_by_group": {g: f"{a}/{b}" for g, (a, b) in by_group.items()}}
            print(key, rows[key])
    out = ROOT / "results" / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "retrieval_traces.json").write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
