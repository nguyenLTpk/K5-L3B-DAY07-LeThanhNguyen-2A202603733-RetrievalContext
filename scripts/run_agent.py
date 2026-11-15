# -*- coding: utf-8 -*-
"""Chạy agent thật (Gemini + retriever trên index Day 6) cho một split và một/nhiều cấu hình.

Chạy:  python scripts/run_agent.py --split dev --configs A0_dense A1_hybrid A2_rerank_adj [--workers 4]
Output: results/agent/<split>/<config>/traces.json, summary_auto.json

summary_auto chỉ gồm số đo tự động (tool use, số lượt gọi, trạng thái, Context Evidence Rate,
Hit của lần search đầu, thời gian, token, chi phí). Grounded answer / citation hỗ trợ / abstention
được chấm riêng (scripts/judge_answers.py + rà soát thủ công) trong labels.json.
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
sys.path.insert(0, str(ROOT / "src"))

from day07 import retrievers as R  # noqa: E402
from day07.agent import Agent  # noqa: E402
from day07.config import DATA, RESULTS  # noqa: E402
from day07.configs import AGENT_CONFIGS  # noqa: E402
from day07.llm import Gemini  # noqa: E402
from day07.metrics import context_full, ranking_metrics  # noqa: E402


def make_agent(name: str) -> Agent:
    cfg = AGENT_CONFIGS[name]
    llm = Gemini(cfg["llm"], thinking_level=None)
    ret = R.build(cfg["retriever"], llm=Gemini("gemini-3.5-flash-lite", thinking_level=None))
    return Agent(llm, ret, name=name, top_k=cfg["top_k"], max_tool_calls=cfg["max_tool_calls"],
                 budget_chars=cfg["budget_chars"], allow_adjacent=cfg["allow_adjacent"],
                 strict_abstain=cfg.get("strict_abstain", False))


def auto_summary(traces: list[dict], qs: dict[str, dict], cfg_name: str, split: str) -> dict:
    ok = [t for t in traces if t["status"] != "error"]
    ans_q = [t for t in ok if qs[t["query_id"]]["answerable"]]
    una_q = [t for t in ok if not qs[t["query_id"]]["answerable"]]
    by_id = R.chunk_by_id()
    ctx_full = first_h1 = first_h3 = 0
    for t in ans_q:
        q = qs[t["query_id"]]
        if context_full([by_id[i] for i in t["context_ids"]], q):
            ctx_full += 1
        first = next((e for e in t["events"] if e["event"] == "search" and e.get("results")), None)
        if first:
            m = ranking_metrics([by_id[r["id"]] for r in first["results"]], q)
            first_h1 += m["hit1"]
            first_h3 += m["hit3"]
    n_ans = len(ans_q)
    return {
        "config": cfg_name, "split": split, "n": len(traces), "errors": len(traces) - len(ok),
        "status_counts": dict(Counter(t["status"] for t in traces)),
        "tool_use": f"{sum(t['n_search'] > 0 for t in ok)}/{len(ok)}",
        "search_calls_distribution": dict(sorted(Counter(t["n_search"] for t in ok).items())),
        "tool_calls_mean": round(statistics.mean(t["n_tool_calls"] for t in ok), 2) if ok else None,
        "context_evidence_rate": f"{ctx_full}/{n_ans}",
        "first_search_hit1": f"{first_h1}/{n_ans}", "first_search_hit3": f"{first_h3}/{n_ans}",
        "answered_on_answerable": f"{sum(t['status'] == 'answered' for t in ans_q)}/{n_ans}",
        "abstain_on_unanswerable": f"{sum(t['status'] == 'abstain' for t in una_q)}/{len(una_q)}",
        "answers_with_citation": f"{sum(bool(t['citations']) for t in ok if t['status'] == 'answered')}/{sum(t['status'] == 'answered' for t in ok)}",
        "context_chars_mean": round(statistics.mean(t["context_chars"] for t in ok), 1) if ok else None,
        "median_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else None,
        "total_s": round(sum(t["elapsed_s"] for t in traces), 1),
        "llm_calls": sum(t["usage"]["calls"] + t["retriever_llm_usage"]["calls"] for t in traces),
        "prompt_tokens": sum(t["usage"]["prompt_tokens"] + t["retriever_llm_usage"]["prompt_tokens"] for t in traces),
        "output_tokens": sum(t["usage"]["output_tokens"] + t["usage"]["thinking_tokens"] + t["retriever_llm_usage"]["output_tokens"] for t in traces),
        "cost_usd": round(sum(t["cost_usd"] for t in traces), 5),
        "cost_usd_per_query": round(sum(t["cost_usd"] for t in traces) / max(1, len(traces)), 6),
        "error_messages": [t["error"] for t in traces if t["error"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="chỉ chạy các query_id này (debug)")
    args = ap.parse_args()

    all_q = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
    qs = [q for q in all_q if q["split"] == args.split and (not args.only or q["id"] in args.only)]
    qmap = {q["id"]: q for q in qs}
    for name in args.configs:
        local = threading.local()

        def work(q):
            if not hasattr(local, "agent"):
                local.agent = make_agent(name)
            # chỉ gửi câu hỏi + filter người dùng; KHÔNG gửi gold/expected_answer
            return local.agent.run(q["question"], filters=q.get("filter"), query_id=q["id"])

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            traces = list(ex.map(work, qs))
        wall = time.perf_counter() - t0
        out = RESULTS / "agent" / args.split / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        s = auto_summary(traces, qmap, name, args.split)
        s["config_detail"] = AGENT_CONFIGS[name]
        s["wall_s"] = round(wall, 1)
        if not args.only:
            (out / "summary_auto.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: s[k] for k in ("config", "errors", "status_counts", "context_evidence_rate", "first_search_hit3",
                                              "answered_on_answerable", "abstain_on_unanswerable", "median_s", "cost_usd")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
