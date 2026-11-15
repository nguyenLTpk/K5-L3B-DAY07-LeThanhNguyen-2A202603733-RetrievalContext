# -*- coding: utf-8 -*-
"""In bảng markdown từ results/ (để báo cáo dùng đúng số trong file kết quả). Chạy: python scripts/make_tables.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(s: str) -> str:
    n, d = s.split("/")
    return f"{s} ({100 * int(n) / int(d):.0f}%)" if int(d) else s


def retrieval(split: str) -> None:
    s = load(RES / "retrieval" / split / "summary.json")
    if not s:
        return
    print(f"\n### Retrieval-only — {split}\n")
    print("| Phương pháp | Filter | Hit@1 | Evidence Hit@3 | Full@3 | Full@5 | MRR@10 | Trung vị/câu | Chi phí |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, r in s.items():
        print(f"| {k} | {r['filter']} | {pct(r['hit1'])} | {pct(r['hit3'])} | {r['full3']} | {r['full5']} | {r['mrr10']} | "
              f"{r['median_query_s']} s | ${r['cost_usd']} |")


def agent(split: str) -> None:
    base = RES / "agent" / split
    if not base.exists():
        return
    print(f"\n### Agent — {split}\n")
    print("| Cấu hình | Tool use | Search/câu (phân bố) | Context Evidence Rate | Lần search đầu Hit@1 / Hit@3 | Grounded 0/1/2 (TB) | "
          "Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Trung vị | Token in/out | Chi phí ước tính |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        a, lab = load(d / "summary_auto.json"), load(d / "summary_labels.json")
        if not a:
            continue
        g = lab["grounded_answer"] if lab else None
        gs = f"{g['0']}/{g['1']}/{g['2']} ({g['mean']:.2f})" if g else "chưa chấm"
        cv = f"{lab['citation_validity']['n']}/{lab['citation_validity']['N']}" if lab else "-"
        cc = f"{lab['answer_citation_coverage']['n']}/{lab['answer_citation_coverage']['N']}" if lab else "-"
        ab = f"{lab['abstention']['n']}/{lab['abstention']['N']}" if lab else a["abstain_on_unanswerable"]
        fa = f"{lab['false_abstention']['n']}/{lab['false_abstention']['N']}" if lab else "-"
        print(f"| {d.name} | {a['tool_use']} | {a['tool_calls_mean']} ({a['search_calls_distribution']}) | {pct(a['context_evidence_rate'])} | "
              f"{a['first_search_hit1']} / {a['first_search_hit3']} | {gs} | {cv} | {cc} | {ab} | {fa} | {a['median_s']} s | "
              f"{a['prompt_tokens']}/{a['output_tokens']} | ${a['cost_usd']} |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for sp in ("dev", "test"):
        retrieval(sp)
        agent(sp)
