# -*- coding: utf-8 -*-
"""Unit test cho code Day 7 (không gọi mạng; LLM và retriever là mock — chỉ kiểm tra logic, không phải benchmark)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day07 import agent as A  # noqa: E402
from day07.metrics import context_full, groups_hit, ranking_metrics  # noqa: E402
from day07.retrievers import _match, rrf  # noqa: E402
from day07.textnorm import contains, norm  # noqa: E402


def chunk(cid, text, sid="doc", page=1):
    return {"id": cid, "text": text, "source_id": sid, "source": "x", "page": page,
            "metadata": {"doc_code": "X", "regulation": "r", "articles_covered": "Điều 1", "location": f"p{page}"}}


class TextNormTest(unittest.TestCase):
    def test_norm_and_contains(self):
        self.assertEqual(norm("Điều 9.  Phân loại!"), "điều 9 phân loại")
        self.assertTrue(contains("a) Từ 90 đến 100 điểm: loại xuất sắc;", "từ 90 đến 100 điểm loại xuất sắc"))


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.q = {"gold_groups": [[{"source_id": "doc", "page": 1, "evidence": "alpha beta"}],
                                  [{"source_id": "doc", "page": 2, "evidence": "gamma"}]]}

    def test_hit_requires_source_page_and_evidence(self):
        self.assertEqual(groups_hit(chunk("a", "xx alpha beta yy", page=1), self.q), {0})
        self.assertEqual(groups_hit(chunk("b", "alpha beta", page=3), self.q), set())
        self.assertEqual(groups_hit(chunk("c", "alpha", page=1), self.q), set())

    def test_ranking_and_full(self):
        res = [chunk("x", "nothing"), chunk("a", "alpha beta", page=1), chunk("g", "gamma", page=2)]
        m = ranking_metrics(res, self.q)
        self.assertFalse(m["hit1"])
        self.assertTrue(m["hit3"])
        self.assertTrue(m["full3"])
        self.assertAlmostEqual(m["rr10"], 0.5)
        self.assertFalse(context_full(res[:2], self.q))


class RetrieverUtilTest(unittest.TestCase):
    def test_match_eq_and_ne(self):
        self.assertTrue(_match({"kind": "chart"}, {"kind": {"$ne": "text"}}))
        self.assertFalse(_match({"kind": "text"}, {"kind": {"$ne": "text"}}))
        self.assertTrue(_match({"regulation": "a"}, {"regulation": "a"}))
        self.assertFalse(_match({"regulation": "b"}, {"regulation": "a"}))

    def test_rrf_order(self):
        l1 = [{"id": "a", "rank": 1, "method": "d"}, {"id": "b", "rank": 2, "method": "d"}]
        l2 = [{"id": "b", "rank": 1, "method": "s"}, {"id": "c", "rank": 2, "method": "s"}]
        self.assertEqual([x[0] for x in rrf([l1, l2])][0], "b")


class FakeLLM:
    """Trả lần lượt các function call đã định sẵn (định dạng Gemini)."""
    model = "fake"
    thinking_level = None

    def __init__(self, calls):
        self.calls = list(calls)

    def generate(self, contents, **kw):
        if kw.get("usage") is not None:
            kw["usage"].calls += 1
        name, args = self.calls.pop(0)
        return {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}}]}


class FakeRetriever:
    name = "fake"

    def __init__(self, chunks):
        self.chunks = chunks
        self.seen_filters = []

    def search(self, query, *, filters=None, top_k=3):
        self.seen_filters.append(filters)
        return [{**c, "score": 1.0, "rank": i + 1} for i, c in enumerate(self.chunks[:top_k])]


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.c1 = chunk("tt08-2021:p7:st03", "Một năm học có 02 hoặc 03 học kỳ chính.")
        self.c2 = chunk("tt08-2021:p8:st00", "Đoạn khác.")
        self._orig = A.chunk_by_id
        A.chunk_by_id = lambda: {c["id"]: c for c in (self.c1, self.c2)}

    def tearDown(self):
        A.chunk_by_id = self._orig

    def make(self, calls, **kw):
        return A.Agent(FakeLLM(calls), FakeRetriever([self.c1, self.c2]), name="t", top_k=2, **kw)

    def test_answer_with_valid_citation(self):
        ag = self.make([("search_regulations", {"query": "học kỳ"}),
                        ("final_answer", {"status": "answered", "answer": "02 hoặc 03 [tt08-2021:p7:st03]", "citations": ["tt08-2021:p7:st03"]})])
        out = ag.run("Một năm có mấy học kỳ?", filters={"regulation": "quy_che_dao_tao"})
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["tt08-2021:p7:st03"])
        self.assertEqual(ag.retriever.seen_filters[0], {"regulation": "quy_che_dao_tao"})  # filter người dùng luôn được áp dụng

    def test_invalid_citation_repaired_then_abstain(self):
        bad = ("final_answer", {"status": "answered", "answer": "x [tt99-2000:p1:st0]", "citations": ["tt99-2000:p1:st0"]})
        ag = self.make([("search_regulations", {"query": "q"}), bad, bad])
        out = ag.run("?")
        self.assertEqual(out["status"], "abstain")
        self.assertTrue(any(e["event"] == "invalid_citations" for e in out["events"]))

    def test_budget_limits_context(self):
        ag = self.make([("search_regulations", {"query": "q"}),
                        ("final_answer", {"status": "insufficient_evidence", "answer": "không có", "citations": []})], budget_chars=45)
        out = ag.run("?")
        self.assertEqual(out["context_ids"], ["tt08-2021:p7:st03"])
        self.assertEqual(out["events"][0]["results"][1]["context"], "over_budget")

    def test_strict_abstain_converts_answer(self):
        ag = self.make([("search_regulations", {"query": "q"}),
                        ("final_answer", {"status": "answered", "answer": "Tài liệu không quy định [tt08-2021:p7:st03]",
                                          "citations": ["tt08-2021:p7:st03"], A.STRICT_FIELD: False})], strict_abstain=True)
        out = ag.run("?")
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["citations"], [])

    def test_search_limit_forces_final(self):
        calls = [("search_regulations", {"query": f"q{i}"}) for i in range(2)] + \
                [("final_answer", {"status": "insufficient_evidence", "answer": "không đủ", "citations": []})]
        out = self.make(calls, max_tool_calls=2).run("?")
        self.assertEqual(out["n_search"], 2)
        self.assertEqual(out["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
