"""Sample-corpus checks that run without a GPU.

Retrieval and citation use the expected answer as the value. The model is
not loaded. Question 10 must stay empty even if a guess is supplied.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "starter" / "app"
CORPUS = ROOT / "mc3-corpus"
sys.path.insert(0, str(APP))

from cite import cite, find_span  # noqa: E402
from parse import build_index  # noqa: E402
from retrieve import retrieve  # noqa: E402

QUESTIONS = json.loads((ROOT / "sample-questions.json").read_text(encoding="utf-8"))["queries"]


class SampleCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index = Path(cls._tmp.name)
        cls.chunks = build_index(CORPUS, cls.index)
        cls.by_path: dict[str, list[dict]] = {}
        for chunk in cls.chunks:
            cls.by_path.setdefault(chunk["path"], []).append(chunk)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_walk_keeps_going(self):
        skipped = json.loads((self.index / "skipped.json").read_text(encoding="utf-8"))
        reasons = {row["path"]: row["reason"] for row in skipped}
        self.assertIn("vendor/internal_audit.txt", reasons)
        self.assertIn("vendor/supplier_agreement_ENCRYPTED.pdf", reasons)
        self.assertIn("vendor/telemetry_capture.dat", reasons)
        self.assertIn("specs/tq40_datasheet_r2.pdf", self.by_path)
        self.assertIn("support/bug_database.csv", self.by_path)
        self.assertIn("planning/roadmap_fy27.docx", self.by_path)
        self.assertTrue(any(c["withdrawn"] for c in self.by_path["specs/tq40_datasheet_r1_WITHDRAWN.pdf"]))
        self.assertFalse(any(c["withdrawn"] for c in self.by_path["specs/tq40_datasheet_r2.pdf"]))

    def test_short_quarter_keeps_the_fiscal_year(self):
        chunks = [{"path": "planning/roadmap_fy27.docx", "text": "sampling in Q3 FY27.", "withdrawn": False}]
        span, paths, _confidence = cite("When does sampling start?", "Q3", chunks, chunks)
        self.assertEqual(span, "Q3 FY27")
        self.assertEqual(paths, ["planning/roadmap_fy27.docx"])

    def test_span_does_not_match_inside_a_longer_token(self):
        self.assertIsNone(find_span("ORR-HSK-1180-A", "180"))
        self.assertEqual(find_span("DEFAULT_BATCH_TIMEOUT_S = 180", "180"), "180")

    def test_retrieval_hits_the_expected_files(self):
        for query in QUESTIONS:
            if query.get("unanswerable"):
                continue
            found = {chunk["path"] for chunk in retrieve(query["query"], self.chunks)}
            missing = [path for path in query["expected_citations"] if path not in found]
            self.assertEqual(missing, [], query["query"])

    def test_citations_match_the_sample(self):
        for query in QUESTIONS:
            found = retrieve(query["query"], self.chunks)
            _span, paths, _confidence = cite(
                query["query"],
                query["expected_answer"],
                self.chunks,
                found,
            )
            self.assertEqual(paths, query["expected_citations"], query["query"])

    def test_a_guess_that_is_not_in_the_corpus_is_refused(self):
        query = QUESTIONS[-1]["query"]
        found = retrieve(query, self.chunks)
        span, paths, confidence = cite(query, "6412", self.chunks, found)
        self.assertEqual((span, paths, confidence), ("", [], 0.0))


class SocketTest(unittest.TestCase):
    def test_query_roundtrip_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp)
            sock = str(Path(tmp) / "mc3.sock")
            env = os.environ.copy()
            env.update({
                "MC3_SOCKET": sock,
                "MC3_INDEX_DIR": str(index),
                "MC3_MODEL_DIR": str(Path(tmp) / "no-model"),
                "MC3_CORPUS": str(CORPUS),
            })
            proc = subprocess.Popen(
                [sys.executable, str(APP / "serve.py")],
                cwd=str(APP),
                env=env,
            )
            try:
                self._wait_for_socket(sock)
                os.environ["MC3_SOCKET"] = sock
                os.environ["MC3_INDEX_DIR"] = str(index)
                import app as client

                client.INDEX_DIR = index
                client.index(CORPUS)
                answer, citations, confidence = client.answer(CORPUS, "What is the unit price?")
                self.assertIsInstance(answer, str)
                self.assertIsInstance(citations, list)
                self.assertIsInstance(confidence, float)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _wait_for_socket(self, sock: str) -> None:
        deadline = time.time() + 10
        while time.time() < deadline:
            if os.path.exists(sock):
                try:
                    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    conn.settimeout(0.2)
                    conn.connect(sock)
                    conn.close()
                    return
                except OSError:
                    pass
            time.sleep(0.05)
        self.fail("server socket did not open")


if __name__ == "__main__":
    unittest.main()
