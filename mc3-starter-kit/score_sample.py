"""Score sample answers the way the contest grades them.

A question is 20 points, or 0. The value and the citation set must both match.
The maximum is 200. This scores the sample set only. The graded set is different.

    python3 score_sample.py           # ask the local 3B model, then score
    python3 score_sample.py --rules   # check the scoring rules, no model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "sample-questions.json"
CORPUS = ROOT / "mc3-corpus"
APP = ROOT / "starter" / "app"
MODEL = ROOT / "starter" / "models" / "qwen"
POINTS = 20

_UNIT = re.compile(
    r"^(\d+)(DEGREES|DEGREE|CELSIUS|FAHRENHEIT|C|F|V|A|W|HZ|GHZ|MHZ|KHZ|MM|CM|NM|GB|GIB|MB|KB|MS|SEC|SECONDS|S)$"
)


def norm_answer(value: str) -> str:
    """Uppercase, drop spaces and - . · _, then drop a unit on a bare number."""
    text = (value or "").upper().replace("°", "").replace("º", "")
    text = re.sub(r"[\s\-\.\u00b7_]+", "", text)
    match = _UNIT.match(text)
    if match:
        return match.group(1)
    return text


def norm_cite(path: str) -> str:
    text = path.strip().replace("\\", "/")
    for prefix in ("/app/corpus/", "app/corpus/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.lstrip("./")


def points_for(query: dict, answer: str, citations: list[str]) -> int:
    """20 when the value and the citation set both match, else 0."""
    accepted = {norm_answer(query["expected_answer"])}
    accepted.update(norm_answer(alias) for alias in query.get("answer_aliases", []))
    if norm_answer(answer) not in accepted:
        return 0
    got = {norm_cite(path) for path in citations}
    expected = {norm_cite(path) for path in query["expected_citations"]}
    if got != expected:
        return 0
    return POINTS


def score_all(queries: list[dict], rows: list[dict]) -> tuple[int, list[str]]:
    by_query = {row["query"]: row for row in rows}
    total = 0
    lines = []
    for query in queries:
        row = by_query.get(query["query"], {"answer": None, "citations": ["__missing__"]})
        got = points_for(query, row.get("answer") or "", list(row.get("citations") or []))
        total += got
        mark = "OK" if got else "MISS"
        lines.append(f"{mark}  q{query['n']}  {got:2d}  answer={row.get('answer')!r}")
    lines.append(f"SCORE {total}/200")
    return total, lines


def _rpc(sock: str, payload: dict, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = "socket not ready"
    while True:
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(max(1.0, deadline - time.time()))
            conn.connect(sock)
            conn.sendall(json.dumps(payload).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                part = conn.recv(1 << 20)
                if not part:
                    break
                buf += part
            conn.close()
            return json.loads(buf.decode())
        except (ConnectionRefusedError, FileNotFoundError, ConnectionAbortedError) as exc:
            last = str(exc)
            if time.time() >= deadline:
                raise RuntimeError(last) from exc
            time.sleep(0.3)


def ask_model(queries: list[dict]) -> list[dict]:
    """Load the local 3B checkpoint and ask each sample question."""
    if not (MODEL / "config.json").is_file():
        raise SystemExit(f"No checkpoint at {MODEL}. Run with --rules to test the scorer only.")
    index = Path("/tmp/mc3-score-index")
    sock = "/tmp/mc3-score.sock"
    index.mkdir(parents=True, exist_ok=True)
    if os.path.exists(sock):
        os.remove(sock)
    env = os.environ.copy()
    env.update({
        "MC3_SOCKET": sock,
        "MC3_MODEL_DIR": str(MODEL),
        "MC3_INDEX_DIR": str(index),
        "MC3_CORPUS": str(CORPUS),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    proc = subprocess.Popen(
        [sys.executable, str(APP / "serve.py")],
        cwd=str(APP),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        os.environ["MC3_SOCKET"] = sock
        os.environ["MC3_INDEX_DIR"] = str(index)
        sys.path.insert(0, str(APP))
        import app as client

        client.INDEX_DIR = index
        client.index(CORPUS)
        rows = []
        for query in queries:
            got = _rpc(sock, {"op": "query", "query": query["query"]}, timeout=180)
            rows.append({
                "query": query["query"],
                "answer": got.get("answer", ""),
                "citations": got.get("citations", []),
            })
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the sample set from 0 to 200.")
    parser.add_argument("--rules", action="store_true", help="Score the published aliases. Do not load a model.")
    args = parser.parse_args()
    queries = json.loads(QUESTIONS.read_text(encoding="utf-8"))["queries"]
    if args.rules:
        rows = [
            {
                "query": query["query"],
                "answer": (query.get("answer_aliases") or [query["expected_answer"]])[0],
                "citations": query["expected_citations"],
            }
            for query in queries
        ]
    else:
        rows = ask_model(queries)
    _total, lines = score_all(queries, rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
