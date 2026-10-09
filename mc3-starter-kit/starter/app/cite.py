"""Pick the minimum set of files that make an answer possible.

The model returns a value. This module accepts that value only when a chunk
contains it as a token span, then chooses the files.
"""

from __future__ import annotations

import re

from retrieve import identifiers, rank_paths

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SQUEEZE_RE = re.compile(r"[\s\-\.\u00b7_]+")
FY_AFTER = re.compile(r"\s+FY\d+\b", re.I)


def cite(
    question: str,
    answer: str,
    chunks: list[dict],
    retrieved: list[dict],
) -> tuple[str, list[str], float]:
    """Return (span, relative paths, confidence). Empty means refuse."""
    if not answer or not answer.strip():
        return "", [], 0.0

    spans: dict[str, str] = {}
    file_withdrawn: dict[str, bool] = {}
    file_ids: dict[str, set[str]] = {}
    for chunk in chunks:
        path = chunk["path"]
        flag = bool(chunk.get("withdrawn"))
        file_withdrawn[path] = file_withdrawn.get(path, True) and flag
        file_ids.setdefault(path, set()).update(identifiers(chunk.get("text", "")))
        if path in spans:
            continue
        span = find_span(chunk.get("text", ""), answer)
        if span is not None:
            spans[path] = _with_fiscal_year(chunk.get("text", ""), span)

    if not spans:
        return "", [], 0.0

    current = [path for path in spans if not file_withdrawn.get(path, False)]
    chosen = current or list(spans)
    scores = rank_paths(question, chunks)
    source = max(chosen, key=lambda path: (scores.get(path, 0.0), path))

    question_ids = set(identifiers(question))
    bridge_ids = file_ids.get(source, set()) - question_ids
    retrieved_paths = {chunk["path"] for chunk in retrieved}
    bridges = []
    source_score = scores.get(source, 0.0)
    for path in retrieved_paths:
        if path == source or path in spans:
            continue
        shared = bridge_ids.intersection(file_ids.get(path, set()))
        if shared and scores.get(path, 0.0) > source_score:
            bridges.append(path)

    span = spans[source]
    if bridges:
        bridge = max(bridges, key=lambda path: (scores.get(path, 0.0), path))
        return span, sorted([source, bridge]), 0.6
    return span, [source], 0.9


def _with_fiscal_year(text: str, span: str) -> str:
    """Q3 in the file is Q3 FY27. A short model answer must keep the year."""
    if not re.fullmatch(r"Q\d+", span, re.I):
        return span
    at = text.find(span)
    if at < 0:
        return span
    match = FY_AFTER.match(text[at + len(span) :])
    if not match:
        return span
    return text[at : at + len(span) + match.end()]


def find_span(text: str, answer: str) -> str | None:
    """A run of tokens whose letters and digits equal the answer.

    Matching the squeezed whole file would treat 180 as present inside 1180.
    """
    target = SQUEEZE_RE.sub("", answer).upper()
    if not target:
        return None
    tokens = list(TOKEN_RE.finditer(text))
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, min(start + 8, len(tokens))):
            acc += tokens[end].group().upper()
            if acc == target:
                return text[tokens[start].start() : tokens[end].end()]
            if not target.startswith(acc):
                break
    return None
