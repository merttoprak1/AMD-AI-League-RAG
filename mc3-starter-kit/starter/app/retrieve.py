"""BM25 over the chunks, plus one hop on identifiers the question does not name."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
# Ticket, part, and error tokens. A digit is required so a constant name
# such as DEFAULT_BATCH_TIMEOUT_S does not start a second search.
ID_RE = re.compile(r"\b(?=[A-Z0-9_-]*\d)[A-Z]{2,}(?:[-_][A-Z0-9]+)+\b|\b[A-Z]\d{3,}\b")
POOL = 16


def retrieve(question: str, chunks: list[dict], k: int = 8) -> list[dict]:
    """Return up to k chunks. Identifier hits the question did not name come first."""
    usable = [chunk for chunk in chunks if chunk.get("text")]
    if not usable:
        return []
    ranked = _rank(question, usable)
    pool = ranked[:POOL]
    question_ids = set(identifiers(question))
    fresh: list[str] = []
    for chunk in pool:
        for token in identifiers(chunk["text"]):
            if token not in question_ids and token not in fresh:
                fresh.append(token)
    wanted = set(fresh)
    hop_new: list[dict] = []
    carriers: list[dict] = []
    carried_paths: set[str] = set()
    for chunk in pool:
        if wanted.intersection(identifiers(chunk["text"])) and chunk["path"] not in carried_paths:
            carriers.append(chunk)
            carried_paths.add(chunk["path"])
            if len(carriers) >= 3:
                break
    if fresh:
        already = {(chunk["path"], chunk["text"]) for chunk in pool}
        for chunk in _rank(" ".join(fresh), usable):
            key = (chunk["path"], chunk["text"])
            if key in already or not wanted.intersection(identifiers(chunk["text"])):
                continue
            hop_new.append(chunk)
            if len(hop_new) >= 2:
                break
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for chunk in hop_new + carriers + pool:
        key = (chunk["path"], chunk["text"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= k:
            break
    return merged


def rank_paths(question: str, chunks: list[dict]) -> dict[str, float]:
    """Best question score for each path. Used to pick a citation, not to hop."""
    scores: dict[str, float] = {}
    for chunk in _rank(question, [c for c in chunks if c.get("text")]):
        path = chunk["path"]
        score = float(chunk["score"])
        if score > scores.get(path, 0.0):
            scores[path] = score
    return scores


def identifiers(text: str) -> list[str]:
    return ID_RE.findall(text)


def _rank(question: str, chunks: list[dict]) -> list[dict]:
    # The path is part of the search text. "production log" can match logs/prod_....log.
    docs = [_terms(chunk["path"] + "\n" + chunk["text"]) for chunk in chunks]
    if not any(docs):
        return []
    engine = BM25Okapi(docs)
    raw = engine.get_scores(_terms(question))
    ranked = []
    for chunk, score in zip(chunks, raw):
        if score <= 0:
            continue
        item = dict(chunk)
        item["score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in WORD_RE.finditer(text):
        token = match.group().lower()
        terms.append(token)
        parts = re.split(r"[-_]", token)
        if len(parts) > 1:
            terms.extend(part for part in parts if part)
    return terms
