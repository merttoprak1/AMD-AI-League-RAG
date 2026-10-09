#!/usr/bin/env python3
"""Check your container against the MC-3 submission contract before you submit.

These are the same checks the harness runs, in the same order. If any of them
fails here it will fail there, and there it costs you the submission.

    python3 selfcheck.py your-image:tag path/to/a/sample/corpus

The corpus you pass is any directory laid out like the sample one -- it is only
used to prove your index pass and one query work end to end. It does not have
to contain the real answers.

Checks the two-phase contract MC-2 does not have: an --index pass that runs
once before any graded clock, then one exec per question writing
/app/output/<query-id>_output.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

SCRIPT = "/app/app.py"
CORPUS_DIR = "/app/corpus"
OUTPUT_DIR = "/app/output"
STARTUP_BUDGET_S = 600
INDEX_BUDGET_S = 600
PER_QUERY_BUDGET_S = 30
VRAM_FLOOR_GIB = 1
VRAM_CEILING_GIB = 48

SAMPLE_QUERY = "What is the maximum junction temperature?"
SAMPLE_QUERY_ID = "query_01"


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def ok(msg):
    print(f"  [PASS] {msg}")
    return True


def bad(msg):
    print(f"  [FAIL] {msg}")
    return False


def warn(msg):
    print(f"  [WARN] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="your built image, e.g. ghcr.io/you/rag:v1")
    ap.add_argument("corpus", type=Path, help="a sample corpus directory to try")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        return 1 if bad(f"{args.corpus} is not a directory") else 0

    passed = True
    name = f"selfcheck-{uuid.uuid4().hex[:8]}"

    print("1. size, uncompressed")
    hist = run(["docker", "history", "--no-trunc", "--format", "{{.Size}}", args.image])
    if hist.returncode != 0:
        passed = bad(f"could not read image history: {hist.stderr.strip()}")
    else:
        ok("image exists locally; compare the layer sum against the published limit")

    print("2. the container starts WITH NO NETWORK")
    # --network none on purpose: evaluation runs on an internal network with no
    # route off the box. Anything your code downloads at runtime hangs there.
    started = time.time()
    up = run(["docker", "run", "-d", "--network", "none", "--name", name, args.image])
    if up.returncode != 0:
        return 1 if bad(f"container did not start: {up.stderr.strip()}") else 0
    ok(f"started in {time.time() - started:.1f}s (budget {STARTUP_BUDGET_S}s)")

    try:
        print("3. the corpus copies in, and your index pass runs")
        run(["docker", "exec", name, "mkdir", "-p", CORPUS_DIR, OUTPUT_DIR], timeout=60)
        cp = run(["docker", "cp", f"{args.corpus}/.", f"{name}:{CORPUS_DIR}"], timeout=600)
        if cp.returncode != 0:
            passed = bad(f"could not copy the corpus in: {cp.stderr.strip()}")
        t0 = time.time()
        try:
            proc = run(
                ["docker", "exec", name, "python3", SCRIPT, "--index", CORPUS_DIR],
                timeout=INDEX_BUDGET_S,
            )
        except subprocess.TimeoutExpired:
            passed = bad(f"--index did not finish within {INDEX_BUDGET_S}s")
            proc = None
        if proc is not None:
            if proc.returncode != 0:
                passed = bad(
                    f"--index exited {proc.returncode}: "
                    f"{(proc.stderr or proc.stdout).strip()[:400]}"
                )
            else:
                ok(f"indexed in {time.time() - t0:.1f}s (charged to startup, not to a query)")

        print("4. one query answers in time")
        t0 = time.time()
        try:
            proc = run(
                ["docker", "exec", name, "python3", SCRIPT,
                 "--corpus", CORPUS_DIR,
                 "--query-id", SAMPLE_QUERY_ID,
                 "--query", SAMPLE_QUERY],
                timeout=PER_QUERY_BUDGET_S,
            )
        except subprocess.TimeoutExpired:
            passed = bad(
                f"no answer within the {PER_QUERY_BUDGET_S}s per-query budget. "
                "The harness starts a NEW PROCESS per question -- if you load the "
                "model here rather than reusing work from --index, you will blow "
                "this on every one of the ten."
            )
            proc = None
        if proc is not None:
            if proc.returncode != 0:
                passed = bad(
                    f"{SCRIPT} exited {proc.returncode}: "
                    f"{(proc.stderr or proc.stdout).strip()[:400]}"
                )
            else:
                ok(f"answered in {time.time() - t0:.1f}s")

        print("5. the output file is where we look for it, and the right shape")
        want = f"{OUTPUT_DIR}/{SAMPLE_QUERY_ID}_output.json"
        cat = run(["docker", "exec", name, "cat", want], timeout=60)
        if cat.returncode != 0:
            passed = bad(
                f"no {want}. The name is the --query-id you were GIVEN plus "
                "_output.json -- not derived from the question text."
            )
        else:
            try:
                payload = json.loads(cat.stdout)
            except ValueError:
                passed = bad(f"{want} is not valid JSON")
            else:
                if not isinstance(payload, dict):
                    passed = bad("the JSON must be an object")
                elif not isinstance(payload.get("answer"), str):
                    passed = bad("missing a string 'answer' -- a missing key is malformed and scores zero")
                elif not isinstance(payload.get("citations"), list):
                    passed = bad("missing a list 'citations' -- a missing key is malformed and scores zero")
                elif not all(isinstance(c, str) for c in payload["citations"]):
                    passed = bad("every citation must be a string path")
                else:
                    ok(f"answer={payload['answer']!r} citations={payload['citations']}")
                    for c in payload["citations"]:
                        if c.startswith("/") or c.startswith(CORPUS_DIR):
                            warn(
                                f"citation {c!r} looks absolute. Cite paths RELATIVE to the "
                                "corpus root, e.g. 'specs/datasheet_r2.pdf'."
                            )
                    if payload["answer"] and not payload["citations"]:
                        warn("a non-empty answer with no citations scores zero. Refusals are "
                             "empty answer AND empty citations.")

        print("6. VRAM")
        print("     run this while your solution works and watch the peak:")
        print("       watch -n1 'amd-smi metric --mem | grep USED_VRAM'")
        print(f"     it must stay between {VRAM_FLOOR_GIB} and {VRAM_CEILING_GIB} GiB.")
        print("     under the floor means you ran on the CPU, which scores zero --")
        print("     and pinning a dummy buffer to fake it is detected separately.")
    finally:
        run(["docker", "rm", "-f", name], timeout=120)

    print()
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED - fix these before submitting")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
