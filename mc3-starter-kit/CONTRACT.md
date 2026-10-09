# Mini-Challenge 3 — the submission contract

You submit a Docker image. We run it on an AMD GPU, give it a folder of mixed
documents, ask it ten questions, and read a JSON file back for each one. This
file is the contract that image has to satisfy: how your script is invoked, what
it must write, and what scores zero no matter how good your model is.

The container rules are the same for every challenge in this programme and are
set out in full, with the reasoning behind them, in **Section 2 of "All About
the Challenges"**. Read that first if you have not already — most lost points
are lost there rather than on model quality. The limits themselves are repeated
below so this kit is usable on its own, and the mandated base image is the
`FROM` line of `starter/Dockerfile` here.

## The limits

| | | |
|---|---|---|
| Image size | **60 GiB**, uncompressed | Measured uncompressed — not the smaller number `docker images` prints |
| VRAM | **1 – 48 GiB**, 1% tolerance on the upper bound | Sampled continuously and the peak is what counts. Below the floor means you ran on the CPU |
| Startup | **10 minutes** | Container start, model load, *and* the `--index` pass over the whole corpus |
| Per question | **30 seconds** | One `--query` invocation, from exec to your JSON file |
| Whole run | **10 minutes** | All ten questions, after startup completes |

Ten questions, **20 points each, 200 maximum**. A question scores only when the
answer and the citations are *both* right — there is no partial credit within a
question.

**The image size limit, not the VRAM band, is what bounds your model.** The
mandated base image is 29.2 GiB on its own, which leaves roughly 30 GiB of the
60 for weights — about a 16B parameter model at bf16. The 48 GiB VRAM band is a
separate and larger limit, so sizing your model to *it* produces an image you
cannot submit.

> **If you did Mini-Challenge 2**, the container rules are unchanged — same base
> image, same size and VRAM limits, same "no server, no port" shape. What is new
> is that your script is invoked in **two** different ways, and that answers now
> carry **citations**.

## What a participant gets

| | |
|---|---|
| `starter/` | A working container that already implements the contract. Clone it, fill in two functions. |
| `selfcheck.py` | Run it against your container before submitting. Same checks the harness runs, in the same order. |

## The contract

**Phase 1 — index, once, before any graded clock.** Charged to your startup
budget, not to a question:

```
python3 /app/app.py --index /app/corpus
```

**Phase 2 — one execution per question, ten times:**

```
python3 /app/app.py --corpus /app/corpus \
                    --query-id query_01 \
                    --query "What is the maximum junction temperature?"
```

and the harness reads the file your script writes:

```
/app/output/query_01_output.json
    -> {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

The output stem is the **`--query-id` you are given**, not anything derived from
the question. A question has no filename to take a stem from, so the harness
assigns the id and you must use exactly it.

| field | |
|---|---|
| `answer` | Graded. The **value only** — `94`, not "The maximum junction temperature is 94 C". Qualifiers that are part of the value stay: `Q3 FY27`, not `Q3`. Units and degree signs are normalised away, so `94`, `94 C` and `94°C` all match. |
| `citations` | Graded, as an **exact set**. Corpus-relative paths. |
| `confidence` | Recorded, never scored. |

A missing `answer` or a missing `citations` is **malformed and scores zero** — a
crashed writer and a considered refusal must not look the same.

## Citations: necessity, not relevance

**Cite a file only if removing it would make your answer impossible.** Dumping
your whole retrieval fails almost every question even when the answer is right.
Some questions genuinely need two files; cite both, and only those.

## Refusing

If the corpus does not contain the answer, return an **empty answer and an empty
citation list**. `"I don't know"` is prose, and prose is graded as a wrong
answer. Guessing is wrong — and so is answering from what your model already
knows, because the products in this corpus are fictional. Anything recalled
rather than retrieved is wrong by construction.

## What your code owns

All of it. Parsing, indexing, retrieval, generation, citation selection. The
harness copies a corpus in, runs your script, and reads a JSON file out.

## The things that score zero regardless of your model

- Not building on the mandated ROCm base image. Checked by layer identity before
  your container runs, so renaming an image does not help.
- Going over the uncompressed size limit.
- Exceeding the startup, per-question or whole-run time budget.
- Writing the output anywhere other than `/app/output/<query-id>_output.json`.
- Using no GPU. VRAM is sampled continuously and the peak is what counts; a
  submission that stays under 1 GiB did its work on the CPU and is rejected.
  Place your model on the device explicitly rather than letting it fall back.
  Pinning a dummy buffer to clear the floor is detected separately.
- **Downloading anything at evaluation time.** Your container has **no outbound
  network access** while it is graded. Ship your weights in the image and test
  with `--network none`.

## Two things worth planning for

**The model is loaded per question unless you stop it.** The harness starts a
**new process** for each of the ten questions. A naive implementation loads ten
times and blows the budget. Do the expensive work in `--index`, persist it, and
keep the model resident behind a unix socket — or make loading cheap enough not
to matter. This is the single most likely reason a working solution times out.

**One unreadable file must not cost you the rest of the corpus.** The corpus
deliberately contains an unreadable file, an encrypted PDF, an unknown binary
type and an empty directory. A walk that raises on the first unreadable file
indexes nothing after it — and because walk order follows the directory listing,
*which* files you lose depends on filename order. That is how a submission tests
clean locally and grades badly here. Catch per file, keep going.
