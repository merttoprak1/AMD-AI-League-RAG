"""Resident process. The query script is a new process and only talks to this socket.

Start this from the container CMD. It binds the socket before the model loads,
so --index can wait on the same startup budget.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

from cite import cite
from parse import load_chunks, save_chunks
from retrieve import retrieve

SOCKET_PATH = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/qwen")
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))

_lock = threading.Lock()
_ready = threading.Event()
_chunks: list[dict] = []
_model = None
_processor = None
_boot_error: str | None = None


def main() -> None:
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    threading.Thread(target=_boot, daemon=True).start()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o666)
    server.listen(8)
    while True:
        conn, _addr = server.accept()
        threading.Thread(target=_handle, args=(conn,), daemon=True).start()


def _boot() -> None:
    global _chunks, _model, _processor, _boot_error
    try:
        _chunks = load_chunks(INDEX_DIR)
        model_dir = Path(MODEL_DIR)
        if (model_dir / "config.json").is_file():
            _model, _processor = _load_model(model_dir)
    except Exception as exc:
        _boot_error = f"{type(exc).__name__}: {exc}"
    finally:
        _ready.set()


def _handle(conn: socket.socket) -> None:
    with conn:
        try:
            raw = _readline(conn)
            if not _ready.wait(timeout=540):
                _send(conn, {"ok": False, "error": "model load exceeded 540s"})
                return
            if _boot_error:
                _send(conn, {"ok": False, "error": _boot_error})
                return
            request = json.loads(raw.decode("utf-8"))
            op = request.get("op")
            with _lock:
                if op == "finalize":
                    corpus = Path(request.get("corpus") or os.environ.get("MC3_CORPUS", "/app/corpus"))
                    _send(conn, _finalize(Path(request["index_dir"]), corpus))
                elif op == "query":
                    _send(conn, _query(request.get("query", "")))
                else:
                    _send(conn, {"ok": False, "error": "unknown op"})
        except Exception as exc:
            try:
                _send(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass


def _finalize(index_dir: Path, corpus: Path) -> dict:
    global _chunks
    _chunks = load_chunks(index_dir)
    if _model is not None:
        changed = False
        for chunk in _chunks:
            if not chunk.get("needs_vision"):
                continue
            transcript = _transcribe(corpus, chunk["path"])
            if transcript:
                prior = chunk.get("text") or ""
                chunk["text"] = prior + ("\n" if prior else "") + transcript
                changed = True
            chunk["needs_vision"] = False
        if changed:
            save_chunks(index_dir, _chunks)
    return {"ok": True, "chunks": len(_chunks)}


def _query(question: str) -> dict:
    excerpts = retrieve(question, _chunks)
    raw = _ask_model(question, excerpts) if _model is not None else ""
    answer, citations, confidence = cite(question, raw, _chunks, excerpts)
    return {"ok": True, "answer": answer, "citations": citations, "confidence": confidence}


def _ask_model(question: str, excerpts: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(excerpts, start=1):
        state = " withdrawn" if chunk.get("withdrawn") else ""
        blocks.append(f"[{index}] path: {chunk['path']}{state}\n{chunk['text']}")
    prompt = (
        "Answer from the excerpts only. The products are fictional. "
        'If the excerpts do not contain the answer, return {"answer":""}.\n'
        "Return one JSON object with the key answer. Copy the value from an excerpt. "
        "Keep a qualifier that identifies the value. "
        "A current excerpt wins over a withdrawn excerpt. "
        "An assignment wins over a comment.\n\n"
        f"Excerpts:\n{chr(10).join(blocks)}\n\nQuestion: {question}"
    )
    return _parse_model_json(_generate(prompt))


def _parse_model_json(raw: str) -> str:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return ""
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return ""
    answer = obj.get("answer", "")
    return answer if isinstance(answer, str) else ""


def _load_model(model_dir: Path):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/ROCm device")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def _generate(prompt: str) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], return_tensors="pt")
    inputs = {key: value.to(_model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = _model.generate(**inputs, max_new_tokens=80, do_sample=False)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return _processor.decode(new_tokens, skip_special_tokens=True)


def _transcribe(corpus: Path, rel: str) -> str:
    """Read one image with the vision model. A missing file yields empty text."""
    import torch
    from PIL import Image

    image_path = corpus / rel
    if not image_path.is_file():
        return ""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Transcribe every visible string. Keep each label beside its value.",
                },
            ],
        }
    ]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(_model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = _model.generate(**inputs, max_new_tokens=200, do_sample=False)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return _processor.decode(new_tokens, skip_special_tokens=True).strip()


def _readline(conn: socket.socket) -> bytes:
    buf = b""
    while b"\n" not in buf:
        part = conn.recv(65536)
        if not part:
            break
        buf += part
    return buf.split(b"\n", 1)[0]


def _send(conn: socket.socket, payload: dict) -> None:
    conn.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
