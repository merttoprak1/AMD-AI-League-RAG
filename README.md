# AMD AI League: Mini-Challenge 3

A document question-answering project for the AMD AI League Match 3 challenge.
It parses a mixed-format corpus, retrieves relevant passages with BM25, and returns answers with file citations.
A resident Qwen vision-language model serves requests through a Unix socket.

## Repository contents

| Path | Purpose |
| --- | --- |
| `mc3-starter-kit/starter/app/` | Parsing, retrieval, model service, and citation code |
| `mc3-starter-kit/starter/Dockerfile` | ROCm container for the challenge |
| `mc3-starter-kit/mc3-corpus/` | Sample documents, images, and error cases |
| `mc3-starter-kit/sample-questions.json` | Sample questions with expected answers and citations |
| `mc3-starter-kit/tests/` | Tests that do not load a model |
| `mc3-starter-kit/score_sample.py` | Sample scorer, with a maximum of 200 points |
| `mc3-starter-kit/selfcheck.py` | Container checks |
| `mc3-starter-kit/CONTRACT.md` | Submission requirements |

The sample products and records are fictional challenge fixtures.
The log, encrypted PDF, binary file, and unreadable audit file are intentional test inputs.
Keep them in the corpus. The evaluation corpus and questions differ from these samples.

## Run the local checks

Use Python 3.14 and Tesseract OCR. Install Tesseract with your operating system package manager.
Run these commands from the repository root:

```sh
cd mc3-starter-kit
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r starter/app/requirements.txt
./setup.sh
python -m unittest discover -s tests -v
python score_sample.py --rules
```

`setup.sh` makes the audit fixture unreadable. Run the tests as a regular user on Linux or macOS.
Git does not preserve this permission, so run the setup script after each fresh clone.
Windows does not enforce the same permission behavior.

The tests check parsing, retrieval, citations, scoring, and socket requests without model weights.
The `--rules` command scores the published answers. Its result does not measure model accuracy.

## Build the challenge container

Place the `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint and processor files in
`mc3-starter-kit/starter/models/qwen-7b/` before the build.
Model weights are local files and do not belong in Git.

From the repository root:

```sh
cd mc3-starter-kit
docker build -t amd-mc3:local starter
python3 selfcheck.py amd-mc3:local mc3-corpus
```

The Dockerfile installs the model libraries and copies the checkpoint to `/models/qwen`.
The challenge requires the ROCm base image specified in the Dockerfile and an AMD GPU.
Read [the submission contract](mc3-starter-kit/CONTRACT.md) for runtime limits and offline evaluation rules.

The optional local command `python score_sample.py` uses a separate checkpoint at `starter/models/qwen/`.
It also requires PyTorch, Pillow, Transformers, Tokenizers, and Safetensors in the local environment.
The service selects ROCm/CUDA, Apple MPS, or CPU according to availability.
Local CPU and MPS runs do not prove that the container meets the challenge requirements.

## Local files

Git ignores environments, model weights, credentials, caches, runtime output, and unrelated sheet-layout exports.
The Docker context includes only the application and the checkpoint used by the container.

## License

This repository does not declare a license.
The supplied challenge materials and model weights remain subject to their respective terms.
