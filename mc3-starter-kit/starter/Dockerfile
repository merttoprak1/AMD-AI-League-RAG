# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
#
# It is the same image your development session runs, and it supports both
# gfx942 (MI300X, CDNA) and gfx1100 (Navi 31, RDNA3), so one submission runs on
# whichever GPU you are given. It ships torch and PIL ONLY; add whatever else
# you need to requirements.txt.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# ⚠️ SHIP YOUR WEIGHTS IN THE IMAGE. Your container has NO outbound network
# access during evaluation -- it runs on an internal Docker network with no
# route off the box. Code that downloads a model at evaluation time will hang
# and then time out, scoring zero, no matter how good the model is.
#
# Test that assumption the way the harness enforces it:
#
#     docker run --rm --network none your-image:tag python3 /app/app.py --index /app/corpus
#
# Weights count against the 60 GiB uncompressed image limit.
#   COPY models/ /models/

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/requirements.txt
# --no-deps on the model packages only. A plain "pip install transformers"
# can replace torch 2.13.0+rocm with a CUDA build.
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir --no-deps transformers tokenizers safetensors \
    && python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__"

# Download Qwen/Qwen2.5-VL-7B-Instruct into models/qwen-7b before this build.
# The grade network cannot fetch weights. The container path stays /models/qwen.
COPY models/qwen-7b /models/qwen

COPY app/ /app/

# The harness invokes this script TWICE in different shapes. Once to index,
# before any graded clock:
#
#     python3 /app/app.py --index /app/corpus
#
# then once per question, ten times:
#
#     python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
#
# reading /app/output/query_01_output.json after each. The corpus is copied in
# for you at evaluation time; both directories are created if missing, but
# creating them here costs nothing and makes local testing behave identically.
RUN mkdir -p /app/corpus /app/output /app/index

# serve.py stays in the foreground and holds the model. app.py is a client.
CMD ["python3", "/app/serve.py"]
