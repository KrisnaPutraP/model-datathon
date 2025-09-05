FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/persist/hf \
    TRANSFORMERS_CACHE=/persist/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/persist/hf/hub \
    TORCH_HOME=/persist/torch \
    OFFLOAD_DIR=/persist/offload

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

COPY . .

CMD ["python","-u","handler.py"]
