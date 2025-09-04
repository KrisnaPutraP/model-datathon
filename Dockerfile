# Pakai base image serverless dari GHCR (bukan Docker Hub)
FROM ghcr.io/runpod/serverless:0.5.0-py310
# alternatif yang juga oke:
# FROM ghcr.io/runpod/serverless:latest

WORKDIR /app

# optional: supaya warning "git not found" hilang
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# percepat download model dari HF
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy kode
COPY . .

# start handler RunPod
CMD ["python", "-u", "handler.py"]
