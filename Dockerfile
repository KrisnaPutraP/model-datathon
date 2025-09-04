FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
COPY . .
CMD ["python","-u","handler.py"]
