FROM runpod/serverless:0.5.0-py310
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-u", "handler.py"]