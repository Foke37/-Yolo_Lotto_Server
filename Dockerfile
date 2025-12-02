FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=/usr/bin/tesseract

# System deps for OpenCV + Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

CMD ["python", "ws_ocr_server.py"]
