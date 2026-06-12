FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1 \
    HOME=/data \
    WATTS_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
