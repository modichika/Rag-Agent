# Stage 1: Build stage
FROM python:3.10-slim

WORKDIR /app

# 1. Install CPU-only Torch first to lock the version
# Using --index-url ensures we only get CPU wheels and no NVIDIA blobs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining requirements
# The --extra-index-url allows pip to find CPU torch if required by other packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
