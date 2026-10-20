# ROCm-enabled PyTorch base image; inference itself is CPU/local and needs no model download.
FROM rocm/pytorch:rocm6.2_ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app.py README.md ./
RUN mkdir -p /app/corpus /app/index /app/output
ENTRYPOINT ["python", "/app/app.py"]
