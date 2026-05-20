# ============================================
# Dockerfile - Skin Cancer Classifier
# ============================================
FROM python:3.11-slim

LABEL maintainer="Skin Cancer Classifier Project"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente y modelo
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
