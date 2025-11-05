FROM python:3.10-slim AS base

# Set common environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install common system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================
# Stage 1: Streamlit App
# ============================================
FROM base AS streamlit

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app/src

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/cache

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================
# Stage 2: Agent API
# ============================================
FROM base AS agent

# Copy agent requirements
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent/ .

# Create necessary directories
RUN mkdir -p /app/runs /app/uploaded_files /app/static

# Verify installation
RUN python -c "import langchain_experimental; print('langchain_experimental imported successfully')"

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 3: Data Fetcher
# ============================================
FROM base AS datafetcher

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed

# Optional: Install cron for scheduled updates
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "scripts/fetch_all_data.py"]

# ============================================
# Default stage (can be overridden at build time)
# ============================================
FROM streamlit AS final