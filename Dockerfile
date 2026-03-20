# Brainfast — Docker image for Linux server deployment
# Build:  docker build -t brainfast .
# Run:    docker run -p 8787:8787 -v /data:/data -v /outputs:/outputs brainfast
#
# Environment variables:
#   BRAINFAST_HEADLESS=1        Disable GUI-dependent features (file browser)
#   BRAINFAST_HOST=0.0.0.0      Bind address (default: 0.0.0.0)
#   BRAINFAST_PORT=8787         Port (default: 8787)

FROM python:3.11-slim

# System dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY project/ ./project/

# Install Python dependencies (core + advanced without Cellpose for image size)
RUN pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir scipy scikit-image openpyxl

# Volumes for data, atlas, and outputs
VOLUME ["/data", "/atlas", "/outputs"]

# Default environment
ENV BRAINFAST_HEADLESS=1 \
    BRAINFAST_HOST=0.0.0.0 \
    BRAINFAST_PORT=8787 \
    PYTHONUNBUFFERED=1

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/api/info')"

CMD ["python", "project/frontend/server.py"]
