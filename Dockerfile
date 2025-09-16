FROM python:3.11-slim

# System deps (images, pdf2image, fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev poppler-utils fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

ENV MODEL_WEIGHTS=/app/models/efficientnet_lite0_20250709_2253/best_model.pth
ENV THRESHOLD=0.7753

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and models into container
COPY . .

# Ensure Python can find both app/ and ecg/
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
