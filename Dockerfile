FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model checkpoints
COPY . /app/

# Ensure generated tunes directory exists
RUN mkdir -p /app/generated_tunes

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
