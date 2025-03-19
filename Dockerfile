# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies (add DeadSnakes PPA for Python 3.11 and required tools)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && echo "Installed Python version:" && python3.11 --version \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed and upgraded for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    python3.11 -m pip install --upgrade pip && \
    rm get-pip.py

# Copy requirements file and install Python dependencies, including FastAPI and Uvicorn
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt fastapi uvicorn && \
    echo "Installed Python packages:" && python3.11 -m pip list

# Copy the entire project
COPY . .

# Verify key files are present
RUN ls -la /app && \
    test -f /app/app.py || (echo "ERROR: app.py not found in /app" && exit 1) && \
    test -f /app/utils.py || (echo "ERROR: utils.py not found in /app" && exit 1)

# Expose port for serverless endpoint
EXPOSE 7860

# Set environment variables (optional, can override in RunPod)
ENV PORT=7860

# Run the app with Uvicorn for serverless compatibility
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
