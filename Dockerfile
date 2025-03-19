# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies (add DeadSnakes PPA for Python 3.11)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed and upgraded for Python 3.11
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Set environment variables (optional, can override in RunPod)
ENV PORT=7860

# Run the app
CMD ["python3.11", "app.py"]
