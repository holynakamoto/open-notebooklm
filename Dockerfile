# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies (Python, audio libs, etc.)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Set environment variables (optional, can override in RunPod)
ENV PORT=7860

# Run the app
CMD ["python", "app.py"]
