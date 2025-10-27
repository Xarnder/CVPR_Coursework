# Base: Ubuntu + Python + CUDA 12.8 + cuDNN 9 + PyTorch 2.7.0
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Make shell verbose (-e fail fast, -u undefined errors, -x print commands)
SHELL ["/bin/bash", "-lc"]
RUN set -euxo pipefail

# Workdir inside the container
WORKDIR /app

# System deps (FFmpeg, libsndfile) for torchaudio/MP3, etc.
# Use noninteractive to avoid tz prompts; clean apt lists for smaller image
RUN echo "=== Installing system packages (ffmpeg, libsndfile) ===" && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Make pip chatty and faster
ENV PIP_PROGRESS_BAR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python deps (verbose)
RUN echo "=== pip installing from requirements.txt ===" && \
    python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -v -r /app/requirements.txt

# Copy the rest of your source
COPY . /app

# Optional: keep container running for interactive dev
CMD ["tail", "-f", "/dev/null"]
