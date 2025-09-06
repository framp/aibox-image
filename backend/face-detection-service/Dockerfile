FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files for dependency installation
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN uv sync --frozen

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    && rm -rf /var/lib/apt/lists/*

# Copy service files
COPY . .

# Create temporary directory for warmup
RUN mkdir -p /tmp

# Expose service port
EXPOSE 5555

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Run the service
CMD ["python", "service.py", "--port", "5555"]