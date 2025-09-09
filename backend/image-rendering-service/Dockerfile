FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY service.py ./

# Install uv for fast dependency management
RUN pip install uv

# Install dependencies
RUN uv pip install --system -e .

# Expose the service port
EXPOSE 5557

# Run the service
CMD ["python", "service.py", "--port", "5557"]
