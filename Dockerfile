# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && . ~/.bashrc

# Copy requirements first to leverage Docker cache
COPY pyproject.toml .

# Install Python dependencies using uv
RUN . ~/.bashrc && uv pip install -r pyproject.toml --system

# Copy the rest of the application
COPY . .

# Create directories for uploads and embeddings if they don't exist
RUN mkdir -p uploads embeddings

# Expose the port the app runs on
EXPOSE 7860

# Change to rag_demo directory and run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 