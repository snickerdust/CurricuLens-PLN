FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# ── Step 1: Install torch CPU-only first (biggest layer, most cacheable) ──
# Using PyTorch CPU index so we get ~600MB wheel instead of ~2GB GPU wheel
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Copy requirements and install remaining deps ──
COPY --chown=user backend/requirements.txt .
# Install everything except torch (already installed above)
RUN grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Pre-download NLTK resources
RUN python -m nltk.downloader punkt stopwords

# Copy the rest of the application
COPY --chown=user . .

# Change working directory to backend where app.py is located
WORKDIR $HOME/app/backend

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "app:app"]
