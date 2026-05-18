# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Default port for cloud providers
ENV PORT=8080

# Set work directory
WORKDIR /app

# Install system dependencies (required for image processing, OpenCV, and C-compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
# Using --no-cache-dir to keep the image size minimal
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the upload directory exists with proper permissions
RUN mkdir -p static/uploads && chmod 777 static/uploads

# Expose the default port
EXPOSE 8080

# Start Gunicorn WSGI server
# NOTE: We use 1 worker and 4 threads to prevent TensorFlow from multiplying in memory,
# which avoids out-of-memory (OOM) crashes on standard cloud hosts.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 0 app:app
