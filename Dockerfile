FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY .env.example .
COPY templates/ templates/
COPY static/ static/

# Create directories
RUN mkdir -p static/uploads train test

EXPOSE 5000

ENV FLASK_ENV=production
ENV SECRET_KEY=change-this-to-a-secure-key

CMD ["python", "app.py"]
