FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for the health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create and copy PDF documents into the expected path
# Note: The local folder 'documents' must exist and contain the PDFs
RUN mkdir -p /app/documents
COPY documents/example_post.pdf /app/documents/
COPY documents/company_description.pdf /app/documents/

# Expose the API port
EXPOSE 8080

# Health check 
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]