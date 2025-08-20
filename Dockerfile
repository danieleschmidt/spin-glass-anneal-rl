FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY spin_glass_rl/ ./spin_glass_rl/
COPY examples/ ./examples/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from spin_glass_rl import demo_basic_functionality; demo_basic_functionality()" || exit 1

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "-m", "spin_glass_rl.cli", "--host", "0.0.0.0", "--port", "8080"]
