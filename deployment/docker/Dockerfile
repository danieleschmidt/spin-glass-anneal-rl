# Progressive Quality Gates Production Dockerfile
FROM python:3.11-slim

# Runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1001 pqg \
    && useradd -r -u 1001 -g pqg pqg

# Application code
WORKDIR /app
COPY progressive_quality_gates*.py .
COPY spin_glass_rl/ ./spin_glass_rl/ || true

# Install Python dependencies
RUN pip install --no-cache-dir numpy scipy psutil

# Security hardening
RUN chown -R pqg:pqg /app
USER pqg

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entry point
CMD ["python", "progressive_quality_gates_optimized.py"]
