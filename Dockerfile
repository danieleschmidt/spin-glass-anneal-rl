# Multi-stage Dockerfile for Spin-Glass-Anneal-RL
# Optimized for CUDA development and production deployment

# =============================================================================
# Base Stage: CUDA runtime with Python
# =============================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and development tools
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # System utilities
    htop \
    vim \
    nano \
    tmux \
    screen \
    # Networking
    openssh-client \
    # Development tools
    gdb \
    valgrind \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Development Stage: Full development environment
# =============================================================================
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Development tools
    clang-format \
    shellcheck \
    # Debugging tools
    strace \
    ltrace \
    # Performance tools
    perf-tools-unstable \
    # Documentation tools
    pandoc \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for some development tools
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd -m -s /bin/bash -u 1000 developer && \
    usermod -aG sudo developer && \
    echo "developer:developer" | chpasswd

# Set up workspace
WORKDIR /workspace
COPY requirements-dev.txt .

# Install Python development dependencies
RUN pip install -r requirements-dev.txt

# Switch to developer user
USER developer

# Set up shell environment
RUN echo 'export PATH="/home/developer/.local/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc

# Install pre-commit for the user
RUN pip install --user pre-commit

# =============================================================================
# Build Stage: Application building
# =============================================================================
FROM base as builder

# Install build dependencies
WORKDIR /build

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY spin_glass_rl/ spin_glass_rl/

# Install build dependencies
RUN pip install build wheel

# Build the package
RUN python -m build

# =============================================================================
# Test Stage: Testing environment
# =============================================================================
FROM development as test

# Copy source code and tests
COPY . .

# Install package in development mode
RUN pip install --user -e ".[dev]"

# Run tests by default
CMD ["pytest", "-v", "--cov=spin_glass_rl", "--cov-report=html", "--cov-report=term-missing"]

# =============================================================================
# Runtime Stage: Minimal production image
# =============================================================================
FROM base as runtime

# Install runtime dependencies only
RUN pip install \
    numpy \
    scipy \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Create app user
RUN useradd -m -s /bin/bash -u 1001 appuser

# Set up application directory
WORKDIR /app
RUN chown appuser:appuser /app

# Copy built package from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the package
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Switch to app user
USER appuser

# Set up environment
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PATH="/home/appuser/.local/bin:$PATH"

# Default command
CMD ["python", "-c", "import spin_glass_rl; print(f'Spin-Glass-Anneal-RL {spin_glass_rl.__version__} ready!')"]

# =============================================================================
# Production Stage: Optimized production image
# =============================================================================
FROM runtime as production

# Copy application code
COPY --chown=appuser:appuser spin_glass_rl/ /app/spin_glass_rl/
COPY --chown=appuser:appuser examples/ /app/examples/

# Install additional production dependencies
USER root
RUN pip install \
    gunicorn \
    uvicorn[standard] \
    prometheus-client \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        supervisor \
        nginx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy production configuration
COPY docker/supervisor.conf /etc/supervisor/conf.d/
COPY docker/nginx.conf /etc/nginx/sites-available/default

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spin_glass_rl; print('OK')" || exit 1

# Production command
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# =============================================================================
# Jupyter Stage: Interactive development with Jupyter
# =============================================================================
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install --user \
    jupyterlab \
    jupyter \
    ipywidgets \
    matplotlib \
    plotly \
    seaborn \
    tensorboard \
    wandb

# Copy notebooks
COPY --chown=developer:developer notebooks/ /workspace/notebooks/
COPY --chown=developer:developer examples/ /workspace/examples/

# Expose Jupyter port
EXPOSE 8888

# Jupyter configuration
RUN mkdir -p ~/.jupyter && \
    echo "c.ServerApp.token = ''" > ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_server_config.py

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Benchmarking Stage: Performance testing
# =============================================================================
FROM runtime as benchmark

# Install benchmarking tools
USER root
RUN pip install \
    pytest-benchmark \
    memory-profiler \
    py-spy \
    psutil \
    line-profiler

USER appuser

# Copy benchmarks
COPY --chown=appuser:appuser benchmarks/ /app/benchmarks/
COPY --chown=appuser:appuser tests/ /app/tests/

# Run benchmarks by default
CMD ["python", "-m", "pytest", "benchmarks/", "--benchmark-only", "--benchmark-sort=mean"]

# =============================================================================
# Documentation Stage: Documentation building
# =============================================================================
FROM base as docs

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autoapi \
    sphinx-copybutton

# Copy documentation source
WORKDIR /docs
COPY docs/ .
COPY spin_glass_rl/ ../spin_glass_rl/
COPY README.md .

# Build documentation
RUN make html

# Serve documentation
FROM nginx:alpine as docs-server
COPY --from=docs /docs/_build/html /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]