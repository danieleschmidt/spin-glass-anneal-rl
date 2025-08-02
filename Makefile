# Makefile for Spin-Glass-Anneal-RL
# =============================================================================

.PHONY: help install install-dev test test-fast test-cov lint format clean build docs serve-docs
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip
PYTEST := pytest
PACKAGE_NAME := spin_glass_rl
CUDA_ARCH := 75,80,86  # Common GPU architectures (RTX 20xx, 30xx, 40xx)
BUILD_DIR := build
DIST_DIR := dist
DOCS_DIR := docs
DOCS_BUILD_DIR := $(DOCS_DIR)/_build

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)Spin-Glass-Anneal-RL Development Makefile$(NC)"
	@echo "$(BLUE)==========================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Installation
# =============================================================================

install: ## Install package in development mode
	@echo "$(YELLOW)Installing $(PACKAGE_NAME) in development mode...$(NC)"
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	@echo "$(YELLOW)Installing $(PACKAGE_NAME) with development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	pre-commit install

install-cuda: ## Install package with CUDA dependencies
	@echo "$(YELLOW)Installing $(PACKAGE_NAME) with CUDA dependencies...$(NC)"
	$(PIP) install -e ".[cuda]"

install-all: ## Install package with all optional dependencies
	@echo "$(YELLOW)Installing $(PACKAGE_NAME) with all dependencies...$(NC)"
	$(PIP) install -e ".[all]"

# =============================================================================
# Development Environment
# =============================================================================

setup-env: ## Setup development environment from scratch
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	$(PYTHON) -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	./venv/bin/pip install -e ".[dev]"
	./venv/bin/pre-commit install
	@echo "$(GREEN)Development environment ready! Activate with: source venv/bin/activate$(NC)"

check-cuda: ## Check CUDA availability and GPU information
	@echo "$(YELLOW)Checking CUDA availability...$(NC)"
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); [print(f'  - {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

check-deps: ## Check for missing dependencies
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	$(PIP) check

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linting tools
	@echo "$(YELLOW)Running linting tools...$(NC)"
	ruff check $(PACKAGE_NAME) tests examples benchmarks
	black --check $(PACKAGE_NAME) tests examples benchmarks
	isort --check-only $(PACKAGE_NAME) tests examples benchmarks
	mypy $(PACKAGE_NAME)
	bandit -r $(PACKAGE_NAME) -f json -o bandit-report.json
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	ruff check --fix $(PACKAGE_NAME) tests examples benchmarks
	black $(PACKAGE_NAME) tests examples benchmarks
	isort $(PACKAGE_NAME) tests examples benchmarks
	@echo "$(GREEN)Code formatting completed!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(YELLOW)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

security: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(NC)"
	bandit -r $(PACKAGE_NAME)
	safety check
	@echo "$(GREEN)Security checks completed!$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run full test suite
	@echo "$(YELLOW)Running full test suite...$(NC)"
	$(PYTEST) tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests (exclude slow and GPU tests)
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -m "not slow and not gpu and not cuda" --cov=$(PACKAGE_NAME)

test-gpu: ## Run GPU-specific tests
	@echo "$(YELLOW)Running GPU tests...$(NC)"
	$(PYTEST) tests/ -v -m "gpu or cuda" --cov=$(PACKAGE_NAME)

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) tests/ -v -m "integration" --cov=$(PACKAGE_NAME)

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTEST) tests/ -v -m "unit" --cov=$(PACKAGE_NAME)

test-cov: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-benchmark: ## Run benchmark tests
	@echo "$(YELLOW)Running benchmark tests...$(NC)"
	$(PYTEST) benchmarks/ -v --benchmark-only --benchmark-sort=mean

# =============================================================================
# Build and Distribution
# =============================================================================

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	rm -rf logs profiles
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	@echo "$(GREEN)Cleanup completed!$(NC)"

build: clean ## Build source and wheel distributions
	@echo "$(YELLOW)Building distributions...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build completed! Check $(DIST_DIR)/$(NC)"

build-cuda: ## Build CUDA extensions
	@echo "$(YELLOW)Building CUDA extensions...$(NC)"
	$(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)CUDA extensions built!$(NC)"

install-local: build ## Install from local build
	@echo "$(YELLOW)Installing from local build...$(NC)"
	$(PIP) install $(DIST_DIR)/*.whl --force-reinstall

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(NC)"
	cd $(DOCS_DIR) && make html
	@echo "$(GREEN)Documentation built! Open $(DOCS_BUILD_DIR)/html/index.html$(NC)"

docs-clean: ## Clean documentation build
	@echo "$(YELLOW)Cleaning documentation build...$(NC)"
	cd $(DOCS_DIR) && make clean

docs-live: ## Build and serve documentation with live reload
	@echo "$(YELLOW)Starting documentation server with live reload...$(NC)"
	sphinx-autobuild $(DOCS_DIR) $(DOCS_BUILD_DIR)/html --port 8000 --open-browser

serve-docs: docs ## Serve built documentation locally
	@echo "$(YELLOW)Serving documentation at http://localhost:8080$(NC)"
	cd $(DOCS_BUILD_DIR)/html && $(PYTHON) -m http.server 8080

# =============================================================================
# Development Tasks
# =============================================================================

jupyter: ## Start Jupyter Lab
	@echo "$(YELLOW)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard: ## Start TensorBoard
	@echo "$(YELLOW)Starting TensorBoard...$(NC)"
	tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006

profile: ## Run profiling on example script
	@echo "$(YELLOW)Running profiler...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats examples/basic_scheduling.py
	snakeviz profile.stats

memory-profile: ## Run memory profiling
	@echo "$(YELLOW)Running memory profiler...$(NC)"
	mprof run examples/basic_scheduling.py
	mprof plot

# =============================================================================
# Benchmarking and Performance
# =============================================================================

benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	$(PYTHON) benchmarks/run_benchmarks.py --output=benchmark_results.json
	@echo "$(GREEN)Benchmark results saved to benchmark_results.json$(NC)"

benchmark-gpu: ## Run GPU-specific benchmarks
	@echo "$(YELLOW)Running GPU benchmarks...$(NC)"
	$(PYTHON) benchmarks/gpu_benchmarks.py --output=gpu_benchmark_results.json

stress-test: ## Run stress tests
	@echo "$(YELLOW)Running stress tests...$(NC)"
	$(PYTHON) tests/stress_tests/run_stress_tests.py

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	docker build -t $(PACKAGE_NAME):latest .

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(NC)"
	docker run --rm -it --gpus all -p 8888:8888 -p 6006:6006 $(PACKAGE_NAME):latest

docker-dev: ## Run Docker container in development mode
	@echo "$(YELLOW)Running Docker container in development mode...$(NC)"
	docker run --rm -it --gpus all -v $(PWD):/workspace -p 8888:8888 -p 6006:6006 $(PACKAGE_NAME):latest

# =============================================================================
# Release Management
# =============================================================================

bump-patch: ## Bump patch version
	@echo "$(YELLOW)Bumping patch version...$(NC)"
	bump2version patch

bump-minor: ## Bump minor version
	@echo "$(YELLOW)Bumping minor version...$(NC)"
	bump2version minor

bump-major: ## Bump major version
	@echo "$(YELLOW)Bumping major version...$(NC)"
	bump2version major

tag-release: ## Create release tag
	@echo "$(YELLOW)Creating release tag...$(NC)"
	git tag -a v$(shell python -c "from $(PACKAGE_NAME)._version import __version__; print(__version__)") -m "Release v$(shell python -c "from $(PACKAGE_NAME)._version import __version__; print(__version__)")"

# =============================================================================
# Utility Tasks
# =============================================================================

check: lint test ## Run linting and tests
	@echo "$(GREEN)All checks passed!$(NC)"

ci: lint test-cov security ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed!$(NC)"

deps-update: ## Update dependencies
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	$(PIP) list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U
	@echo "$(GREEN)Dependencies updated!$(NC)"

size: ## Show package size breakdown
	@echo "$(YELLOW)Package size breakdown:$(NC)"
	du -sh $(PACKAGE_NAME)/*

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "$(BLUE)==================$(NC)"
	@echo "Package: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
	@echo "CUDA available: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed')"

# =============================================================================
# Quick Commands
# =============================================================================

dev: install-dev ## Quick development setup
	@echo "$(GREEN)Development environment ready!$(NC)"

test-quick: test-fast ## Quick test run (alias for test-fast)

all: clean install-dev lint test build docs ## Run everything