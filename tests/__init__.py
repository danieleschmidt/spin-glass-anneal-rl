"""Test suite for Spin-Glass-Anneal-RL."""

import os
import sys
from pathlib import Path

# Add the package to the path for testing
test_dir = Path(__file__).parent
package_dir = test_dir.parent
sys.path.insert(0, str(package_dir))

# Test configuration
TEST_DATA_DIR = test_dir / "data"
TEST_FIXTURES_DIR = test_dir / "fixtures"
TEST_OUTPUT_DIR = test_dir / "output"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_FIXTURES_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test markers
UNIT_TEST = "unit"
INTEGRATION_TEST = "integration"
E2E_TEST = "e2e"
SLOW_TEST = "slow"
GPU_TEST = "gpu"
CUDA_TEST = "cuda"
QUANTUM_TEST = "quantum"
BENCHMARK_TEST = "benchmark"

# Utility functions for dependency checking
def pytest_available():
    """Check if pytest is available."""
    try:
        import pytest
        return True
    except ImportError:
        return False

def numpy_available():
    """Check if numpy is available."""
    try:
        import numpy
        return True
    except ImportError:
        return False

def torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False

def dependencies_available():
    """Check if all required test dependencies are available."""
    return numpy_available() and torch_available()

# Test configuration
SKIP_TESTS_WITHOUT_DEPS = True
VERBOSE_OUTPUT = True