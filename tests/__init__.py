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

# Create test output directory if it doesn't exist
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