"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

# Import test utilities
from tests import TEST_DATA_DIR, TEST_FIXTURES_DIR, TEST_OUTPUT_DIR


# =============================================================================
# Test Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "cuda: Tests requiring CUDA")
    config.addinivalue_line("markers", "quantum: Tests requiring quantum hardware")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["large", "stress", "benchmark"]):
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Device and Hardware Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the appropriate device (CPU/CUDA) for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_count():
    """Get the number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if torch.cuda.device_count() == 0:
        pytest.skip("No GPU available")


# =============================================================================
# Random Seed Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def fixtures_dir():
    """Get the test fixtures directory."""
    return TEST_FIXTURES_DIR


@pytest.fixture
def output_dir():
    """Get the test output directory."""
    return TEST_OUTPUT_DIR


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.device = "cpu"
    config.batch_size = 32
    config.learning_rate = 0.001
    config.n_epochs = 100
    return config


# =============================================================================
# Problem Instance Fixtures
# =============================================================================

@pytest.fixture
def small_scheduling_problem():
    """Create a small scheduling problem for testing."""
    return {
        "n_agents": 5,
        "n_tasks": 10,
        "n_resources": 3,
        "time_horizon": 20,
        "agent_capacities": [2, 3, 2, 1, 2],
        "task_durations": np.random.randint(1, 5, size=10),
        "task_resources": np.random.randint(0, 3, size=10),
        "task_deadlines": np.random.randint(5, 20, size=10),
    }


@pytest.fixture
def medium_scheduling_problem():
    """Create a medium-sized scheduling problem for testing."""
    return {
        "n_agents": 20,
        "n_tasks": 50,
        "n_resources": 8,
        "time_horizon": 100,
        "agent_capacities": np.random.randint(2, 6, size=20),
        "task_durations": np.random.randint(1, 10, size=50),
        "task_resources": np.random.randint(0, 8, size=50),
        "task_deadlines": np.random.randint(10, 100, size=50),
    }


@pytest.fixture
def large_scheduling_problem():
    """Create a large scheduling problem for testing (marked as slow)."""
    return {
        "n_agents": 100,
        "n_tasks": 500,
        "n_resources": 20,
        "time_horizon": 500,
        "agent_capacities": np.random.randint(3, 10, size=100),
        "task_durations": np.random.randint(1, 20, size=500),
        "task_resources": np.random.randint(0, 20, size=500),
        "task_deadlines": np.random.randint(20, 500, size=500),
    }


# =============================================================================
# Ising Model Fixtures
# =============================================================================

@pytest.fixture
def simple_ising_model():
    """Create a simple Ising model for testing."""
    n_spins = 10
    coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
    coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(coupling_matrix, 0)  # No self-coupling
    
    external_field = np.random.randn(n_spins) * 0.5
    
    return {
        "n_spins": n_spins,
        "coupling_matrix": coupling_matrix,
        "external_field": external_field,
    }


@pytest.fixture
def medium_ising_model():
    """Create a medium-sized Ising model for testing."""
    n_spins = 100
    coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
    coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
    np.fill_diagonal(coupling_matrix, 0)
    
    external_field = np.random.randn(n_spins) * 0.5
    
    return {
        "n_spins": n_spins,
        "coupling_matrix": coupling_matrix,
        "external_field": external_field,
    }


# =============================================================================
# Neural Network Fixtures
# =============================================================================

@pytest.fixture
def simple_policy_network():
    """Create a simple policy network for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
        nn.Tanh()
    )


@pytest.fixture
def simple_value_network():
    """Create a simple value network for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def benchmark_timer():
    """Fixture for timing benchmark tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# =============================================================================
# Memory Testing Fixtures
# =============================================================================

@pytest.fixture
def memory_monitor():
    """Fixture for monitoring memory usage during tests."""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.peak_memory = None
        
        def start(self):
            self.start_memory = self.process.memory_info().rss
            self.peak_memory = self.start_memory
        
        def update(self):
            current_memory = self.process.memory_info().rss
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        
        @property
        def memory_increase(self):
            if self.start_memory is None:
                return None
            current_memory = self.process.memory_info().rss
            return current_memory - self.start_memory
        
        @property
        def peak_memory_increase(self):
            if self.start_memory is None or self.peak_memory is None:
                return None
            return self.peak_memory - self.start_memory
    
    return MemoryMonitor()


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Clean up CUDA cache after each test if CUDA is available."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up any temporary files created during tests."""
    yield
    # Clean up any temporary files in test output directory
    for file in TEST_OUTPUT_DIR.glob("test_*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            import shutil
            shutil.rmtree(file)