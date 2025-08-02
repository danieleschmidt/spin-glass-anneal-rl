# Testing Guide for Spin-Glass-Anneal-RL

This document provides comprehensive guidance on testing within the Spin-Glass-Anneal-RL project.

## Overview

Our testing infrastructure is designed to ensure reliability, performance, and correctness across all components of the system. We use pytest as our primary testing framework with extensive fixtures and custom markers.

## Test Structure

```
tests/
├── __init__.py                 # Test configuration and constants
├── conftest.py                 # Pytest configuration and shared fixtures
├── unit/                       # Unit tests
│   ├── __init__.py
│   └── test_*.py              # Individual unit test files
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_*.py              # Integration test files
├── e2e/                       # End-to-end tests
│   ├── __init__.py
│   └── test_*.py              # E2E test files
├── fixtures/                  # Test fixtures and sample data
│   ├── __init__.py
│   └── sample_problems.py     # Problem instance generators
├── data/                      # Test data files
│   └── .gitkeep
└── output/                    # Test output directory (created automatically)
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions and classes in isolation
- **Location**: `tests/unit/`
- **Marker**: `@pytest.mark.unit`
- **Characteristics**: Fast, isolated, no external dependencies

### Integration Tests
- **Purpose**: Test component interactions and data flow
- **Location**: `tests/integration/`
- **Marker**: `@pytest.mark.integration`
- **Characteristics**: Test multiple components working together

### End-to-End Tests
- **Purpose**: Test complete workflows and user scenarios
- **Location**: `tests/e2e/`
- **Marker**: `@pytest.mark.e2e`
- **Characteristics**: Full system tests, may be slower

## Test Markers

We use pytest markers to categorize and selectively run tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.gpu`: Tests requiring GPU hardware
- `@pytest.mark.cuda`: Tests requiring CUDA
- `@pytest.mark.quantum`: Tests requiring quantum hardware access
- `@pytest.mark.benchmark`: Performance benchmark tests

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m "unit"

# Run all tests except slow ones
pytest -m "not slow"

# Run GPU tests only
pytest -m "gpu"

# Run integration and e2e tests
pytest -m "integration or e2e"

# Run fast unit and integration tests
pytest -m "(unit or integration) and not slow"
```

## Fixtures

### Hardware Fixtures
- `device`: Appropriate device (CPU/CUDA) for testing
- `cuda_available`: Boolean indicating CUDA availability
- `gpu_count`: Number of available GPUs
- `skip_if_no_cuda`: Skip test if CUDA unavailable
- `skip_if_no_gpu`: Skip test if no GPU available

### Problem Instance Fixtures
- `small_scheduling_problem`: Small scheduling problem for quick tests
- `medium_scheduling_problem`: Medium-sized problem
- `large_scheduling_problem`: Large problem (marked as slow)
- `simple_ising_model`: Basic Ising model for testing
- `medium_ising_model`: Larger Ising model

### Neural Network Fixtures
- `simple_policy_network`: Basic policy network
- `simple_value_network`: Basic value network

### Utility Fixtures
- `temp_dir`: Temporary directory for test files
- `mock_logger`: Mock logger for testing
- `benchmark_timer`: Timer for performance testing
- `memory_monitor`: Memory usage monitoring

## Sample Problems

The `tests/fixtures/sample_problems.py` module provides generators for various optimization problems:

```python
from tests.fixtures.sample_problems import get_sample_problem

# Get a small job shop scheduling problem
problem = get_sample_problem("job_shop_scheduling", size="small")

# Get a medium vehicle routing problem
problem = get_sample_problem("vehicle_routing", size="medium")

# Available problem types
from tests.fixtures.sample_problems import list_available_problems
print(list_available_problems())
```

Available problem types:
- `job_shop_scheduling`: Job shop scheduling problems
- `vehicle_routing`: Vehicle routing problems with capacity constraints
- `facility_location`: Facility location optimization
- `max_cut`: Maximum cut problems on graphs
- `quadratic_assignment`: Quadratic assignment problems
- `portfolio_optimization`: Portfolio optimization with risk constraints

## Writing Tests

### Unit Test Example

```python
import pytest
import numpy as np
from spin_glass_rl.core import IsingModel

@pytest.mark.unit
class TestIsingModel:
    def test_energy_calculation(self, simple_ising_model):
        """Test basic energy calculation."""
        model = IsingModel(
            simple_ising_model["coupling_matrix"],
            simple_ising_model["external_field"]
        )
        
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        energy = model.calculate_energy(spins)
        
        assert isinstance(energy, (int, float))
        assert not np.isnan(energy)
        assert not np.isinf(energy)
    
    def test_spin_flip_energy_change(self, simple_ising_model):
        """Test energy change calculation for spin flip."""
        model = IsingModel(
            simple_ising_model["coupling_matrix"],
            simple_ising_model["external_field"]
        )
        
        spins = np.random.choice([-1, 1], size=model.n_spins)
        
        # Calculate energy change for flipping spin 0
        energy_change = model.calculate_energy_change(spins, 0)
        
        # Verify by actual flip
        energy_before = model.calculate_energy(spins)
        spins[0] *= -1
        energy_after = model.calculate_energy(spins)
        actual_change = energy_after - energy_before
        
        assert abs(energy_change - actual_change) < 1e-10
```

### Integration Test Example

```python
@pytest.mark.integration
class TestAnnealingPipeline:
    def test_problem_to_solution_pipeline(self, small_scheduling_problem, device):
        """Test complete annealing pipeline."""
        # Convert problem to Ising model
        ising_model = convert_to_ising(small_scheduling_problem)
        
        # Initialize annealer
        annealer = SimulatedAnnealer(device=device)
        
        # Run annealing
        result = annealer.anneal(ising_model, n_sweeps=100)
        
        # Convert back to schedule
        schedule = convert_to_schedule(result.best_configuration, small_scheduling_problem)
        
        # Validate solution
        assert validate_schedule(schedule, small_scheduling_problem)
```

### GPU Test Example

```python
@pytest.mark.gpu
class TestGPUAnnealing:
    def test_cuda_annealing(self, skip_if_no_cuda, medium_ising_model):
        """Test CUDA-accelerated annealing."""
        device = torch.device("cuda:0")
        
        # Create annealer on GPU
        annealer = CUDAAnnealer(device=device)
        
        # Run annealing
        result = annealer.anneal(medium_ising_model, n_sweeps=1000)
        
        # Verify result is on GPU
        assert result.best_configuration.device == device
        assert torch.isfinite(result.best_energy)
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_annealing_performance(benchmark, medium_ising_model):
    """Benchmark annealing performance."""
    annealer = SimulatedAnnealer()
    
    # Benchmark the annealing process
    result = benchmark(annealer.anneal, medium_ising_model, n_sweeps=1000)
    
    # Assert performance criteria
    assert result.best_energy is not None
```

### Memory Tests

```python
def test_memory_usage(memory_monitor, large_scheduling_problem):
    """Test memory usage during large problem solving."""
    memory_monitor.start()
    
    # Run large problem
    solve_large_problem(large_scheduling_problem)
    
    # Check memory usage
    memory_increase = memory_monitor.memory_increase
    assert memory_increase < 1e9  # Less than 1GB increase
```

## Test Configuration

### pytest.ini Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=spin_glass_rl",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests requiring GPU",
    "cuda: marks tests requiring CUDA",
    "quantum: marks tests requiring quantum hardware",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
    "unit: marks tests as unit tests",
]
```

### Coverage Configuration

Coverage settings are also in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["spin_glass_rl"]
omit = [
    "*/tests/*",
    "*/benchmarks/*",
    "*/examples/*",
    "spin_glass_rl/_version.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_example.py

# Run specific test class
pytest tests/unit/test_example.py::TestBasicFunctionality

# Run specific test method
pytest tests/unit/test_example.py::TestBasicFunctionality::test_numpy_operations
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run tests on 4 cores
pytest -n 4
```

### Test Selection

```bash
# Run fast tests only
pytest -m "not slow"

# Run GPU tests if available
pytest -m "gpu"

# Run specific categories
pytest -m "unit or integration"

# Run tests matching pattern
pytest -k "test_energy"
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=spin_glass_rl --cov-report=html

# View coverage in terminal
pytest --cov=spin_glass_rl --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=spin_glass_rl --cov-fail-under=80
```

## Continuous Integration

Tests are automatically run in CI/CD pipelines. The typical CI workflow includes:

1. **Unit Tests**: Fast tests run on every commit
2. **Integration Tests**: Run on pull requests
3. **GPU Tests**: Run on dedicated GPU runners
4. **Performance Tests**: Run nightly or on releases
5. **Coverage Reports**: Generated and uploaded to coverage services

## Test Data Management

### Test Data Organization

- **Small test data**: Include directly in test files
- **Medium test data**: Use fixtures to generate programmatically
- **Large test data**: Store in `tests/data/` and load as needed
- **External data**: Download and cache during test setup

### Test Data Guidelines

1. **Reproducibility**: Use fixed random seeds
2. **Size limits**: Keep test data reasonably small
3. **Format**: Prefer standard formats (JSON, NPZ, HDF5)
4. **Cleanup**: Clean up temporary data after tests

## Best Practices

### Test Design

1. **Independence**: Tests should not depend on each other
2. **Determinism**: Use fixed random seeds for reproducible results
3. **Clarity**: Test names should clearly describe what is being tested
4. **Focus**: Each test should focus on one specific behavior
5. **Speed**: Keep tests as fast as possible while maintaining coverage

### Fixture Usage

1. **Scope**: Use appropriate fixture scope (function, class, module, session)
2. **Cleanup**: Ensure proper cleanup of resources
3. **Parameterization**: Use parametrized fixtures for testing multiple scenarios
4. **Documentation**: Document complex fixtures clearly

### Mock Usage

1. **External dependencies**: Mock external services and hardware
2. **Expensive operations**: Mock computationally expensive calls
3. **Non-deterministic behavior**: Mock random or time-dependent behavior
4. **Clear boundaries**: Clearly define what is mocked vs real

### Assertion Guidelines

1. **Specific assertions**: Use specific assertion methods (e.g., `assert_allclose` for floats)
2. **Error messages**: Provide helpful error messages for failed assertions
3. **Edge cases**: Test boundary conditions and edge cases
4. **Error conditions**: Test error handling and exception cases

## Debugging Tests

### Common Issues

1. **Random failures**: Check for race conditions and proper seed setting
2. **Platform differences**: Test on multiple platforms and Python versions
3. **Resource leaks**: Monitor memory and GPU memory usage
4. **Slow tests**: Profile slow tests and optimize or mark as slow

### Debugging Tools

```bash
# Run tests with debugging output
pytest -s -v --tb=long

# Run single test with PDB
pytest --pdb tests/unit/test_example.py::test_function

# Capture print statements
pytest -s

# Show local variables in tracebacks
pytest --tb=long --showlocals
```

## Test Maintenance

### Regular Tasks

1. **Update fixtures**: Keep test data current with code changes
2. **Review markers**: Ensure tests are properly categorized
3. **Performance monitoring**: Track test execution times
4. **Coverage analysis**: Identify untested code paths
5. **Dependency updates**: Keep test dependencies current

### Refactoring Tests

1. **Extract common patterns**: Create reusable fixtures and utilities
2. **Remove duplication**: Consolidate similar test logic
3. **Update assertions**: Use more specific assertion methods
4. **Improve naming**: Ensure test names are descriptive and current

This testing guide should help you write effective tests and maintain high code quality in the Spin-Glass-Anneal-RL project.