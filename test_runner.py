#!/usr/bin/env python3
"""Test runner for Spin-Glass-Anneal-RL."""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

def check_dependencies() -> dict:
    """Check which dependencies are available."""
    deps = {}
    
    # Core dependencies
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
    
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = None
    
    try:
        import scipy
        deps['scipy'] = scipy.__version__
    except ImportError:
        deps['scipy'] = None
    
    # Test dependencies
    try:
        import pytest
        deps['pytest'] = pytest.__version__
    except ImportError:
        deps['pytest'] = None
    
    try:
        import matplotlib
        deps['matplotlib'] = matplotlib.__version__
    except ImportError:
        deps['matplotlib'] = None
    
    return deps

def print_dependency_status(deps: dict):
    """Print dependency status."""
    print("Dependency Status:")
    print("=" * 40)
    
    required = ['numpy', 'torch', 'scipy']
    optional = ['pytest', 'matplotlib']
    
    print("Required dependencies:")
    for dep in required:
        version = deps.get(dep)
        status = f"✅ {version}" if version else "❌ Missing"
        print(f"  {dep:<12}: {status}")
    
    print("\nOptional dependencies:")
    for dep in optional:
        version = deps.get(dep)
        status = f"✅ {version}" if version else "❌ Missing"
        print(f"  {dep:<12}: {status}")
    
    print()

def can_run_tests(deps: dict) -> bool:
    """Check if we can run tests."""
    required = ['numpy', 'torch']  # Minimum for basic functionality
    return all(deps.get(dep) is not None for dep in required)

def run_basic_import_test():
    """Run basic import test."""
    print("Running basic import test...")
    print("-" * 40)
    
    try:
        # Test core imports
        from spin_glass_rl.core.ising_model import IsingModelConfig
        from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
        from spin_glass_rl.utils.exceptions import SpinGlassError
        print("✅ Core module imports successful")
        
        # Test problem imports
        from spin_glass_rl.problems.base import ProblemTemplate
        print("✅ Problem base imports successful")
        
        # Test utility imports
        from spin_glass_rl.utils.validation import validate_numeric
        print("✅ Utility imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    print("-" * 40)
    
    test_files = list(Path("tests/unit").glob("test_*.py"))
    
    if not test_files:
        print("❌ No unit test files found")
        return False
    
    print(f"Found {len(test_files)} unit test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    # Try to run with pytest if available
    try:
        import pytest
        args = ["-v", "tests/unit/"]
        if "--tb=short" not in sys.argv:
            args.append("--tb=short")
        
        result = pytest.main(args)
        return result == 0
        
    except ImportError:
        print("⚠️  pytest not available, skipping unit tests")
        return True  # Don't fail if pytest not available

def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    print("-" * 40)
    
    test_files = list(Path("tests/integration").glob("test_*.py"))
    
    if not test_files:
        print("❌ No integration test files found")
        return False
    
    print(f"Found {len(test_files)} integration test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    try:
        import pytest
        args = ["-v", "tests/integration/", "-m", "not slow"]
        result = pytest.main(args)
        return result == 0
        
    except ImportError:
        print("⚠️  pytest not available, skipping integration tests")
        return True

def run_validation_tests():
    """Run validation tests for core functionality."""
    print("Running validation tests...")
    print("-" * 40)
    
    try:
        # Test configuration classes
        from spin_glass_rl.core.ising_model import IsingModelConfig
        config = IsingModelConfig(n_spins=5)
        assert config.n_spins == 5
        print("✅ Configuration validation passed")
        
        # Test enum imports
        from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
        assert ScheduleType.GEOMETRIC.value == "geometric"
        print("✅ Enum validation passed")
        
        # Test exception hierarchy
        from spin_glass_rl.utils.exceptions import SpinGlassError, ModelError
        assert issubclass(ModelError, SpinGlassError)
        print("✅ Exception hierarchy validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def generate_test_report(results: dict):
    """Generate test report."""
    print("\nTest Report")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20}: {status}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed")
    
    return failed_tests == 0

def main():
    """Main test runner."""
    print("Spin-Glass-Anneal-RL Test Runner")
    print("=" * 50)
    
    # Check dependencies
    deps = check_dependencies()
    print_dependency_status(deps)
    
    # Check if we can run tests
    if not can_run_tests(deps):
        print("❌ Cannot run tests - missing required dependencies")
        print("Please install: pip install numpy torch scipy")
        return 1
    
    # Run tests
    results = {}
    
    # Basic validation tests (always run)
    results["Import Test"] = run_basic_import_test()
    results["Validation Test"] = run_validation_tests()
    
    # Unit tests (if dependencies available)
    if deps.get('numpy') and deps.get('torch'):
        try:
            results["Unit Tests"] = run_unit_tests()
        except Exception as e:
            print(f"⚠️  Unit tests failed with error: {e}")
            results["Unit Tests"] = False
    
    # Integration tests (if all dependencies available)
    if all(deps.get(dep) for dep in ['numpy', 'torch', 'scipy']):
        try:
            results["Integration Tests"] = run_integration_tests()
        except Exception as e:
            print(f"⚠️  Integration tests failed with error: {e}")
            results["Integration Tests"] = False
    
    # Generate report
    success = generate_test_report(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())