#!/usr/bin/env python3
"""Minimal test without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from spin_glass_rl.core.ising_model import IsingModelConfig
        print("✓ IsingModelConfig imported")
    except ImportError as e:
        print(f"✗ Failed to import IsingModelConfig: {e}")
        return False
    
    try:
        from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
        print("✓ ScheduleType imported")
    except ImportError as e:
        print(f"✗ Failed to import ScheduleType: {e}")
        return False
    
    try:
        from spin_glass_rl.annealing.gpu_annealer import GPUAnnealerConfig
        print("✓ GPUAnnealerConfig imported")
    except ImportError as e:
        print(f"✗ Failed to import GPUAnnealerConfig: {e}")
        return False
    
    return True

def test_basic_configs():
    """Test basic configuration creation."""
    print("\nTesting basic configurations...")
    
    try:
        from spin_glass_rl.core.ising_model import IsingModelConfig
        config = IsingModelConfig(n_spins=10)
        print(f"✓ Created IsingModelConfig: {config.n_spins} spins")
    except Exception as e:
        print(f"✗ Failed to create IsingModelConfig: {e}")
        return False
    
    try:
        from spin_glass_rl.annealing.gpu_annealer import GPUAnnealerConfig
        from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
        config = GPUAnnealerConfig(
            n_sweeps=100,
            schedule_type=ScheduleType.GEOMETRIC
        )
        print(f"✓ Created GPUAnnealerConfig: {config.n_sweeps} sweeps")
    except Exception as e:
        print(f"✗ Failed to create GPUAnnealerConfig: {e}")
        return False
    
    return True

def main():
    """Run minimal tests."""
    print("="*50)
    print("MINIMAL FUNCTIONALITY TEST")
    print("="*50)
    
    success = True
    success &= test_imports()
    success &= test_basic_configs()
    
    print("\n" + "="*50)
    if success:
        print("✅ MINIMAL TESTS PASSED!")
        print("✓ Core modules can be imported")
        print("✓ Basic configurations work")
        print("\nNote: Full functionality requires torch, numpy, etc.")
    else:
        print("❌ TESTS FAILED!")
    print("="*50)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)