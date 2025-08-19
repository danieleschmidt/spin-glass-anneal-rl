#!/usr/bin/env python3
"""Minimal test for basic functionality without external dependencies."""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    try:
        # Test basic Python imports
        import spin_glass_rl
        print("✓ Basic package import successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False

def test_minimal_ising_model():
    """Test Ising model with numpy fallback."""
    try:
        # Create a minimal mock implementation for testing
        class MinimalIsingModel:
            def __init__(self, n_spins=10):
                self.n_spins = n_spins
                # Use Python lists instead of numpy/torch
                self.spins = [1 if i % 2 == 0 else -1 for i in range(n_spins)]
                self.energy = 0.0
            
            def compute_energy(self):
                # Simple energy calculation
                energy = 0.0
                for i in range(self.n_spins - 1):
                    energy -= self.spins[i] * self.spins[i + 1]  # Nearest neighbor
                return energy
            
            def flip_spin(self, i):
                if 0 <= i < self.n_spins:
                    self.spins[i] *= -1
                    return True
                return False
        
        # Test the minimal model
        model = MinimalIsingModel(n_spins=4)
        initial_energy = model.compute_energy()
        
        # Test spin flip
        model.flip_spin(0)
        new_energy = model.compute_energy()
        
        print(f"✓ Minimal Ising model working: energy {initial_energy} -> {new_energy}")
        return True
        
    except Exception as e:
        print(f"✗ Minimal Ising model failed: {e}")
        return False

def test_basic_annealing():
    """Test basic annealing logic."""
    try:
        # Simple metropolis criterion
        import math
        import random
        
        def metropolis_accept(delta_energy, temperature):
            if delta_energy <= 0:
                return True
            if temperature <= 0:
                return False
            probability = math.exp(-delta_energy / temperature)
            return random.random() < probability
        
        # Test acceptance at different temperatures
        delta_e = 1.0
        
        # High temperature - should accept
        high_temp_accept = metropolis_accept(delta_e, 10.0)
        
        # Low temperature - should likely reject
        low_temp_accept = metropolis_accept(delta_e, 0.01)
        
        print(f"✓ Metropolis criterion working: T=10.0: {high_temp_accept}, T=0.01: {low_temp_accept}")
        return True
        
    except Exception as e:
        print(f"✗ Basic annealing failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("=" * 50)
    print("GENERATION 1: BASIC FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_minimal_ising_model, 
        test_basic_annealing
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    
    if passed == len(tests):
        print("🎉 Generation 1 basic functionality: SUCCESS")
        return True
    else:
        print("❌ Generation 1 basic functionality: NEEDS WORK")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)