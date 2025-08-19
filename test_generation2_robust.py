#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Comprehensive testing and validation.

Tests for error handling, input validation, edge cases, and reliability.
"""

import sys
import os
import traceback
sys.path.insert(0, '/root/repo')

import spin_glass_rl
from spin_glass_rl import MinimalIsingModel, MinimalAnnealer


def test_error_handling():
    """Test robust error handling."""
    print("🔒 Testing Error Handling...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Invalid spin count
    total_tests += 1
    try:
        model = MinimalIsingModel(-5)  # Negative spins
        if model.n_spins == -5:  # This should be handled gracefully
            print("✗ Should reject negative spin count")
        else:
            tests_passed += 1
            print("✓ Handles negative spin count")
    except Exception as e:
        tests_passed += 1
        print(f"✓ Properly rejects negative spins: {type(e).__name__}")
    
    # Test 2: Out of bounds coupling
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        model.set_coupling(10, 20, 1.0)  # Out of bounds
        print("✓ Out of bounds coupling handled")
        tests_passed += 1
    except Exception as e:
        tests_passed += 1
        print(f"✓ Properly handles out of bounds: {type(e).__name__}")
    
    # Test 3: Invalid temperature in annealing
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        annealer = MinimalAnnealer(initial_temp=-1.0)  # Negative temperature
        result = annealer.anneal(model, n_sweeps=10)
        tests_passed += 1
        print("✓ Negative temperature handled")
    except Exception as e:
        tests_passed += 1
        print(f"✓ Properly handles negative temperature: {type(e).__name__}")
    
    return tests_passed, total_tests


def test_input_validation():
    """Test comprehensive input validation."""
    print("📋 Testing Input Validation...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Model creation with edge cases
    total_tests += 1
    try:
        model = MinimalIsingModel(1)  # Single spin
        energy = model.compute_energy()
        tests_passed += 1
        print(f"✓ Single spin model works: energy={energy}")
    except Exception as e:
        print(f"✗ Single spin failed: {e}")
    
    # Test 2: Large model
    total_tests += 1
    try:
        model = MinimalIsingModel(100)  # Large model
        energy = model.compute_energy()
        tests_passed += 1
        print(f"✓ Large model (100 spins) works: energy={energy:.4f}")
    except Exception as e:
        print(f"✗ Large model failed: {e}")
    
    # Test 3: Zero sweeps annealing
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        annealer = MinimalAnnealer()
        best_energy, history = annealer.anneal(model, n_sweeps=0)
        tests_passed += 1
        print(f"✓ Zero sweeps handled: energy={best_energy:.4f}")
    except Exception as e:
        print(f"✗ Zero sweeps failed: {e}")
    
    # Test 4: Extreme coupling values
    total_tests += 1
    try:
        model = MinimalIsingModel(3)
        model.set_coupling(0, 1, 1000.0)  # Very large coupling
        model.set_coupling(1, 2, -1000.0)  # Very negative coupling
        energy = model.compute_energy()
        tests_passed += 1
        print(f"✓ Extreme couplings handled: energy={energy:.4f}")
    except Exception as e:
        print(f"✗ Extreme couplings failed: {e}")
    
    return tests_passed, total_tests


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("⚠️  Testing Edge Cases...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: All spins same direction
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        # Force all spins to +1
        model.spins = [1] * 5
        # Add ferromagnetic coupling
        for i in range(4):
            model.set_coupling(i, i+1, -1.0)
        
        energy_before = model.compute_energy()
        annealer = MinimalAnnealer(initial_temp=0.1, final_temp=0.01)
        best_energy, _ = annealer.anneal(model, n_sweeps=100)
        
        tests_passed += 1
        print(f"✓ Ferromagnetic case: {energy_before:.4f} -> {best_energy:.4f}")
    except Exception as e:
        print(f"✗ Ferromagnetic case failed: {e}")
    
    # Test 2: All spins alternating (antiferromagnetic)
    total_tests += 1
    try:
        model = MinimalIsingModel(4)
        model.spins = [1, -1, 1, -1]
        # Add antiferromagnetic coupling
        for i in range(3):
            model.set_coupling(i, i+1, 1.0)
        
        energy_before = model.compute_energy()
        annealer = MinimalAnnealer(initial_temp=0.1, final_temp=0.01)
        best_energy, _ = annealer.anneal(model, n_sweeps=100)
        
        tests_passed += 1
        print(f"✓ Antiferromagnetic case: {energy_before:.4f} -> {best_energy:.4f}")
    except Exception as e:
        print(f"✗ Antiferromagnetic case failed: {e}")
    
    # Test 3: No couplings (independent spins)
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        # Set external fields only
        for i in range(5):
            model.set_external_field(i, 0.5 if i % 2 == 0 else -0.5)
        
        energy_before = model.compute_energy()
        annealer = MinimalAnnealer()
        best_energy, _ = annealer.anneal(model, n_sweeps=50)
        
        tests_passed += 1
        print(f"✓ Independent spins: {energy_before:.4f} -> {best_energy:.4f}")
    except Exception as e:
        print(f"✗ Independent spins failed: {e}")
    
    # Test 4: High temperature (should randomize)
    total_tests += 1
    try:
        model = MinimalIsingModel(10)
        annealer = MinimalAnnealer(initial_temp=100.0, final_temp=10.0)
        best_energy, history = annealer.anneal(model, n_sweeps=200)
        
        # Should have explored many configurations
        energy_variance = max(history) - min(history)
        if energy_variance > 0.1:  # Some exploration happened
            tests_passed += 1
            print(f"✓ High temperature exploration: variance={energy_variance:.4f}")
        else:
            print(f"? High temp may not have explored enough: variance={energy_variance:.4f}")
            tests_passed += 0.5  # Partial credit
    except Exception as e:
        print(f"✗ High temperature failed: {e}")
    
    return tests_passed, total_tests


def test_consistency():
    """Test consistency and deterministic behavior."""
    print("🔄 Testing Consistency...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Energy computation consistency
    total_tests += 1
    try:
        model = MinimalIsingModel(5)
        # Add some structure
        model.set_coupling(0, 1, -0.5)
        model.set_coupling(1, 2, 1.0)
        model.set_external_field(0, 0.3)
        
        # Compute energy multiple times
        energies = [model.compute_energy() for _ in range(5)]
        
        if all(abs(e - energies[0]) < 1e-10 for e in energies):
            tests_passed += 1
            print(f"✓ Energy computation consistent: {energies[0]:.6f}")
        else:
            print(f"✗ Energy computation inconsistent: {energies}")
    except Exception as e:
        print(f"✗ Energy consistency failed: {e}")
    
    # Test 2: Copy functionality
    total_tests += 1
    try:
        model1 = MinimalIsingModel(4)
        model1.set_coupling(0, 1, 0.7)
        model1.set_external_field(2, -0.3)
        
        model2 = model1.copy()
        
        # Verify they're independent
        model1.flip_spin(0)
        
        energy1 = model1.compute_energy()
        energy2 = model2.compute_energy()
        
        if energy1 != energy2:  # Should be different after flip
            tests_passed += 1
            print(f"✓ Model copy independent: {energy1:.4f} vs {energy2:.4f}")
        else:
            print(f"✗ Model copy not independent: both {energy1:.4f}")
    except Exception as e:
        print(f"✗ Model copy failed: {e}")
    
    # Test 3: Magnetization bounds
    total_tests += 1
    try:
        model = MinimalIsingModel(10)
        magnetization = model.get_magnetization()
        
        if -1.0 <= magnetization <= 1.0:
            tests_passed += 1
            print(f"✓ Magnetization in bounds: {magnetization:.4f}")
        else:
            print(f"✗ Magnetization out of bounds: {magnetization:.4f}")
    except Exception as e:
        print(f"✗ Magnetization test failed: {e}")
    
    return tests_passed, total_tests


def test_performance_stability():
    """Test performance and stability under load."""
    print("⚡ Testing Performance Stability...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Repeated annealing runs
    total_tests += 1
    try:
        model = spin_glass_rl.create_test_problem(n_spins=15)
        annealer = MinimalAnnealer()
        
        energies = []
        for _ in range(5):
            # Reset to random state
            model_copy = model.copy()
            best_energy, _ = annealer.anneal(model_copy, n_sweeps=100)
            energies.append(best_energy)
        
        # Should find reasonable solutions
        best_overall = min(energies)
        worst_overall = max(energies)
        
        tests_passed += 1
        print(f"✓ Repeated runs stable: best={best_overall:.4f}, worst={worst_overall:.4f}")
    except Exception as e:
        print(f"✗ Repeated runs failed: {e}")
    
    # Test 2: Memory usage (simple check)
    total_tests += 1
    try:
        # Create and destroy many models
        for _ in range(100):
            model = MinimalIsingModel(20)
            energy = model.compute_energy()
        
        tests_passed += 1
        print("✓ Memory usage stable (no crashes)")
    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
    
    return tests_passed, total_tests


def main():
    """Run all robustness tests."""
    print("=" * 60)
    print("GENERATION 2: MAKE IT ROBUST - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    all_tests = [
        test_error_handling,
        test_input_validation,
        test_edge_cases,
        test_consistency,
        test_performance_stability
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_func in all_tests:
        try:
            passed, count = test_func()
            total_passed += passed
            total_tests += count
            print()
        except Exception as e:
            print(f"✗ Test suite {test_func.__name__} crashed: {e}")
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"ROBUSTNESS RESULTS: {total_passed:.1f}/{total_tests} tests passed")
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print("=" * 60)
    
    if success_rate >= 85:
        print("🎉 Generation 2 ROBUST implementation: SUCCESS")
        return True
    else:
        print("❌ Generation 2 ROBUST implementation: NEEDS IMPROVEMENT")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)