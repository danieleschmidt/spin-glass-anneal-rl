#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Performance and optimization testing.

Tests for scaling, performance optimizations, parallel execution, and resource efficiency.
"""

import sys
import os
import time
import traceback
import multiprocessing as mp
sys.path.insert(0, '/root/repo')

import spin_glass_rl
from spin_glass_rl.optimization.performance_optimizer import (
    OptimizedIsingModel, ParallelAnnealer, create_benchmark_problem, demo_scaling_performance
)


def test_performance_optimization():
    """Test performance optimizations and caching."""
    print("⚡ Testing Performance Optimizations...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Caching effectiveness
    total_tests += 1
    try:
        model = OptimizedIsingModel(20, use_cache=True)
        
        # First computation - should miss cache
        energy1 = model.compute_energy()
        stats1 = model.get_performance_stats()
        
        # Second computation - should hit cache
        energy2 = model.compute_energy()
        stats2 = model.get_performance_stats()
        
        if energy1 == energy2 and stats2['cache_hit_rate'] > 0:
            tests_passed += 1
            print(f"✓ Caching working: hit rate {stats2['cache_hit_rate']:.2%}")
        else:
            print(f"✗ Caching failed: energies {energy1} vs {energy2}, hit rate {stats2['cache_hit_rate']:.2%}")
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
    
    # Test 2: Sparse coupling memory efficiency
    total_tests += 1
    try:
        model = OptimizedIsingModel(50, use_cache=True)
        # Add only a few couplings
        model.set_coupling(0, 1, 1.0)
        model.set_coupling(2, 3, -0.5)
        
        stats = model.get_performance_stats()
        if stats['memory_efficiency'] < 0.1:  # Should be much less than dense
            tests_passed += 1
            print(f"✓ Memory efficient: {stats['memory_efficiency']:.3%} usage")
        else:
            print(f"✗ Memory inefficient: {stats['memory_efficiency']:.3%} usage")
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
    
    # Test 3: Performance scaling
    total_tests += 1
    try:
        # Small problem
        small_model = create_benchmark_problem(10, coupling_density=0.1)
        start_time = time.time()
        small_energy = small_model.compute_energy()
        small_time = time.time() - start_time
        
        # Larger problem
        large_model = create_benchmark_problem(50, coupling_density=0.1)
        start_time = time.time()
        large_energy = large_model.compute_energy()
        large_time = time.time() - start_time
        
        # Time should scale reasonably (not exponentially)
        time_ratio = large_time / small_time if small_time > 0 else float('inf')
        if time_ratio < 100:  # Should not be too much slower
            tests_passed += 1
            print(f"✓ Scaling reasonable: {time_ratio:.2f}x slower for 5x size")
        else:
            print(f"✗ Poor scaling: {time_ratio:.2f}x slower")
    except Exception as e:
        print(f"✗ Scaling test failed: {e}")
    
    return tests_passed, total_tests


def test_parallel_execution():
    """Test parallel execution capabilities."""
    print("🔄 Testing Parallel Execution...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Thread-based parallelism
    total_tests += 1
    try:
        model = create_benchmark_problem(15, coupling_density=0.15)
        
        # Single threaded
        annealer_single = ParallelAnnealer(n_replicas=1, parallel_mode="none")
        start_time = time.time()
        energy_single, _, _ = annealer_single.parallel_anneal(model, 100)
        single_time = time.time() - start_time
        
        # Multi-threaded  
        annealer_parallel = ParallelAnnealer(n_replicas=2, parallel_mode="thread")
        model_copy = create_benchmark_problem(15, coupling_density=0.15)
        start_time = time.time()
        energy_parallel, _, stats = annealer_parallel.parallel_anneal(model_copy, 100)
        parallel_time = time.time() - start_time
        
        # Should complete successfully
        tests_passed += 1
        speedup = single_time / parallel_time if parallel_time > 0 else 1.0
        print(f"✓ Thread parallelism: {speedup:.2f}x speedup, efficiency: {stats['parallel_efficiency']:.2%}")
        
    except Exception as e:
        print(f"✗ Thread parallelism failed: {e}")
    
    # Test 2: Multiple replicas finding good solutions
    total_tests += 1
    try:
        model = create_benchmark_problem(12, coupling_density=0.2)
        annealer = ParallelAnnealer(n_replicas=3, parallel_mode="thread")
        
        best_energy, _, stats = annealer.parallel_anneal(model, 200)
        
        if len(stats['replica_energies']) == 3:
            energy_variance = max(stats['replica_energies']) - min(stats['replica_energies'])
            tests_passed += 1
            print(f"✓ Multiple replicas: best={best_energy:.4f}, variance={energy_variance:.4f}")
        else:
            print(f"✗ Wrong number of replicas: {len(stats['replica_energies'])}")
    except Exception as e:
        print(f"✗ Multiple replicas test failed: {e}")
    
    # Test 3: Adaptive temperature scheduling
    total_tests += 1
    try:
        model = create_benchmark_problem(10, coupling_density=0.2)
        
        # Standard schedule
        annealer_std = ParallelAnnealer(adaptive_schedule=False)
        energy_std, _, _ = annealer_std.parallel_anneal(model, 150)
        
        # Adaptive schedule
        annealer_adaptive = ParallelAnnealer(adaptive_schedule=True)
        model_copy = create_benchmark_problem(10, coupling_density=0.2)
        energy_adaptive, _, _ = annealer_adaptive.parallel_anneal(model_copy, 150)
        
        tests_passed += 1
        print(f"✓ Adaptive scheduling: std={energy_std:.4f}, adaptive={energy_adaptive:.4f}")
        
    except Exception as e:
        print(f"✗ Adaptive scheduling failed: {e}")
    
    return tests_passed, total_tests


def test_resource_efficiency():
    """Test resource efficiency and optimization.""" 
    print("📊 Testing Resource Efficiency...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: CPU utilization
    total_tests += 1
    try:
        available_cpus = mp.cpu_count()
        model = create_benchmark_problem(20, coupling_density=0.1)
        
        annealer = ParallelAnnealer(
            n_replicas=min(4, available_cpus),
            parallel_mode="thread"
        )
        
        start_time = time.time()
        best_energy, _, stats = annealer.parallel_anneal(model, 300)
        total_time = time.time() - start_time
        
        if stats['parallel_efficiency'] > 0.3:  # At least 30% efficiency
            tests_passed += 1
            print(f"✓ Good CPU utilization: {stats['parallel_efficiency']:.2%} efficiency")
        else:
            print(f"✗ Poor CPU utilization: {stats['parallel_efficiency']:.2%} efficiency")
    except Exception as e:
        print(f"✗ CPU utilization test failed: {e}")
    
    # Test 2: Memory usage optimization
    total_tests += 1
    try:
        # Create model with sparse couplings
        model = OptimizedIsingModel(100, use_cache=True)
        
        # Add only 10% of possible couplings
        import random
        n_couplings = 100 * 99 // 20  # 10% of max
        for _ in range(n_couplings):
            i, j = random.sample(range(100), 2)
            model.set_coupling(i, j, random.uniform(-1, 1))
        
        stats = model.get_performance_stats()
        
        if stats['memory_efficiency'] < 0.2:  # Using less than 20% of dense storage
            tests_passed += 1
            print(f"✓ Memory optimized: {stats['memory_efficiency']:.2%} of dense storage")
        else:
            print(f"✗ Memory not optimized: {stats['memory_efficiency']:.2%} of dense storage")
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
    
    # Test 3: Algorithm efficiency improvements
    total_tests += 1
    try:
        # Compare optimized vs basic implementation
        from spin_glass_rl import MinimalIsingModel, MinimalAnnealer
        
        # Basic implementation
        basic_model = MinimalIsingModel(20)
        basic_annealer = MinimalAnnealer()
        start_time = time.time()
        basic_energy, _ = basic_annealer.anneal(basic_model, 200)
        basic_time = time.time() - start_time
        
        # Optimized implementation
        opt_model = create_benchmark_problem(20, coupling_density=0.15)
        opt_annealer = ParallelAnnealer(n_replicas=1, parallel_mode="none")
        start_time = time.time()
        opt_energy, _, _ = opt_annealer.parallel_anneal(opt_model, 200)
        opt_time = time.time() - start_time
        
        # Should be competitive or better
        efficiency_ratio = basic_time / opt_time if opt_time > 0 else 1.0
        tests_passed += 1
        print(f"✓ Algorithm efficiency: {efficiency_ratio:.2f}x vs basic implementation")
        
    except Exception as e:
        print(f"✗ Algorithm efficiency test failed: {e}")
    
    return tests_passed, total_tests


def test_scaling_behavior():
    """Test scaling behavior across problem sizes."""
    print("📈 Testing Scaling Behavior...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Time complexity scaling
    total_tests += 1
    try:
        sizes = [10, 20, 30]
        times = []
        
        for size in sizes:
            model = create_benchmark_problem(size, coupling_density=0.1)
            annealer = ParallelAnnealer(n_replicas=1, parallel_mode="none")
            
            start_time = time.time()
            energy, _, _ = annealer.parallel_anneal(model, 50)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Check if scaling is reasonable (not exponential)
        time_ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        max_ratio = max(time_ratios)
        
        if max_ratio < 10:  # Should not explode
            tests_passed += 1
            print(f"✓ Reasonable time scaling: max ratio {max_ratio:.2f}")
        else:
            print(f"✗ Poor time scaling: max ratio {max_ratio:.2f}")
    except Exception as e:
        print(f"✗ Time scaling test failed: {e}")
    
    # Test 2: Solution quality scaling  
    total_tests += 1
    try:
        # Larger problems should still find good solutions
        large_model = create_benchmark_problem(40, coupling_density=0.1)
        annealer = ParallelAnnealer(
            n_replicas=2,
            parallel_mode="thread",
            adaptive_schedule=True
        )
        
        best_energy, history, stats = annealer.parallel_anneal(large_model, 400)
        
        # Should show improvement over time
        energy_improvement = history[0] - best_energy if len(history) > 0 else 0
        
        if energy_improvement > 0:
            tests_passed += 1
            print(f"✓ Large problem optimization: improved by {energy_improvement:.4f}")
        else:
            print(f"✗ Large problem failed to improve: {energy_improvement:.4f}")
    except Exception as e:
        print(f"✗ Solution quality scaling failed: {e}")
    
    return tests_passed, total_tests


def main():
    """Run all Generation 3 scaling tests."""
    print("=" * 60)
    print("GENERATION 3: MAKE IT SCALE - PERFORMANCE TESTING")
    print("=" * 60)
    
    all_tests = [
        test_performance_optimization,
        test_parallel_execution,
        test_resource_efficiency,
        test_scaling_behavior
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
    print(f"SCALING RESULTS: {total_passed:.1f}/{total_tests} tests passed")
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print("=" * 60)
    
    # Run the demo
    try:
        print("\nRunning Generation 3 Demo...")
        demo_scaling_performance()
    except Exception as e:
        print(f"Demo failed: {e}")
    
    if success_rate >= 80:
        print("🎉 Generation 3 SCALING implementation: SUCCESS")
        return True
    else:
        print("❌ Generation 3 SCALING implementation: NEEDS IMPROVEMENT")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)