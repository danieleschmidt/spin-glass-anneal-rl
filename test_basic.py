#!/usr/bin/env python3
"""Basic test to verify spin-glass-anneal-rl works."""

import sys
import torch
import numpy as np

def test_basic_ising_model():
    """Test basic Ising model functionality."""
    print("Testing basic Ising model...")
    
    try:
        from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
        
        config = IsingModelConfig(n_spins=10, device="cpu")
        model = IsingModel(config)
        
        # Add some couplings
        model.set_coupling(0, 1, -1.0)
        model.set_coupling(1, 2, -1.0)
        
        # Set external field
        model.set_external_field(0, 0.5)
        
        # Compute energy
        energy = model.compute_energy()
        print(f"✓ Ising model created, energy: {energy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ising model test failed: {e}")
        return False

def test_basic_annealing():
    """Test basic annealing functionality."""
    print("Testing basic annealing...")
    
    try:
        from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
        from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
        
        # Create simple model
        config = IsingModelConfig(n_spins=20, device="cpu")
        model = IsingModel(config)
        
        # Add random couplings
        for i in range(10):
            j = (i + 1) % 20
            model.set_coupling(i, j, -1.0)
        
        initial_energy = model.compute_energy()
        
        # Configure annealer  
        annealer_config = GPUAnnealerConfig(
            n_sweeps=500,
            initial_temp=2.0,
            final_temp=0.01,
            random_seed=42
        )
        
        annealer = GPUAnnealer(annealer_config)
        result = annealer.anneal(model)
        
        print(f"✓ Annealing completed")
        print(f"  Initial energy: {initial_energy:.4f}")
        print(f"  Final energy: {result.best_energy:.4f}")
        print(f"  Improvement: {initial_energy - result.best_energy:.4f}")
        print(f"  Time: {result.total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Annealing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduling_problem():
    """Test scheduling problem functionality."""
    print("Testing scheduling problem...")
    
    try:
        from spin_glass_rl.problems.scheduling import SchedulingProblem, Task, Agent
        
        scheduler = SchedulingProblem()
        
        # Add simple tasks and agents
        tasks = [
            Task(id=0, duration=5.0),
            Task(id=1, duration=3.0),
            Task(id=2, duration=4.0),
        ]
        
        for task in tasks:
            scheduler.add_task(task)
        
        agents = [
            Agent(id=0, name="Agent_A"),
            Agent(id=1, name="Agent_B"),
        ]
        
        for agent in agents:
            scheduler.add_agent(agent)
        
        print(f"✓ Scheduling problem created with {len(tasks)} tasks and {len(agents)} agents")
        
        return True
        
    except Exception as e:
        print(f"✗ Scheduling problem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("Spin-Glass-Anneal-RL Basic Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_ising_model,
        test_basic_annealing,
        test_scheduling_problem,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())