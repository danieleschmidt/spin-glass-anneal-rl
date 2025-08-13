#!/usr/bin/env python3
"""Basic test for Generation 1 functionality."""

import torch
import numpy as np
from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.coupling_matrix import CouplingMatrix
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType

def test_basic_ising_model():
    """Test basic Ising model functionality."""
    print("Testing basic Ising model...")
    
    # Create small model
    config = IsingModelConfig(
        n_spins=10,
        coupling_strength=1.0,
        external_field_strength=0.5,
        use_sparse=False,  # Use dense for simplicity
        device="cpu"
    )
    
    model = IsingModel(config)
    print(f"✓ Created Ising model with {model.n_spins} spins")
    
    # Add some couplings
    for i in range(model.n_spins - 1):
        model.set_coupling(i, i + 1, -1.0)  # Ferromagnetic chain
    
    # Compute initial energy
    initial_energy = model.compute_energy()
    print(f"✓ Initial energy: {initial_energy:.6f}")
    
    # Test spin flip
    old_energy = model.compute_energy()
    delta_energy = model.flip_spin(0)
    new_energy = model.compute_energy()
    
    expected_delta = new_energy - old_energy
    assert abs(delta_energy - expected_delta) < 1e-6, "Energy delta mismatch"
    print(f"✓ Spin flip test passed (ΔE = {delta_energy:.6f})")
    
    return model

def test_coupling_matrix():
    """Test coupling matrix functionality."""
    print("\nTesting coupling matrix...")
    
    coupling_matrix = CouplingMatrix(10, use_sparse=False)
    
    # Test random graph pattern
    coupling_matrix.generate_pattern(
        "random_graph",
        strength_range=(-1.0, 1.0),
        edge_probability=0.3
    )
    
    stats = coupling_matrix.get_connectivity_stats()
    print(f"✓ Generated random graph: {stats['n_couplings']} edges, density={stats['density']:.3f}")
    
    # Test nearest neighbor pattern
    coupling_matrix2 = CouplingMatrix(16, use_sparse=False)
    coupling_matrix2.generate_pattern(
        "nearest_neighbor",
        strength_range=(-1.0, 1.0),
        topology="grid"
    )
    
    stats2 = coupling_matrix2.get_connectivity_stats()
    print(f"✓ Generated grid topology: {stats2['n_couplings']} edges, mean degree={stats2['mean_degree']:.1f}")
    
    return coupling_matrix

def test_gpu_annealer():
    """Test GPU annealer functionality."""
    print("\nTesting GPU annealer...")
    
    # Create test model
    config = IsingModelConfig(n_spins=20, use_sparse=False, device="cpu")
    model = IsingModel(config)
    
    # Add frustrated couplings
    np.random.seed(42)
    for _ in range(30):
        i, j = np.random.randint(0, model.n_spins, 2)
        if i != j:
            strength = np.random.choice([-1.0, 1.0])
            model.set_coupling(i, j, strength)
    
    initial_energy = model.compute_energy()
    print(f"✓ Created frustrated model with initial energy: {initial_energy:.6f}")
    
    # Configure annealer
    annealer_config = GPUAnnealerConfig(
        n_sweeps=500,
        initial_temp=5.0,
        final_temp=0.1,
        schedule_type=ScheduleType.GEOMETRIC,
        random_seed=42
    )
    
    annealer = GPUAnnealer(annealer_config)
    print(f"✓ Created annealer: {annealer}")
    
    # Run optimization
    result = annealer.anneal(model)
    
    print(f"✓ Optimization completed:")
    print(f"  Final energy: {result.best_energy:.6f}")
    print(f"  Energy improvement: {initial_energy - result.best_energy:.6f}")
    print(f"  Time: {result.total_time:.4f}s")
    print(f"  Sweeps: {result.n_sweeps}")
    print(f"  Final acceptance rate: {result.final_acceptance_rate:.4f}")
    
    # Verify improvement
    assert result.best_energy <= initial_energy, "Energy should not increase"
    print("✓ Energy improvement verified")
    
    return result

def test_temperature_schedules():
    """Test different temperature schedules."""
    print("\nTesting temperature schedules...")
    
    from spin_glass_rl.annealing.temperature_scheduler import TemperatureScheduler
    
    # Test schedule creation
    schedules = [ScheduleType.LINEAR, ScheduleType.GEOMETRIC, ScheduleType.EXPONENTIAL]
    
    for schedule_type in schedules:
        schedule = TemperatureScheduler.create_schedule(
            schedule_type, 
            initial_temp=10.0,
            final_temp=0.1,
            total_sweeps=100
        )
        
        # Test temperature progression
        temp_start = schedule.get_temperature(0)
        temp_mid = schedule.get_temperature(50)
        temp_end = schedule.get_temperature(99)
        
        assert temp_start >= temp_mid >= temp_end, f"Temperature should decrease for {schedule_type}"
        print(f"✓ {schedule_type.value}: {temp_start:.2f} → {temp_mid:.2f} → {temp_end:.2f}")

def main():
    """Run all Generation 1 tests."""
    print("="*60)
    print("GENERATION 1 BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    try:
        # Test core components
        model = test_basic_ising_model()
        coupling_matrix = test_coupling_matrix()
        test_temperature_schedules()
        result = test_gpu_annealer()
        
        print("\n" + "="*60)
        print("✅ ALL GENERATION 1 TESTS PASSED!")
        print("="*60)
        print("\nGeneration 1 (Make it Work) is complete:")
        print("• Basic Ising model creation and manipulation ✓")
        print("• Coupling matrix patterns and connectivity ✓") 
        print("• GPU annealer with temperature schedules ✓")
        print("• Energy computation and optimization ✓")
        print("\nReady to proceed to Generation 2 (Make it Robust)!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)