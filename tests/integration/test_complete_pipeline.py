"""Integration test for complete optimization pipeline."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Import the components we want to test
from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.coupling_matrix import CouplingMatrix
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.routing import TSPProblem
from spin_glass_rl.problems.scheduling import SchedulingProblem
from spin_glass_rl.benchmarks.standard_problems import MaxCutProblem, StandardBenchmarkSuite


class TestCompletePipeline:
    """Test the complete optimization pipeline integration."""
    
    def test_basic_ising_model_creation(self):
        """Test basic Ising model creation and operations."""
        config = IsingModelConfig(
            n_spins=10,
            coupling_strength=1.0,
            external_field_strength=0.5,
            use_sparse=True,
            device="cpu"
        )
        
        model = IsingModel(config)
        assert model.n_spins == 10
        
        # Test setting couplings
        model.set_coupling(0, 1, -1.0)
        assert model.get_coupling(0, 1) == -1.0
        
        # Test energy computation
        energy = model.compute_energy()
        assert isinstance(energy, float)
        
        # Test spin operations
        original_spins = model.get_spins().clone()
        model.flip_spin(0)
        new_spins = model.get_spins()
        assert not torch.equal(original_spins, new_spins)
    
    def test_coupling_matrix_patterns(self):
        """Test coupling matrix pattern generation."""
        n_spins = 20
        coupling_matrix = CouplingMatrix(n_spins, use_sparse=True)
        
        # Test different patterns
        patterns = ["fully_connected", "random_graph", "small_world", "scale_free"]
        
        for pattern in patterns:
            coupling_matrix.generate_pattern(
                pattern, 
                strength_range=(-1.0, 1.0),
                edge_probability=0.3
            )
            
            assert coupling_matrix.n_couplings > 0
            assert coupling_matrix.matrix.shape == (n_spins, n_spins)
    
    def test_annealing_with_different_schedules(self):
        """Test annealing with different temperature schedules."""
        # Create simple Ising model
        config = IsingModelConfig(n_spins=10, use_sparse=True, device="cpu")
        model = IsingModel(config)
        
        # Add some couplings
        for i in range(9):
            model.set_coupling(i, i + 1, -1.0)  # Ferromagnetic chain
        
        initial_energy = model.compute_energy()
        
        schedules = [ScheduleType.LINEAR, ScheduleType.EXPONENTIAL, ScheduleType.GEOMETRIC]
        
        for schedule_type in schedules:
            model_copy = model.copy()
            
            annealer_config = GPUAnnealerConfig(
                n_sweeps=100,  # Short for testing
                initial_temp=5.0,
                final_temp=0.1,
                schedule_type=schedule_type,
                random_seed=42
            )
            
            annealer = GPUAnnealer(annealer_config)
            result = annealer.anneal(model_copy)
            
            # Verify result structure
            assert hasattr(result, 'best_energy')
            assert hasattr(result, 'best_configuration')
            assert hasattr(result, 'energy_history')
            assert hasattr(result, 'total_time')
            
            # Energy should improve or stay same
            assert result.best_energy <= initial_energy + 1e-6  # Small tolerance
    
    def test_tsp_problem_encoding_decoding(self):
        """Test TSP problem encoding and decoding."""
        tsp = TSPProblem()
        
        # Generate small instance
        instance_params = tsp.generate_random_instance(
            n_locations=6,  # Small for testing
            area_size=100.0
        )
        
        assert len(tsp.locations) == 6
        
        # Encode to Ising model
        ising_model = tsp.encode_to_ising(
            penalty_weights={
                "city_visit": 10.0,
                "position_fill": 10.0
            }
        )
        
        assert ising_model.n_spins == 36  # 6 cities × 6 positions
        
        # Test decoding with a simple configuration
        # Create a valid tour configuration
        spins = torch.ones(36) * (-1)  # All -1 initially
        
        # Set one valid tour: 0->1->2->3->4->5->0
        tour = [0, 1, 2, 3, 4, 5]
        for pos, city in enumerate(tour):
            spin_idx = tsp._get_spin_index(city, pos)
            spins[spin_idx] = 1
        
        solution = tsp.decode_solution(spins)
        
        assert hasattr(solution, 'objective_value')
        assert hasattr(solution, 'is_feasible')
        assert hasattr(solution, 'variables')
        assert 'tour' in solution.variables
    
    def test_scheduling_problem(self):
        """Test scheduling problem functionality."""
        scheduler = SchedulingProblem()
        
        # Generate small instance
        instance_params = scheduler.generate_random_instance(
            n_tasks=4,
            n_agents=2,
            time_horizon=20.0
        )
        
        assert len(scheduler.tasks) == 4
        assert len(scheduler.agents) == 2
        
        # Encode to Ising model
        ising_model = scheduler.encode_to_ising(
            objective="makespan",
            penalty_weights={
                "assignment": 50.0,
                "capacity": 25.0,
                "time_window": 15.0
            }
        )
        
        assert ising_model.n_spins > 0
        
        # Test with annealing
        annealer_config = GPUAnnealerConfig(
            n_sweeps=50,  # Short for testing
            initial_temp=2.0,
            final_temp=0.1,
            random_seed=42
        )
        
        annealer = GPUAnnealer(annealer_config)
        result = annealer.anneal(ising_model)
        
        # Decode solution
        solution = scheduler.decode_solution(result.best_configuration)
        
        assert hasattr(solution, 'objective_value')
        assert hasattr(solution, 'metadata')
    
    def test_maxcut_benchmark(self):
        """Test MaxCut benchmark problem."""
        maxcut = MaxCutProblem(n_vertices=10)
        
        # Generate instance
        instance_params = maxcut.generate_instance(seed=42)
        assert instance_params['n_vertices'] == 10
        
        # Encode to Ising model
        ising_model = maxcut.encode_to_ising()
        assert ising_model.n_spins == 10
        
        # Test annealing
        annealer_config = GPUAnnealerConfig(
            n_sweeps=100,
            initial_temp=3.0,
            final_temp=0.05,
            random_seed=42
        )
        
        annealer = GPUAnnealer(annealer_config)
        result = annealer.anneal(ising_model)
        
        # Decode solution
        solution = maxcut.decode_solution(result.best_configuration)
        
        assert solution.is_feasible  # MaxCut has no hard constraints
        assert solution.objective_value >= 0  # Cut value should be non-negative
        assert 'partition' in solution.variables
        assert 'cut_edges' in solution.metadata
    
    def test_benchmark_suite(self):
        """Test the standard benchmark suite."""
        suite = StandardBenchmarkSuite()
        
        # List problems
        problems = suite.list_problems()
        assert len(problems) > 0
        assert any('maxcut' in p for p in problems)
        
        # Test running a small benchmark
        config = GPUAnnealerConfig(
            n_sweeps=50,  # Very short for testing
            initial_temp=2.0,
            final_temp=0.1,
            random_seed=42
        )
        
        # Run on smallest MaxCut problem
        if 'maxcut_20' in problems:
            result = suite.run_benchmark('maxcut_20', config, seed=42)
            
            assert 'problem_name' in result
            assert 'solution' in result
            assert 'annealing_result' in result
            assert result['problem_name'] == 'maxcut_20'
    
    def test_magnetization_tracking(self):
        """Test magnetization history tracking in spin dynamics."""
        from spin_glass_rl.core.spin_dynamics import SpinDynamics, UpdateRule
        
        # Create simple model
        config = IsingModelConfig(n_spins=8, use_sparse=True, device="cpu")
        model = IsingModel(config)
        
        # Add some couplings
        for i in range(7):
            model.set_coupling(i, i + 1, -1.0)
        
        # Create dynamics
        dynamics = SpinDynamics(model, temperature=1.0, update_rule=UpdateRule.METROPOLIS)
        
        # Run some sweeps
        dynamics.run_dynamics(n_sweeps=10, record_interval=1)
        
        # Check that magnetization history is tracked
        assert len(dynamics.magnetization_history) > 0
        assert len(dynamics.energy_history) > 0
        assert len(dynamics.magnetization_history) == len(dynamics.energy_history)
        
        # Test autocorrelation calculation
        tau_energy = dynamics.estimate_autocorrelation_time("energy")
        tau_magnetization = dynamics.estimate_autocorrelation_time("magnetization")
        
        assert isinstance(tau_energy, float)
        assert isinstance(tau_magnetization, float)
    
    def test_enhanced_validation(self):
        """Test enhanced validation in TSP and other problems."""
        tsp = TSPProblem()
        
        # Test validation with empty locations
        with pytest.raises(ValueError, match="TSP requires at least one location"):
            tsp.encode_to_ising()
        
        # Test validation with invalid penalty weights
        tsp.generate_random_instance(n_locations=5)
        
        with pytest.raises(ValueError, match="Missing required penalty weights"):
            tsp.encode_to_ising(penalty_weights={"city_visit": 10.0})  # Missing position_fill
        
        with pytest.raises(ValueError, match="must be a positive number"):
            tsp.encode_to_ising(penalty_weights={
                "city_visit": -5.0,  # Invalid negative weight
                "position_fill": 10.0
            })
    
    def test_annealing_result_validation(self):
        """Test validation in AnnealingResult."""
        from spin_glass_rl.annealing.result import AnnealingResult
        
        # Test with valid data
        valid_result = AnnealingResult(
            best_configuration=torch.tensor([1, -1, 1, -1]),
            best_energy=-2.0,
            energy_history=[-1.5, -1.8, -2.0],
            temperature_history=[2.0, 1.0, 0.5],
            acceptance_rate_history=[0.8, 0.6, 0.4],
            total_time=1.5,
            n_sweeps=100
        )
        
        assert valid_result.best_energy == -2.0
        assert valid_result.energy_std > 0
        
        # Test with invalid data
        with pytest.raises(ValueError, match="contains invalid values"):
            AnnealingResult(
                best_configuration=torch.tensor([1, -1, 1, -1]),
                best_energy=float('nan'),  # Invalid NaN
                energy_history=[],
                temperature_history=[],
                acceptance_rate_history=[],
                total_time=1.0,
                n_sweeps=100
            )
        
        with pytest.raises(ValueError, match="must be positive"):
            AnnealingResult(
                best_configuration=torch.tensor([1, -1, 1, -1]),
                best_energy=-2.0,
                energy_history=[],
                temperature_history=[],
                acceptance_rate_history=[],
                total_time=1.0,
                n_sweeps=0  # Invalid n_sweeps
            )
    
    def test_constraint_higher_order_handling(self):
        """Test higher-order constraint handling."""
        from spin_glass_rl.core.constraints import ConstraintEncoder
        
        config = IsingModelConfig(n_spins=6, use_sparse=True, device="cpu")
        model = IsingModel(config)
        
        encoder = ConstraintEncoder(model)
        
        # Test higher-order constraint (should use approximation)
        spin_indices = [0, 1, 2, 3, 4]  # 5 spins - higher order
        coefficients = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # This should trigger the higher-order approximation
        encoder.add_equality_constraint(
            spin_indices, coefficients, target=2.0, penalty_weight=10.0,
            description="High-order test constraint"
        )
        
        # Verify constraint was added
        assert len(encoder.constraints) == 1
        
        # Check that external fields were modified (approximation method)
        initial_fields = torch.zeros(6)
        current_fields = model.external_fields
        assert not torch.equal(initial_fields, current_fields)
    
    @pytest.mark.parametrize("schedule_type", [
        ScheduleType.LINEAR,
        ScheduleType.EXPONENTIAL, 
        ScheduleType.GEOMETRIC,
        ScheduleType.ADAPTIVE
    ])
    def test_all_temperature_schedules(self, schedule_type):
        """Test all temperature schedule types."""
        config = IsingModelConfig(n_spins=5, use_sparse=True, device="cpu")
        model = IsingModel(config)
        
        # Simple ferromagnetic chain
        for i in range(4):
            model.set_coupling(i, i + 1, -1.0)
        
        annealer_config = GPUAnnealerConfig(
            n_sweeps=20,  # Very short for testing
            initial_temp=3.0,
            final_temp=0.1,
            schedule_type=schedule_type,
            random_seed=42
        )
        
        annealer = GPUAnnealer(annealer_config)
        result = annealer.anneal(model)
        
        # Verify basic result properties
        assert isinstance(result.best_energy, float)
        assert len(result.energy_history) > 0
        assert len(result.temperature_history) > 0
        assert result.total_time > 0
        
        # Verify temperature decreases (for non-adaptive schedules)
        if schedule_type != ScheduleType.ADAPTIVE:
            temps = result.temperature_history
            assert temps[0] >= temps[-1]  # Should decrease or stay same


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_large_problem_handling(self):
        """Test handling of reasonably large problems."""
        # Create larger TSP instance
        tsp = TSPProblem()
        instance_params = tsp.generate_random_instance(n_locations=20)  # Reasonable size
        
        assert len(tsp.locations) == 20
        
        # Encoding should work
        ising_model = tsp.encode_to_ising()
        assert ising_model.n_spins == 400  # 20 × 20
        
        # Should be able to compute energy
        energy = ising_model.compute_energy()
        assert isinstance(energy, float)
        
    def test_memory_efficient_sparse_matrices(self):
        """Test sparse matrix memory efficiency."""
        n_spins = 100
        
        # Dense matrix
        config_dense = IsingModelConfig(n_spins=n_spins, use_sparse=False, device="cpu")
        model_dense = IsingModel(config_dense)
        
        # Sparse matrix  
        config_sparse = IsingModelConfig(n_spins=n_spins, use_sparse=True, device="cpu")
        model_sparse = IsingModel(config_sparse)
        
        # Add same sparse coupling pattern to both
        for i in range(0, n_spins, 10):  # Every 10th spin
            if i + 1 < n_spins:
                model_dense.set_coupling(i, i + 1, -1.0)
                model_sparse.set_coupling(i, i + 1, -1.0)
        
        # Both should give same energy
        energy_dense = model_dense.compute_energy()
        energy_sparse = model_sparse.compute_energy()
        
        assert abs(energy_dense - energy_sparse) < 1e-6
        
    def test_batch_operations(self):
        """Test batch operations for performance."""
        n_models = 5
        models = []
        
        for i in range(n_models):
            config = IsingModelConfig(n_spins=10, use_sparse=True, device="cpu")
            model = IsingModel(config)
            
            # Add random couplings
            np.random.seed(i)
            for _ in range(10):
                i_spin = np.random.randint(0, 10)
                j_spin = np.random.randint(0, 10)
                if i_spin != j_spin:
                    model.set_coupling(i_spin, j_spin, np.random.uniform(-1, 1))
            
            models.append(model)
        
        # Compute energies for all models
        energies = []
        for model in models:
            energies.append(model.compute_energy())
        
        assert len(energies) == n_models
        assert all(isinstance(e, float) for e in energies)


def test_import_all_new_modules():
    """Test that all new modules can be imported without errors."""
    modules_to_test = [
        'spin_glass_rl.benchmarks.standard_problems',
        'spin_glass_rl.rl_integration.training_pipeline',
        'spin_glass_rl.utils.comprehensive_monitoring',
        'spin_glass_rl.deployment.production_config'
    ]
    
    for module_name in modules_to_test:
        try:
            # Test import by trying to load module spec
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            assert spec is not None, f"Module {module_name} not found"
            
            # Test that the file exists and is readable
            assert Path(spec.origin).exists(), f"Module file {spec.origin} does not exist"
            
        except Exception as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])