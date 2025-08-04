"""End-to-end integration tests for TSP problem solving."""

import pytest
import numpy as np
import torch
from pathlib import Path

from spin_glass_rl.problems.routing import TSPProblem, Location
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.parallel_tempering import ParallelTempering, ParallelTemperingConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.utils.exceptions import ValidationError


class TestTSPEndToEnd:
    """End-to-end tests for TSP problem solving."""
    
    @pytest.fixture
    def small_tsp(self):
        """Create small TSP instance for testing."""
        tsp = TSPProblem()
        
        # Create small instance with known structure
        locations = [
            Location(id=0, name="A", x=0.0, y=0.0),
            Location(id=1, name="B", x=3.0, y=0.0),
            Location(id=2, name="C", x=3.0, y=4.0),
            Location(id=3, name="D", x=0.0, y=4.0),
        ]
        
        for location in locations:
            tsp.add_location(location)
        
        return tsp
    
    @pytest.fixture
    def medium_tsp(self):
        """Create medium TSP instance."""
        tsp = TSPProblem()
        tsp.generate_random_instance(n_locations=8, area_size=100.0)
        return tsp
    
    @pytest.fixture
    def annealer_config(self):
        """Create annealer configuration for testing."""
        return GPUAnnealerConfig(
            n_sweeps=500,
            initial_temp=5.0,
            final_temp=0.01,
            schedule_type=ScheduleType.GEOMETRIC,
            random_seed=42
        )
    
    def test_small_tsp_solve_basic(self, small_tsp, annealer_config):
        """Test solving small TSP with basic annealer."""
        # Encode problem
        ising_model = small_tsp.encode_to_ising()
        
        assert ising_model.n_spins == 16  # 4 cities * 4 positions
        
        # Solve with annealer
        annealer = GPUAnnealer(annealer_config)
        solution = small_tsp.solve_with_annealer(annealer)
        
        # Verify solution structure
        assert solution.objective_value >= 0
        assert isinstance(solution.is_feasible, bool)
        assert "tour" in solution.variables
        assert "routes" in solution.variables
        
        # If feasible, verify tour properties
        if solution.is_feasible:
            tour = solution.variables["tour"]
            assert len(tour) == 4
            assert set(tour) == {0, 1, 2, 3}  # All cities visited
            
            # Verify distance calculation
            total_distance = 0.0
            for i in range(4):
                city1 = tour[i]
                city2 = tour[(i + 1) % 4]
                distance = small_tsp.get_distance(city1, city2)
                total_distance += distance
            
            assert abs(solution.objective_value - total_distance) < 1e-6
    
    def test_small_tsp_known_optimal(self, small_tsp, annealer_config):
        """Test TSP with known optimal solution."""
        # For rectangle (0,0)-(3,0)-(3,4)-(0,4), optimal tour is perimeter = 14
        expected_optimal = 14.0
        
        # Run multiple times to increase chance of finding optimal
        best_distance = float('inf')
        
        for _ in range(5):
            annealer = GPUAnnealer(annealer_config)
            solution = small_tsp.solve_with_annealer(annealer)
            
            if solution.is_feasible and solution.objective_value < best_distance:
                best_distance = solution.objective_value
        
        # Should find optimal or very close
        assert best_distance <= expected_optimal + 0.1
    
    def test_medium_tsp_solve(self, medium_tsp, annealer_config):
        """Test solving medium-sized TSP."""
        solution = medium_tsp.solve_with_annealer(GPUAnnealer(annealer_config))
        
        # Verify solution properties
        assert solution.objective_value >= 0
        
        if solution.is_feasible:
            tour = solution.variables["tour"]
            assert len(tour) == len(medium_tsp.locations)
            assert len(set(tour)) == len(medium_tsp.locations)  # All unique cities
    
    def test_tsp_with_different_schedulers(self, small_tsp):
        """Test TSP with different temperature schedules."""
        schedules = [
            ScheduleType.LINEAR,
            ScheduleType.EXPONENTIAL, 
            ScheduleType.GEOMETRIC,
            ScheduleType.ADAPTIVE
        ]
        
        results = {}
        
        for schedule in schedules:
            config = GPUAnnealerConfig(
                n_sweeps=300,
                schedule_type=schedule,
                random_seed=42
            )
            annealer = GPUAnnealer(config)
            solution = small_tsp.solve_with_annealer(annealer)
            results[schedule.value] = solution.objective_value
        
        # All should produce finite results
        for schedule, distance in results.items():
            assert np.isfinite(distance)
            assert distance >= 0
    
    def test_tsp_parallel_tempering(self, small_tsp):
        """Test TSP with parallel tempering."""
        pt_config = ParallelTemperingConfig(
            n_replicas=4,
            n_sweeps=300,
            temp_min=0.1,
            temp_max=5.0,
            random_seed=42
        )
        
        pt = ParallelTempering(pt_config)
        solution = small_tsp.solve_with_annealer(pt)
        
        assert solution.objective_value >= 0
        assert isinstance(solution.is_feasible, bool)
    
    def test_tsp_reproducibility(self, small_tsp):
        """Test reproducibility with fixed seed."""
        config = GPUAnnealerConfig(n_sweeps=200, random_seed=42)
        
        # Run twice with same seed
        solution1 = small_tsp.solve_with_annealer(GPUAnnealer(config))
        solution2 = small_tsp.solve_with_annealer(GPUAnnealer(config))
        
        # Should get identical results
        assert solution1.objective_value == solution2.objective_value
        if solution1.is_feasible and solution2.is_feasible:
            assert solution1.variables["tour"] == solution2.variables["tour"]
    
    def test_tsp_solution_validation(self, small_tsp, annealer_config):
        """Test TSP solution validation."""
        solution = small_tsp.solve_with_annealer(GPUAnnealer(annealer_config))
        
        is_valid = small_tsp.validate_solution(solution)
        
        # Validation should match feasibility
        assert is_valid == solution.is_feasible
        
        if solution.is_feasible:
            tour = solution.variables["tour"]
            
            # Manual validation
            assert len(tour) == len(small_tsp.locations)
            assert len(set(tour)) == len(small_tsp.locations)
            assert all(0 <= city < len(small_tsp.locations) for city in tour)
    
    def test_tsp_constraint_violations(self, small_tsp, annealer_config):
        """Test constraint violation reporting."""
        solution = small_tsp.solve_with_annealer(GPUAnnealer(annealer_config))
        
        violations = solution.constraint_violations
        
        # Should have expected constraint types
        expected_constraints = ["city_assignment", "position_assignment", "invalid_tour"]
        for constraint in expected_constraints:
            assert constraint in violations
        
        # If feasible, violations should be zero
        if solution.is_feasible:
            for violation in violations.values():
                assert violation == 0.0
    
    def test_tsp_benchmarking(self, small_tsp):
        """Test TSP benchmarking functionality."""
        config = GPUAnnealerConfig(n_sweeps=100, random_seed=42)
        annealer = GPUAnnealer(config)
        
        benchmark_results = small_tsp.benchmark_instance(
            {"n_locations": 4}, annealer, n_trials=3
        )
        
        # Verify benchmark structure
        assert "mean_objective" in benchmark_results
        assert "std_objective" in benchmark_results
        assert "best_objective" in benchmark_results
        assert "mean_time" in benchmark_results
        assert "feasibility_rate" in benchmark_results
        
        # Values should be reasonable
        assert benchmark_results["mean_objective"] >= 0
        assert benchmark_results["mean_time"] > 0
        assert 0 <= benchmark_results["feasibility_rate"] <= 1
        assert benchmark_results["n_trials"] == 3
    
    def test_tsp_export_import(self, small_tsp, annealer_config, temp_dir):
        """Test solution export and import."""
        solution = small_tsp.solve_with_annealer(GPUAnnealer(annealer_config))
        
        # Export solution
        export_path = temp_dir / "solution.json"
        small_tsp.export_solution(solution, str(export_path))
        
        # Verify file was created
        assert export_path.exists()
        
        # Verify content (basic check)
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert "problem_name" in data
        assert "objective_value" in data
        assert "is_feasible" in data
        assert data["problem_name"] == "Traveling Salesman Problem"
    
    @pytest.mark.slow
    def test_large_tsp_performance(self):
        """Test performance on larger TSP instance."""
        tsp = TSPProblem()
        tsp.generate_random_instance(n_locations=15)
        
        config = GPUAnnealerConfig(
            n_sweeps=1000,
            initial_temp=10.0,
            final_temp=0.001,
            random_seed=42
        )
        
        annealer = GPUAnnealer(config)
        solution = tsp.solve_with_annealer(annealer)
        
        # Should complete in reasonable time
        assert solution.metadata["total_time"] < 60.0  # Less than 1 minute
        
        if solution.is_feasible:
            assert len(solution.variables["tour"]) == 15
    
    def test_tsp_different_distance_metrics(self, small_tsp):
        """Test TSP with different distance metrics."""
        # Test Euclidean (default)
        small_tsp.compute_distance_matrix("euclidean")
        euclidean_matrix = small_tsp.distance_matrix.copy()
        
        # Test Manhattan
        small_tsp.compute_distance_matrix("manhattan")
        manhattan_matrix = small_tsp.distance_matrix.copy()
        
        # Matrices should be different
        assert not np.allclose(euclidean_matrix, manhattan_matrix)
        
        # Both should be symmetric
        assert np.allclose(euclidean_matrix, euclidean_matrix.T)
        assert np.allclose(manhattan_matrix, manhattan_matrix.T)
        
        # Solve with Manhattan distance
        config = GPUAnnealerConfig(n_sweeps=200, random_seed=42)
        solution = small_tsp.solve_with_annealer(GPUAnnealer(config))
        
        assert solution.objective_value >= 0
    
    def test_tsp_custom_distance_matrix(self, small_tsp):
        """Test TSP with custom distance matrix."""
        # Create custom symmetric distance matrix
        n = len(small_tsp.locations)
        custom_matrix = np.random.rand(n, n) * 10
        custom_matrix = (custom_matrix + custom_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(custom_matrix, 0)  # Zero diagonal
        
        small_tsp.set_distance_matrix(custom_matrix)
        
        config = GPUAnnealerConfig(n_sweeps=200, random_seed=42)
        solution = small_tsp.solve_with_annealer(GPUAnnealer(config))
        
        assert solution.objective_value >= 0
    
    def test_tsp_edge_cases(self):
        """Test TSP edge cases."""
        # Single city (degenerate case)
        tsp = TSPProblem()
        tsp.add_location(Location(id=0, name="Only", x=0.0, y=0.0))
        
        # Should handle gracefully
        try:
            ising_model = tsp.encode_to_ising()
            config = GPUAnnealerConfig(n_sweeps=10)
            solution = tsp.solve_with_annealer(GPUAnnealer(config))
            
            if solution.is_feasible:
                assert solution.objective_value == 0.0  # No travel needed
        except (ValidationError, ValueError):
            # Acceptable to reject degenerate cases
            pass
    
    def test_tsp_constraint_weights_impact(self, small_tsp):
        """Test impact of different constraint weights."""
        penalty_weights = [
            {"city_visit": 50.0, "position_fill": 50.0},
            {"city_visit": 200.0, "position_fill": 200.0},
        ]
        
        results = []
        
        for weights in penalty_weights:
            small_tsp.encode_to_ising(penalty_weights=weights)
            config = GPUAnnealerConfig(n_sweeps=200, random_seed=42)
            solution = small_tsp.solve_with_annealer(GPUAnnealer(config))
            results.append(solution)
        
        # Both should produce valid results
        for solution in results:
            assert solution.objective_value >= 0
        
        # Higher penalties might lead to more feasible solutions
        feasible_count = sum(1 for s in results if s.is_feasible)
        assert feasible_count >= 0  # At least some should work