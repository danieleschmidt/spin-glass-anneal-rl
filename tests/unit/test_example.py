"""Example unit tests demonstrating testing patterns."""

import numpy as np
import pytest
import torch

from tests import UNIT_TEST


@pytest.mark.unit
class TestBasicFunctionality:
    """Test basic functionality and patterns."""
    
    def test_numpy_operations(self):
        """Test basic numpy operations."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])
        
        result = a + b
        expected = np.array([3, 5, 7, 9, 11])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_torch_operations(self, device):
        """Test basic torch operations."""
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        b = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device)
        
        result = a + b
        expected = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0], device=device)
        
        torch.testing.assert_close(result, expected)
    
    def test_ising_energy_calculation(self, simple_ising_model):
        """Test Ising energy calculation."""
        # Create a simple spin configuration
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        
        coupling_matrix = simple_ising_model["coupling_matrix"]
        external_field = simple_ising_model["external_field"]
        
        # Calculate energy: E = -sum(J_ij * s_i * s_j) - sum(h_i * s_i)
        interaction_energy = -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
        field_energy = -np.sum(external_field * spins)
        total_energy = interaction_energy + field_energy
        
        assert isinstance(total_energy, (int, float))
        assert not np.isnan(total_energy)
        assert not np.isinf(total_energy)
    
    def test_spin_flip_energy_change(self, simple_ising_model):
        """Test energy change calculation for spin flip."""
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        coupling_matrix = simple_ising_model["coupling_matrix"]
        external_field = simple_ising_model["external_field"]
        
        # Calculate initial energy
        initial_energy = (
            -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
            - np.sum(external_field * spins)
        )
        
        # Flip one spin
        flip_index = 3
        spins_flipped = spins.copy()
        spins_flipped[flip_index] *= -1
        
        # Calculate new energy
        new_energy = (
            -0.5 * np.sum(coupling_matrix * np.outer(spins_flipped, spins_flipped))
            - np.sum(external_field * spins_flipped)
        )
        
        # Calculate energy change directly
        local_field = np.sum(coupling_matrix[flip_index] * spins) + external_field[flip_index]
        energy_change_direct = 2 * spins[flip_index] * local_field
        energy_change_calculated = new_energy - initial_energy
        
        assert abs(energy_change_direct - energy_change_calculated) < 1e-10
    
    def test_temperature_schedules(self):
        """Test different temperature schedules."""
        # Linear schedule
        n_steps = 100
        t_initial = 10.0
        t_final = 0.1
        
        linear_schedule = np.linspace(t_initial, t_final, n_steps)
        assert linear_schedule[0] == t_initial
        assert linear_schedule[-1] == t_final
        assert len(linear_schedule) == n_steps
        
        # Geometric schedule
        geometric_schedule = t_initial * (t_final / t_initial) ** (np.arange(n_steps) / (n_steps - 1))
        assert abs(geometric_schedule[0] - t_initial) < 1e-10
        assert abs(geometric_schedule[-1] - t_final) < 1e-10
        assert len(geometric_schedule) == n_steps
        
        # Exponential schedule
        alpha = np.log(t_final / t_initial) / (n_steps - 1)
        exponential_schedule = t_initial * np.exp(alpha * np.arange(n_steps))
        assert abs(exponential_schedule[0] - t_initial) < 1e-10
        assert abs(exponential_schedule[-1] - t_final) < 1e-10
        assert len(exponential_schedule) == n_steps


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_spin_configuration(self):
        """Test handling of invalid spin configurations."""
        # Test non-binary spins
        invalid_spins = np.array([0.5, -0.3, 1.2, -1.0, 1.0])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, AssertionError)):
            # This would be a call to a spin validation function
            # validate_spins(invalid_spins)
            assert all(spin in [-1, 1] for spin in invalid_spins)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        coupling_matrix = np.random.randn(5, 5)
        external_field = np.random.randn(10)  # Wrong size
        
        # Should catch dimension mismatch
        with pytest.raises((ValueError, IndexError)):
            # This would be a call to an energy calculation function
            # calculate_energy(coupling_matrix, external_field, spins)
            assert coupling_matrix.shape[0] == len(external_field)
    
    def test_memory_limits(self):
        """Test behavior with large problem sizes."""
        # Test with a reasonably large but not excessive size
        n_spins = 1000
        coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
        
        # Should be able to handle this size
        assert coupling_matrix.shape == (n_spins, n_spins)
        assert not np.any(np.isnan(coupling_matrix))


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_coupling_matrix_symmetry(self):
        """Test that coupling matrices are symmetric."""
        n_spins = 20
        asymmetric_matrix = np.random.randn(n_spins, n_spins)
        
        # Make symmetric
        symmetric_matrix = (asymmetric_matrix + asymmetric_matrix.T) / 2
        
        # Test symmetry
        np.testing.assert_array_almost_equal(symmetric_matrix, symmetric_matrix.T)
    
    def test_spin_initialization(self):
        """Test different spin initialization methods."""
        n_spins = 50
        
        # Random initialization
        random_spins = np.random.choice([-1, 1], size=n_spins)
        assert len(random_spins) == n_spins
        assert all(spin in [-1, 1] for spin in random_spins)
        
        # All up initialization
        all_up_spins = np.ones(n_spins, dtype=int)
        assert len(all_up_spins) == n_spins
        assert all(spin == 1 for spin in all_up_spins)
        
        # All down initialization
        all_down_spins = -np.ones(n_spins, dtype=int)
        assert len(all_down_spins) == n_spins
        assert all(spin == -1 for spin in all_down_spins)
    
    def test_configuration_to_schedule_mapping(self, small_scheduling_problem):
        """Test mapping from spin configuration to schedule."""
        problem = small_scheduling_problem
        n_agents = problem["n_agents"]
        n_tasks = problem["n_tasks"]
        
        # Create a valid assignment (each task assigned to exactly one agent)
        assignment_matrix = np.zeros((n_agents, n_tasks), dtype=int)
        for task in range(n_tasks):
            agent = task % n_agents  # Simple round-robin assignment
            assignment_matrix[agent, task] = 1
        
        # Verify constraints
        # Each task assigned to exactly one agent
        task_assignments = np.sum(assignment_matrix, axis=0)
        assert all(count == 1 for count in task_assignments)
        
        # No agent exceeds capacity (simplified check)
        agent_loads = np.sum(assignment_matrix, axis=1)
        capacities = problem["agent_capacities"]
        for agent, load in enumerate(agent_loads):
            assert load <= capacities[agent], f"Agent {agent} overloaded: {load} > {capacities[agent]}"


@pytest.mark.unit
@pytest.mark.parametrize("n_spins,temperature", [
    (10, 1.0),
    (20, 0.5),
    (50, 0.1),
])
def test_metropolis_acceptance(n_spins, temperature):
    """Test Metropolis acceptance criterion."""
    # Generate random energy changes
    energy_changes = np.random.randn(100) * 10
    
    # Calculate acceptance probabilities
    acceptance_probs = np.minimum(1.0, np.exp(-energy_changes / temperature))
    
    # Verify properties
    assert all(0 <= prob <= 1 for prob in acceptance_probs)
    
    # Negative energy changes should always be accepted
    negative_changes = energy_changes < 0
    assert all(acceptance_probs[negative_changes] == 1.0)
    
    # Higher temperature should lead to higher acceptance of positive changes
    high_temp_probs = np.minimum(1.0, np.exp(-energy_changes / (temperature * 2)))
    positive_changes = energy_changes > 0
    if np.any(positive_changes):
        assert np.all(high_temp_probs[positive_changes] >= acceptance_probs[positive_changes])


@pytest.mark.unit
def test_performance_metrics():
    """Test calculation of performance metrics."""
    # Mock solution quality metrics
    true_optimal = 100.0
    found_solutions = [102.0, 105.0, 98.0, 110.0, 101.0]
    
    # Calculate metrics
    best_solution = min(found_solutions)
    worst_solution = max(found_solutions)
    average_solution = np.mean(found_solutions)
    std_solution = np.std(found_solutions)
    
    # Gap to optimal
    optimality_gap = (best_solution - true_optimal) / true_optimal * 100
    
    # Verify calculations
    assert best_solution == 98.0
    assert worst_solution == 110.0
    assert abs(average_solution - 103.2) < 1e-10
    assert optimality_gap < 0  # Found better than "optimal"
    
    # Success rate (solutions within 5% of optimal)
    threshold = true_optimal * 1.05
    successful_runs = sum(1 for sol in found_solutions if sol <= threshold)
    success_rate = successful_runs / len(found_solutions)
    
    assert success_rate == 0.6  # 3 out of 5 solutions within 5%