"""Example integration tests for end-to-end pipelines."""

import numpy as np
import pytest
import torch

from tests import INTEGRATION_TEST


@pytest.mark.integration
class TestAnnealingPipeline:
    """Test complete annealing pipeline integration."""
    
    def test_problem_to_solution_pipeline(self, small_scheduling_problem, device):
        """Test complete pipeline from problem to solution."""
        problem = small_scheduling_problem
        
        # Step 1: Problem definition
        n_agents = problem["n_agents"]
        n_tasks = problem["n_tasks"]
        
        # Step 2: Convert to Ising model (mock implementation)
        n_spins = n_agents * n_tasks
        coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
        coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
        np.fill_diagonal(coupling_matrix, 0)
        external_field = np.random.randn(n_spins) * 0.5
        
        # Step 3: Initialize spin configuration
        spins = np.random.choice([-1, 1], size=n_spins)
        
        # Step 4: Annealing simulation (simplified)
        n_sweeps = 100
        temperature = 1.0
        
        for sweep in range(n_sweeps):
            for i in range(n_spins):
                # Calculate local field
                local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                
                # Energy change for flip
                energy_change = 2 * spins[i] * local_field
                
                # Metropolis criterion
                if energy_change < 0 or np.random.random() < np.exp(-energy_change / temperature):
                    spins[i] *= -1
            
            # Cool down
            temperature *= 0.99
        
        # Step 5: Convert back to schedule
        assignment_matrix = spins.reshape(n_agents, n_tasks)
        assignment_matrix = (assignment_matrix + 1) // 2  # Convert to 0/1
        
        # Step 6: Validate solution
        assert assignment_matrix.shape == (n_agents, n_tasks)
        assert np.all((assignment_matrix == 0) | (assignment_matrix == 1))
        
        # Basic constraint check (tasks should be assigned)
        total_assignments = np.sum(assignment_matrix)
        assert total_assignments > 0
    
    def test_multi_replica_annealing(self, simple_ising_model, device):
        """Test parallel tempering with multiple replicas."""
        ising = simple_ising_model
        n_spins = ising["n_spins"]
        coupling_matrix = ising["coupling_matrix"]
        external_field = ising["external_field"]
        
        # Setup parallel tempering
        n_replicas = 8
        temperatures = np.logspace(-1, 1, n_replicas)  # 0.1 to 10
        
        # Initialize replicas
        replicas = [np.random.choice([-1, 1], size=n_spins) for _ in range(n_replicas)]
        energies = []
        
        # Calculate initial energies
        for replica in replicas:
            energy = (
                -0.5 * np.sum(coupling_matrix * np.outer(replica, replica))
                - np.sum(external_field * replica)
            )
            energies.append(energy)
        
        # Run parallel tempering
        n_sweeps = 50
        exchange_interval = 10
        
        for sweep in range(n_sweeps):
            # Update each replica
            for r in range(n_replicas):
                for i in range(n_spins):
                    local_field = (
                        np.sum(coupling_matrix[i] * replicas[r]) + external_field[i]
                    )
                    energy_change = 2 * replicas[r][i] * local_field
                    
                    if (energy_change < 0 or 
                        np.random.random() < np.exp(-energy_change / temperatures[r])):
                        replicas[r][i] *= -1
                        energies[r] += energy_change
            
            # Replica exchange
            if sweep % exchange_interval == 0:
                for r in range(n_replicas - 1):
                    energy_diff = (energies[r+1] - energies[r])
                    temp_diff = (1/temperatures[r] - 1/temperatures[r+1])
                    
                    if np.random.random() < np.exp(energy_diff * temp_diff):
                        # Exchange replicas
                        replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                        energies[r], energies[r+1] = energies[r+1], energies[r]
        
        # Verify results
        assert len(replicas) == n_replicas
        assert len(energies) == n_replicas
        assert all(len(replica) == n_spins for replica in replicas)
        assert all(np.all((replica == -1) | (replica == 1)) for replica in replicas)
        
        # Best solution should be in coldest replica (lowest temperature)
        best_energy = min(energies)
        coldest_replica_energy = energies[0]  # temperatures[0] is lowest
        
        # The coldest replica should generally have good energy
        # (though not guaranteed to be the best due to stochasticity)
        assert isinstance(best_energy, (int, float))
        assert isinstance(coldest_replica_energy, (int, float))


@pytest.mark.integration 
class TestRLIntegration:
    """Test RL-annealing integration."""
    
    def test_policy_guided_initialization(self, simple_policy_network, medium_ising_model, device):
        """Test policy-guided spin initialization."""
        policy = simple_policy_network.to(device)
        ising = medium_ising_model
        n_spins = ising["n_spins"]
        
        # Create problem features (mock)
        problem_features = torch.randn(10, device=device)
        
        # Policy suggests initial configuration
        with torch.no_grad():
            policy_output = policy(problem_features)
            
        # Convert to spin configuration
        spins = torch.sign(policy_output).cpu().numpy()
        spins = np.where(spins == 0, 1, spins)  # Handle zeros
        
        # Verify valid spin configuration
        assert len(spins) == min(5, n_spins)  # Policy outputs 5 values
        assert np.all((spins == -1) | (spins == 1))
        
        # Use this as initialization for annealing
        if len(spins) < n_spins:
            # Extend to full problem size
            full_spins = np.random.choice([-1, 1], size=n_spins)
            full_spins[:len(spins)] = spins
            spins = full_spins
        
        # Run short annealing with policy initialization
        coupling_matrix = ising["coupling_matrix"]
        external_field = ising["external_field"]
        temperature = 1.0
        
        for _ in range(20):  # Short run
            for i in range(n_spins):
                local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                energy_change = 2 * spins[i] * local_field
                
                if energy_change < 0 or np.random.random() < np.exp(-energy_change / temperature):
                    spins[i] *= -1
            
            temperature *= 0.95
        
        # Final verification
        assert len(spins) == n_spins
        assert np.all((spins == -1) | (spins == 1))
    
    def test_reward_feedback_loop(self, simple_value_network, device):
        """Test reward feedback for policy training."""
        value_network = simple_value_network.to(device)
        
        # Mock trajectory data
        states = torch.randn(10, 10, device=device)  # 10 timesteps, 10 features
        actions = torch.randn(10, 5, device=device)   # 10 timesteps, 5 actions
        
        # Calculate values
        with torch.no_grad():
            values = value_network(states).squeeze()
        
        # Mock rewards (energy-based)
        mock_energies = np.random.randn(10) * 10
        rewards = torch.tensor(-mock_energies, device=device)  # Negative energy as reward
        
        # Calculate returns (simplified)
        gamma = 0.99
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Calculate advantages
        advantages = returns - values
        
        # Verify shapes and values
        assert values.shape == (10,)
        assert rewards.shape == (10,)
        assert returns.shape == (10,)
        assert advantages.shape == (10,)
        
        # Basic sanity checks
        assert torch.all(torch.isfinite(values))
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Test integration with larger problem sizes."""
    
    def test_scaling_behavior(self, device):
        """Test behavior as problem size scales."""
        problem_sizes = [10, 50, 100]
        results = []
        
        for n_spins in problem_sizes:
            # Create problem
            coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
            coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
            np.fill_diagonal(coupling_matrix, 0)
            external_field = np.random.randn(n_spins) * 0.5
            
            # Initialize
            spins = np.random.choice([-1, 1], size=n_spins)
            
            # Run short annealing
            temperature = 1.0
            n_sweeps = 20  # Keep short for testing
            
            initial_energy = (
                -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                - np.sum(external_field * spins)
            )
            
            for _ in range(n_sweeps):
                for i in range(n_spins):
                    local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                    energy_change = 2 * spins[i] * local_field
                    
                    if energy_change < 0 or np.random.random() < np.exp(-energy_change / temperature):
                        spins[i] *= -1
                
                temperature *= 0.95
            
            final_energy = (
                -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                - np.sum(external_field * spins)
            )
            
            results.append({
                "n_spins": n_spins,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_improvement": initial_energy - final_energy
            })
        
        # Verify scaling behavior
        assert len(results) == len(problem_sizes)
        
        # Should be able to handle all sizes
        for result in results:
            assert result["n_spins"] > 0
            assert np.isfinite(result["initial_energy"])
            assert np.isfinite(result["final_energy"])
            
        # Energy should generally improve (though not guaranteed)
        improvements = [r["energy_improvement"] for r in results]
        avg_improvement = np.mean(improvements)
        
        # At least some improvement on average (stochastic, so allow some tolerance)
        assert avg_improvement > -10  # Not too much worse on average


@pytest.mark.integration
class TestHardwareIntegration:
    """Test hardware-specific integration."""
    
    @pytest.mark.gpu
    def test_gpu_tensor_operations(self, device, skip_if_no_cuda):
        """Test GPU tensor operations for annealing."""
        if device.type == "cpu":
            pytest.skip("GPU not available")
        
        n_spins = 1000
        batch_size = 32
        
        # Create tensors on GPU
        coupling_matrix = torch.randn(n_spins, n_spins, device=device) * 0.1
        coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
        coupling_matrix.fill_diagonal_(0)
        
        external_field = torch.randn(n_spins, device=device) * 0.5
        spins = torch.randint(0, 2, (batch_size, n_spins), device=device) * 2 - 1
        
        # Batch energy calculation
        interaction_energy = -0.5 * torch.sum(
            coupling_matrix.unsqueeze(0) * 
            torch.bmm(spins.unsqueeze(2), spins.unsqueeze(1)), 
            dim=(1, 2)
        )
        field_energy = -torch.sum(external_field.unsqueeze(0) * spins, dim=1)
        total_energy = interaction_energy + field_energy
        
        # Verify GPU computation
        assert total_energy.device == device
        assert total_energy.shape == (batch_size,)
        assert torch.all(torch.isfinite(total_energy))
        
        # Test memory efficiency
        memory_before = torch.cuda.memory_allocated(device)
        
        # Large batch operation
        large_batch = 128
        large_spins = torch.randint(0, 2, (large_batch, n_spins), device=device) * 2 - 1
        
        # Should not cause memory issues
        large_energy = -torch.sum(external_field.unsqueeze(0) * large_spins, dim=1)
        
        memory_after = torch.cuda.memory_allocated(device)
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        expected_memory = large_batch * n_spins * 4  # 4 bytes per float32
        assert memory_increase <= expected_memory * 2  # Allow 2x overhead
    
    def test_cpu_fallback(self, device):
        """Test CPU fallback when GPU is not available."""
        # Force CPU computation
        cpu_device = torch.device("cpu")
        
        n_spins = 100
        coupling_matrix = torch.randn(n_spins, n_spins, device=cpu_device) * 0.1
        external_field = torch.randn(n_spins, device=cpu_device) * 0.5
        spins = torch.randint(0, 2, (n_spins,), device=cpu_device) * 2 - 1
        
        # CPU energy calculation
        energy = (
            -0.5 * torch.sum(coupling_matrix * torch.outer(spins, spins))
            - torch.sum(external_field * spins)
        )
        
        # Verify CPU computation
        assert energy.device == cpu_device
        assert torch.isfinite(energy)
        
        # Should work regardless of original device preference
        assert True  # Test passes if no exceptions thrown