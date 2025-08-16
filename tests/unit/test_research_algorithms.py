"""Unit tests for novel research algorithms."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

# Mock imports for testing without dependencies
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'numpy': MagicMock(),
}):
    from spin_glass_rl.research.novel_algorithms import (
        AdaptiveQuantumInspiredAnnealing,
        MultiScaleHierarchicalOptimization, 
        LearningEnhancedSpinDynamics
    )
    from spin_glass_rl.research.experimental_validation import ExperimentalFramework


class TestAdaptiveQuantumInspiredAnnealing:
    """Test Adaptive Quantum-Inspired Annealing (AQIA)."""
    
    @pytest.fixture
    def aqia_config(self):
        """Create AQIA configuration."""
        return {
            'initial_field_strength': 5.0,
            'field_decay_rate': 0.1,
            'tunnel_threshold': 0.8,
            'adaptation_rate': 0.05
        }
    
    @pytest.fixture
    def aqia(self, aqia_config):
        """Create AQIA instance.""" 
        return AdaptiveQuantumInspiredAnnealing(**aqia_config)
    
    def test_initialization(self, aqia, aqia_config):
        """Test AQIA initialization."""
        assert aqia.initial_field_strength == aqia_config['initial_field_strength']
        assert aqia.field_decay_rate == aqia_config['field_decay_rate']
        assert aqia.tunnel_threshold == aqia_config['tunnel_threshold']
        assert aqia.adaptation_rate == aqia_config['adaptation_rate']
    
    def test_quantum_field_update(self, aqia):
        """Test quantum field strength adaptation."""
        initial_field = 5.0
        acceptance_rate = 0.3
        
        new_field = aqia.update_quantum_field(initial_field, acceptance_rate)
        
        # Field should adapt based on acceptance rate
        assert isinstance(new_field, float)
        assert new_field > 0.0
    
    def test_tunneling_probability(self, aqia):
        """Test quantum tunneling probability calculation."""
        energy_barrier = 2.0
        field_strength = 1.0
        
        prob = aqia.calculate_tunneling_probability(energy_barrier, field_strength)
        
        assert 0.0 <= prob <= 1.0
    
    def test_adaptive_step(self, aqia):
        """Test adaptive optimization step."""
        # Mock model and current state
        mock_model = Mock()
        mock_model.compute_energy.return_value = 1.5
        mock_model.n_spins = 10
        
        initial_energy = 2.0
        current_field = 1.0
        
        result = aqia.adaptive_step(mock_model, initial_energy, current_field)
        
        assert 'energy' in result
        assert 'field_strength' in result
        assert 'tunnel_events' in result


class TestMultiScaleHierarchicalOptimization:
    """Test Multi-Scale Hierarchical Optimization (MSHO)."""
    
    @pytest.fixture
    def msho_config(self):
        """Create MSHO configuration."""
        return {
            'scale_levels': [4, 2, 1],
            'convergence_threshold': 1e-6,
            'max_iterations_per_scale': 100,
            'information_transfer_rate': 0.8
        }
    
    @pytest.fixture
    def msho(self, msho_config):
        """Create MSHO instance."""
        return MultiScaleHierarchicalOptimization(**msho_config)
    
    def test_scale_decomposition(self, msho):
        """Test problem decomposition across scales."""
        problem_size = 64
        
        decomposed = msho.decompose_problem(problem_size)
        
        assert isinstance(decomposed, list)
        assert len(decomposed) == len(msho.scale_levels)
        
        # Each scale should have appropriate size
        for i, scale_size in enumerate(decomposed):
            expected_size = problem_size // msho.scale_levels[i]
            assert scale_size == expected_size
    
    def test_information_transfer(self, msho):
        """Test information transfer between scales."""
        coarse_solution = np.random.randint(0, 2, 16) * 2 - 1  # Mock spin configuration
        target_size = 64
        
        refined_solution = msho.transfer_information(coarse_solution, target_size)
        
        assert refined_solution.shape[0] == target_size
        assert np.all(np.isin(refined_solution, [-1, 1]))
    
    def test_hierarchical_optimization(self, msho):
        """Test full hierarchical optimization."""
        # Mock problem and initial solution
        mock_problem = Mock()
        mock_problem.n_spins = 64
        mock_problem.compute_energy = Mock(return_value=1.0)
        
        result = msho.optimize(mock_problem)
        
        assert 'solution' in result
        assert 'energy_trajectory' in result
        assert 'scale_results' in result
        assert len(result['scale_results']) == len(msho.scale_levels)


class TestLearningEnhancedSpinDynamics:
    """Test Learning-Enhanced Spin Dynamics (LESD)."""
    
    @pytest.fixture
    def lesd_config(self):
        """Create LESD configuration."""
        return {
            'network_hidden_dim': 64,
            'learning_rate': 0.001,
            'experience_buffer_size': 1000,
            'batch_size': 32,
            'update_frequency': 10
        }
    
    @pytest.fixture
    def lesd(self, lesd_config):
        """Create LESD instance."""
        return LearningEnhancedSpinDynamics(**lesd_config)
    
    def test_neural_network_initialization(self, lesd):
        """Test neural network setup."""
        assert hasattr(lesd, 'policy_network')
        assert hasattr(lesd, 'value_network')
        assert lesd.policy_network is not None
    
    def test_experience_replay(self, lesd):
        """Test experience replay mechanism."""
        # Add some mock experiences
        for i in range(10):
            state = np.random.rand(20)
            action = np.random.randint(0, 20)
            reward = np.random.rand()
            next_state = np.random.rand(20)
            
            lesd.add_experience(state, action, reward, next_state)
        
        assert len(lesd.experience_buffer) == 10
        
        # Test sampling
        batch = lesd.sample_experience_batch(5)
        assert len(batch) == 5
    
    def test_policy_update(self, lesd):
        """Test policy network update."""
        # Mock state and reward data
        states = np.random.rand(32, 20)
        actions = np.random.randint(0, 20, 32)
        rewards = np.random.rand(32)
        
        initial_loss = lesd.get_current_loss()
        lesd.update_policy(states, actions, rewards)
        final_loss = lesd.get_current_loss()
        
        # Loss should be computed (not necessarily decreased in one step)
        assert final_loss is not None
    
    def test_adaptive_spin_selection(self, lesd):
        """Test learning-guided spin selection."""
        # Mock system state
        system_state = np.random.rand(20)
        temperature = 1.0
        
        selected_spins = lesd.select_spins_to_flip(system_state, temperature, n_flips=5)
        
        assert len(selected_spins) == 5
        assert all(0 <= spin < 20 for spin in selected_spins)


class TestExperimentalFramework:
    """Test experimental validation framework."""
    
    @pytest.fixture
    def framework(self):
        """Create experimental framework."""
        return ExperimentalFramework()
    
    def test_benchmark_suite_loading(self, framework):
        """Test benchmark problem loading."""
        problems = framework.load_benchmark_suite(['random_3sat', 'max_cut'])
        
        assert isinstance(problems, list)
        assert len(problems) >= 2
    
    def test_statistical_validation(self, framework):
        """Test statistical significance testing."""
        # Mock experimental results
        algorithm_a_results = np.random.normal(100, 10, 30)
        algorithm_b_results = np.random.normal(105, 10, 30)
        
        stats = framework.statistical_comparison(algorithm_a_results, algorithm_b_results)
        
        assert 'p_value' in stats
        assert 'effect_size' in stats
        assert 'confidence_interval' in stats
        assert 'significant' in stats
    
    def test_reproducibility_validation(self, framework):
        """Test reproducibility across multiple runs."""
        # Mock algorithm function
        def mock_algorithm(problem, seed):
            np.random.seed(seed)
            return np.random.normal(100, 5)
        
        mock_problem = Mock()
        
        results = framework.test_reproducibility(mock_algorithm, mock_problem, n_runs=5)
        
        assert 'mean_result' in results
        assert 'std_result' in results
        assert 'reproducibility_score' in results
        assert len(results['individual_runs']) == 5
    
    def test_performance_profiling(self, framework):
        """Test algorithm performance profiling."""
        def mock_algorithm(problem):
            import time
            time.sleep(0.01)  # Simulate computation
            return {'energy': 1.0, 'iterations': 100}
        
        mock_problem = Mock()
        
        profile = framework.profile_algorithm(mock_algorithm, mock_problem)
        
        assert 'execution_time' in profile
        assert 'memory_usage' in profile
        assert 'result' in profile
        assert profile['execution_time'] > 0


# Integration test for research pipeline
class TestResearchIntegration:
    """Test integration of research algorithms."""
    
    def test_algorithm_comparison(self):
        """Test comparative evaluation of algorithms."""
        # Mock problem
        mock_problem = Mock()
        mock_problem.n_spins = 50
        mock_problem.compute_energy = Mock(return_value=1.0)
        
        # Initialize algorithms
        aqia = AdaptiveQuantumInspiredAnnealing()
        msho = MultiScaleHierarchicalOptimization()
        lesd = LearningEnhancedSpinDynamics()
        
        algorithms = {'AQIA': aqia, 'MSHO': msho, 'LESD': lesd}
        
        # Test that all algorithms can be initialized
        for name, alg in algorithms.items():
            assert alg is not None
            assert hasattr(alg, 'optimize') or hasattr(alg, 'adaptive_step')
    
    def test_research_pipeline_integration(self):
        """Test full research validation pipeline."""
        framework = ExperimentalFramework()
        
        # Mock algorithm results
        results = {
            'AQIA': [95, 98, 92, 97, 94],
            'MSHO': [88, 91, 89, 90, 87], 
            'LESD': [102, 105, 101, 104, 103],
            'Baseline': [80, 82, 79, 81, 78]
        }
        
        # Validate research quality
        validation_report = framework.generate_research_report(results)
        
        assert 'statistical_tests' in validation_report
        assert 'effect_sizes' in validation_report
        assert 'publication_ready' in validation_report
        assert validation_report['publication_ready'] == True


if __name__ == '__main__':
    pytest.main([__file__])