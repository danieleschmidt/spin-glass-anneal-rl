"""Unit tests for annealing components."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import time

# Mock imports to avoid dependency issues
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'numpy': MagicMock(),
}):
    from spin_glass_rl.annealing.temperature_scheduler import (
        TemperatureScheduler, ScheduleType, GeometricSchedule, LinearSchedule
    )
    from spin_glass_rl.annealing.parallel_tempering import (
        ParallelTempering, ReplicaExchange
    )
    from spin_glass_rl.annealing.batch_processor import (
        BatchAnnealer, BatchConfig
    )
    from spin_glass_rl.annealing.result import AnnealingResult


class TestTemperatureScheduler:
    """Test temperature scheduling algorithms."""
    
    def test_geometric_schedule_creation(self):
        """Test geometric temperature schedule."""
        schedule = TemperatureScheduler.create_schedule(
            ScheduleType.GEOMETRIC,
            initial_temp=10.0,
            final_temp=0.1,
            total_sweeps=100,
            alpha=0.95
        )
        
        assert isinstance(schedule, GeometricSchedule)
        assert schedule.initial_temp == 10.0
        assert schedule.final_temp == 0.1
        assert schedule.alpha == 0.95
    
    def test_linear_schedule_creation(self):
        """Test linear temperature schedule."""
        schedule = TemperatureScheduler.create_schedule(
            ScheduleType.LINEAR,
            initial_temp=5.0,
            final_temp=0.5,
            total_sweeps=50
        )
        
        assert isinstance(schedule, LinearSchedule)
        assert schedule.initial_temp == 5.0
        assert schedule.final_temp == 0.5
    
    def test_geometric_temperature_progression(self):
        """Test geometric schedule temperature progression."""
        schedule = GeometricSchedule(
            initial_temp=10.0,
            final_temp=0.1,
            alpha=0.9
        )
        
        temp_0 = schedule.get_temperature(0)
        temp_10 = schedule.get_temperature(10)
        temp_20 = schedule.get_temperature(20)
        
        assert temp_0 == 10.0
        assert temp_10 < temp_0
        assert temp_20 < temp_10
        assert temp_20 > 0
    
    def test_linear_temperature_progression(self):
        """Test linear schedule temperature progression."""
        schedule = LinearSchedule(
            initial_temp=8.0,
            final_temp=0.2,
            total_sweeps=100
        )
        
        temp_0 = schedule.get_temperature(0)
        temp_50 = schedule.get_temperature(50)
        temp_100 = schedule.get_temperature(100)
        
        assert temp_0 == 8.0
        assert temp_100 == 0.2
        assert temp_50 == pytest.approx((8.0 + 0.2) / 2, rel=0.1)
    
    def test_adaptive_schedule_update(self):
        """Test adaptive temperature schedule."""
        schedule = TemperatureScheduler.create_schedule(
            ScheduleType.ADAPTIVE,
            initial_temp=5.0,
            final_temp=0.1,
            total_sweeps=100,
            target_acceptance=0.3
        )
        
        # Test adaptation with different acceptance rates
        temp_1 = schedule.update(10, acceptance_rate=0.5)  # High acceptance
        temp_2 = schedule.update(20, acceptance_rate=0.1)  # Low acceptance
        
        assert temp_1 != temp_2
        assert temp_1 > 0 and temp_2 > 0


class TestParallelTempering:
    """Test parallel tempering implementation."""
    
    @pytest.fixture
    def pt_config(self):
        """Create parallel tempering configuration."""
        return {
            'n_replicas': 4,
            'temp_min': 0.1,
            'temp_max': 10.0,
            'exchange_interval': 10,
            'temp_distribution': 'geometric'
        }
    
    @pytest.fixture
    def parallel_tempering(self, pt_config):
        """Create parallel tempering instance."""
        return ParallelTempering(**pt_config)
    
    def test_temperature_ladder_creation(self, parallel_tempering):
        """Test temperature ladder setup."""
        temps = parallel_tempering.get_temperature_ladder()
        
        assert len(temps) == 4
        assert temps[0] == 0.1  # Minimum temperature
        assert temps[-1] == 10.0  # Maximum temperature
        assert all(temps[i] <= temps[i+1] for i in range(len(temps)-1))
    
    def test_replica_initialization(self, parallel_tempering):
        """Test replica initialization."""
        mock_problem = Mock()
        mock_problem.n_spins = 20
        mock_problem.copy = Mock(return_value=mock_problem)
        
        parallel_tempering.initialize_replicas(mock_problem)
        
        assert len(parallel_tempering.replicas) == 4
        assert all(replica is not None for replica in parallel_tempering.replicas)
    
    def test_exchange_probability_calculation(self, parallel_tempering):
        """Test replica exchange probability."""
        energy_1 = 5.0
        energy_2 = 7.0
        temp_1 = 1.0
        temp_2 = 2.0
        
        prob = parallel_tempering.calculate_exchange_probability(
            energy_1, energy_2, temp_1, temp_2
        )
        
        assert 0.0 <= prob <= 1.0
    
    def test_replica_exchange(self, parallel_tempering):
        """Test replica exchange mechanism."""
        # Mock replicas with different energies
        replica_1 = Mock()
        replica_1.compute_energy = Mock(return_value=3.0)
        replica_2 = Mock()
        replica_2.compute_energy = Mock(return_value=5.0)
        
        parallel_tempering.replicas = [replica_1, replica_2]
        parallel_tempering.temperatures = [1.0, 2.0]
        
        exchanges = parallel_tempering.attempt_exchanges()
        
        assert isinstance(exchanges, int)
        assert exchanges >= 0
    
    def test_parallel_tempering_step(self, parallel_tempering):
        """Test single parallel tempering step."""
        mock_problem = Mock()
        mock_problem.n_spins = 10
        mock_problem.copy = Mock(return_value=mock_problem)
        mock_problem.compute_energy = Mock(return_value=2.0)
        
        parallel_tempering.initialize_replicas(mock_problem)
        
        result = parallel_tempering.step()
        
        assert 'best_energy' in result
        assert 'exchange_count' in result
        assert 'replica_energies' in result


class TestBatchAnnealer:
    """Test batch annealing processor."""
    
    @pytest.fixture
    def batch_config(self):
        """Create batch configuration."""
        return BatchConfig(
            batch_size=8,
            max_workers=2,
            memory_limit_mb=500,
            enable_gpu_batching=False
        )
    
    @pytest.fixture
    def batch_annealer(self, batch_config):
        """Create batch annealer."""
        return BatchAnnealer(batch_config)
    
    def test_problem_batching(self, batch_annealer):
        """Test problem batching."""
        # Create mock problems
        problems = []
        for i in range(10):
            problem = Mock()
            problem.n_spins = 20
            problem.id = f"problem_{i}"
            problems.append(problem)
        
        batches = batch_annealer.create_batches(problems)
        
        assert len(batches) == 2  # 10 problems / 8 batch_size = 1.25 -> 2 batches
        assert len(batches[0]) == 8
        assert len(batches[1]) == 2
    
    def test_parallel_batch_processing(self, batch_annealer):
        """Test parallel processing of problem batches."""
        def mock_annealing_function(problem):
            return {
                'best_energy': np.random.rand(),
                'solution': np.random.randint(0, 2, problem.n_spins) * 2 - 1,
                'iterations': 100
            }
        
        # Create mock problems
        problems = []
        for i in range(5):
            problem = Mock()
            problem.n_spins = 15
            problems.append(problem)
        
        results = batch_annealer.process_batch_parallel(problems, mock_annealing_function)
        
        assert len(results) == 5
        assert all('best_energy' in result for result in results)
        assert all('solution' in result for result in results)
    
    def test_memory_efficient_processing(self, batch_annealer):
        """Test memory-efficient batch processing."""
        def memory_intensive_annealing(problem):
            # Simulate memory usage
            large_array = np.random.rand(1000, 1000)  # 8MB array
            return {
                'energy': np.sum(large_array) % 100,
                'memory_used': large_array.nbytes
            }
        
        # Create problems that would exceed memory limit if processed together
        problems = [Mock(n_spins=20) for _ in range(5)]
        
        results = batch_annealer.process_with_memory_limit(
            problems, 
            memory_intensive_annealing
        )
        
        assert len(results) == 5
        assert all('energy' in result for result in results)
    
    def test_gpu_batch_coordination(self, batch_annealer):
        """Test GPU batch coordination."""
        if not batch_annealer.config.enable_gpu_batching:
            batch_annealer.config.enable_gpu_batching = True
        
        mock_problems = [Mock(n_spins=30) for _ in range(4)]
        
        # Mock GPU processing function
        def gpu_batch_process(problem_batch):
            return [{
                'energy': i * 0.5,
                'gpu_processed': True
            } for i in range(len(problem_batch))]
        
        results = batch_annealer.process_gpu_batch(mock_problems, gpu_batch_process)
        
        assert len(results) == 4
        assert all(result['gpu_processed'] for result in results)


class TestAnnealingResult:
    """Test annealing result data structure."""
    
    def test_result_creation(self):
        """Test annealing result creation."""
        config = np.array([1, -1, 1, -1, 1])
        energy = -2.5
        history = [-1.0, -1.5, -2.0, -2.5]
        
        result = AnnealingResult(
            best_configuration=config,
            best_energy=energy,
            energy_history=history,
            temperature_history=[2.0, 1.5, 1.0, 0.5],
            acceptance_rate_history=[0.8, 0.6, 0.4, 0.2],
            total_time=1.5,
            n_sweeps=1000,
            algorithm="simulated_annealing"
        )
        
        assert np.array_equal(result.best_configuration, config)
        assert result.best_energy == energy
        assert result.energy_history == history
        assert result.total_time == 1.5
        assert result.n_sweeps == 1000
    
    def test_result_statistics(self):
        """Test result statistics calculation."""
        result = AnnealingResult(
            best_configuration=np.array([1, -1]),
            best_energy=-1.0,
            energy_history=[-0.5, -0.7, -0.9, -1.0],
            temperature_history=[2.0, 1.0, 0.5, 0.1],
            acceptance_rate_history=[0.9, 0.7, 0.5, 0.3],
            total_time=2.0,
            n_sweeps=500
        )
        
        stats = result.get_statistics()
        
        assert 'energy_improvement' in stats
        assert 'final_acceptance_rate' in stats
        assert 'convergence_rate' in stats
        assert 'sweeps_per_second' in stats
        
        assert stats['energy_improvement'] == 0.5  # -0.5 to -1.0
        assert stats['final_acceptance_rate'] == 0.3
        assert stats['sweeps_per_second'] == 250.0  # 500 sweeps / 2.0 seconds
    
    def test_result_export(self):
        """Test result export functionality."""
        result = AnnealingResult(
            best_configuration=np.array([1, -1, 1]),
            best_energy=-1.5,
            energy_history=[-1.0, -1.2, -1.5],
            temperature_history=[1.0, 0.5, 0.1],
            acceptance_rate_history=[0.8, 0.6, 0.4],
            total_time=0.5,
            n_sweeps=300
        )
        
        exported = result.to_dict()
        
        assert 'best_configuration' in exported
        assert 'best_energy' in exported
        assert 'energy_history' in exported
        assert 'metadata' in exported
        
        # Test configuration conversion
        assert exported['best_configuration'] == [1, -1, 1]
        assert exported['best_energy'] == -1.5
    
    def test_result_comparison(self):
        """Test result comparison methods."""
        result1 = AnnealingResult(
            best_configuration=np.array([1, -1]),
            best_energy=-2.0,
            energy_history=[-1.0, -2.0],
            total_time=1.0
        )
        
        result2 = AnnealingResult(
            best_configuration=np.array([-1, 1]),
            best_energy=-1.5,
            energy_history=[-1.0, -1.5],
            total_time=1.2
        )
        
        assert result1.is_better_than(result2)  # Lower energy is better
        assert not result2.is_better_than(result1)
        
        # Test comparison metrics
        comparison = result1.compare_with(result2)
        
        assert 'energy_difference' in comparison
        assert 'time_difference' in comparison
        assert comparison['energy_difference'] == -0.5  # result1 is 0.5 lower
        assert comparison['time_difference'] == -0.2    # result1 is 0.2s faster


# Integration tests
class TestAnnealingIntegration:
    """Test integration of annealing components."""
    
    def test_parallel_tempering_with_batch_processing(self):
        """Test parallel tempering with batch processing."""
        # Create parallel tempering setup
        pt = ParallelTempering(
            n_replicas=2,
            temp_min=0.5,
            temp_max=2.0,
            exchange_interval=5
        )
        
        # Create batch processor
        batch_config = BatchConfig(batch_size=4, max_workers=1)
        batch_processor = BatchAnnealer(batch_config)
        
        # Mock problems
        problems = [Mock(n_spins=10) for _ in range(4)]
        
        def pt_annealing_function(problem):
            pt.initialize_replicas(problem)
            results = []
            for _ in range(10):
                step_result = pt.step()
                results.append(step_result)
            
            return {
                'best_energy': min(r['best_energy'] for r in results),
                'final_step': results[-1]
            }
        
        batch_results = batch_processor.process_batch_parallel(
            problems, 
            pt_annealing_function
        )
        
        assert len(batch_results) == 4
        assert all('best_energy' in result for result in batch_results)
    
    def test_adaptive_temperature_with_parallel_tempering(self):
        """Test adaptive temperature scheduling with parallel tempering."""
        # Create adaptive scheduler
        adaptive_schedule = TemperatureScheduler.create_schedule(
            ScheduleType.ADAPTIVE,
            initial_temp=3.0,
            final_temp=0.1,
            total_sweeps=100,
            target_acceptance=0.4
        )
        
        # Create parallel tempering with adaptive temperatures
        pt = ParallelTempering(n_replicas=3, temp_min=0.1, temp_max=3.0)
        
        # Mock problem
        mock_problem = Mock()
        mock_problem.n_spins = 15
        mock_problem.copy = Mock(return_value=mock_problem)
        mock_problem.compute_energy = Mock(return_value=1.0)
        
        pt.initialize_replicas(mock_problem)
        
        # Run with adaptive temperature updates
        results = []
        for sweep in range(20):
            step_result = pt.step()
            
            # Update temperature adaptively for each replica
            for i, replica in enumerate(pt.replicas):
                acceptance_rate = 0.3 + 0.1 * i  # Mock different acceptance rates
                new_temp = adaptive_schedule.update(sweep, acceptance_rate)
                pt.temperatures[i] = new_temp
            
            results.append(step_result)
        
        assert len(results) == 20
        assert all('best_energy' in result for result in results)
    
    def test_complete_annealing_pipeline(self):
        """Test complete annealing pipeline integration."""
        # Create components
        temperature_schedule = TemperatureScheduler.create_schedule(
            ScheduleType.GEOMETRIC,
            initial_temp=5.0,
            final_temp=0.05,
            total_sweeps=100,
            alpha=0.95
        )
        
        batch_config = BatchConfig(batch_size=2, max_workers=1)
        batch_processor = BatchAnnealer(batch_config)
        
        # Mock optimization pipeline
        def complete_annealing_pipeline(problem):
            results = []
            
            for sweep in range(10):
                temp = temperature_schedule.get_temperature(sweep)
                
                # Mock annealing step
                energy = 5.0 - sweep * 0.3  # Decreasing energy
                
                results.append({
                    'sweep': sweep,
                    'temperature': temp,
                    'energy': energy
                })
            
            final_result = AnnealingResult(
                best_configuration=np.random.randint(0, 2, problem.n_spins) * 2 - 1,
                best_energy=min(r['energy'] for r in results),
                energy_history=[r['energy'] for r in results],
                temperature_history=[r['temperature'] for r in results],
                total_time=0.1,
                n_sweeps=10
            )
            
            return final_result
        
        # Test with multiple problems
        problems = [Mock(n_spins=8) for _ in range(3)]
        
        final_results = batch_processor.process_batch_parallel(
            problems,
            complete_annealing_pipeline
        )
        
        assert len(final_results) == 3
        assert all(isinstance(result, AnnealingResult) for result in final_results)
        assert all(len(result.energy_history) == 10 for result in final_results)


if __name__ == '__main__':
    pytest.main([__file__])