"""Unit tests for GPU annealer implementation."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.annealing.result import AnnealingResult
from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.spin_dynamics import UpdateRule
from spin_glass_rl.utils.exceptions import AnnealingError, DeviceError


class TestGPUAnnealerConfig:
    """Test GPUAnnealerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GPUAnnealerConfig()
        
        assert config.n_sweeps == 1000
        assert config.initial_temp == 10.0
        assert config.final_temp == 0.01
        assert config.schedule_type == ScheduleType.GEOMETRIC
        assert config.schedule_params == {"alpha": 0.95}
        assert config.block_size == 256
        assert config.record_interval == 10
        assert config.random_seed is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPUAnnealerConfig(
            n_sweeps=2000,
            initial_temp=5.0,
            final_temp=0.001,
            schedule_type=ScheduleType.LINEAR,
            schedule_params={"custom": "value"},
            record_interval=5,
            random_seed=42
        )
        
        assert config.n_sweeps == 2000
        assert config.initial_temp == 5.0
        assert config.final_temp == 0.001
        assert config.schedule_type == ScheduleType.LINEAR
        assert config.schedule_params == {"custom": "value"}
        assert config.record_interval == 5
        assert config.random_seed == 42
    
    def test_post_init(self):
        """Test __post_init__ method."""
        config = GPUAnnealerConfig(schedule_params=None)
        assert config.schedule_params == {"alpha": 0.95}


class TestGPUAnnealer:
    """Test GPUAnnealer class."""
    
    @pytest.fixture
    def config(self):
        """Create test annealer configuration."""
        return GPUAnnealerConfig(
            n_sweeps=100,
            initial_temp=2.0,
            final_temp=0.1,
            random_seed=42
        )
    
    @pytest.fixture
    def annealer(self, config):
        """Create test GPU annealer."""
        return GPUAnnealer(config)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple Ising model for testing."""
        config = IsingModelConfig(n_spins=10, device="cpu")
        model = IsingModel(config)
        
        # Add some couplings
        model.set_coupling(0, 1, -1.0)
        model.set_coupling(1, 2, 1.0)
        model.set_external_field(0, 0.5)
        
        return model
    
    def test_initialization(self, config):
        """Test annealer initialization."""
        annealer = GPUAnnealer(config)
        
        assert annealer.config == config
        assert annealer.device is not None
        assert annealer.total_flips == 0
        assert annealer.total_time == 0.0
    
    def test_device_selection(self):
        """Test device selection logic."""
        config = GPUAnnealerConfig()
        annealer = GPUAnnealer(config)
        
        # Should select appropriate device
        if torch.cuda.is_available():
            assert annealer.device.type == "cuda"
            assert annealer.use_cuda == True
        else:
            assert annealer.device.type == "cpu"
            assert annealer.use_cuda == False
    
    def test_anneal_basic(self, annealer, simple_model):
        """Test basic annealing functionality."""
        initial_energy = simple_model.compute_energy()
        
        result = annealer.anneal(simple_model)
        
        # Verify result structure
        assert isinstance(result, AnnealingResult)
        assert result.best_configuration is not None
        assert isinstance(result.best_energy, float)
        assert len(result.energy_history) > 0
        assert len(result.temperature_history) > 0
        assert len(result.acceptance_rate_history) > 0
        assert result.total_time > 0
        assert result.n_sweeps > 0
    
    def test_anneal_with_different_update_rules(self, annealer, simple_model):
        """Test annealing with different update rules."""
        update_rules = [UpdateRule.METROPOLIS, UpdateRule.GLAUBER, UpdateRule.HEAT_BATH]
        
        results = {}
        for rule in update_rules:
            result = annealer.anneal(simple_model.copy(), update_rule=rule)
            results[rule] = result
            
            assert isinstance(result, AnnealingResult)
            assert torch.isfinite(torch.tensor(result.best_energy))
    
    def test_convergence_detection(self, simple_model):
        """Test convergence detection."""
        config = GPUAnnealerConfig(
            n_sweeps=1000,
            energy_tolerance=1e-8,
            random_seed=42
        )
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        # Should detect convergence or complete all sweeps
        assert result.n_sweeps <= config.n_sweeps
    
    def test_temperature_scheduling(self, annealer, simple_model):
        """Test temperature scheduling during annealing."""
        result = annealer.anneal(simple_model)
        
        # Temperature should decrease over time
        temps = result.temperature_history
        assert temps[0] > temps[-1]
        
        # Should start near initial temperature
        assert abs(temps[0] - annealer.config.initial_temp) < 0.1
    
    def test_energy_improvement(self, annealer, simple_model):
        """Test that annealing generally improves energy."""
        initial_energy = simple_model.compute_energy()
        
        result = annealer.anneal(simple_model)
        
        # Energy should generally improve (or at least not get much worse)
        # Allow some tolerance for stochastic optimization
        assert result.best_energy <= initial_energy + 1.0
    
    def test_move_model_to_gpu(self, annealer):
        """Test moving model to GPU."""
        cpu_config = IsingModelConfig(n_spins=5, device="cpu")
        cpu_model = IsingModel(cpu_config)
        
        if annealer.use_cuda:
            gpu_model = annealer._move_model_to_gpu(cpu_model)
            assert gpu_model.device.type == "cuda"
            assert gpu_model.n_spins == cpu_model.n_spins
        else:
            # Should handle gracefully when CUDA not available
            gpu_model = annealer._move_model_to_gpu(cpu_model)
            assert gpu_model.device.type == "cpu"
    
    def test_memory_usage_tracking(self, annealer):
        """Test GPU memory usage tracking."""
        memory_info = annealer.get_memory_usage()
        
        assert "device" in memory_info
        assert "memory_allocated" in memory_info
        assert "memory_reserved" in memory_info
        
        if annealer.use_cuda:
            assert memory_info["device"] == str(annealer.device)
            assert memory_info["memory_allocated"] >= 0
            assert memory_info["memory_reserved"] >= 0
    
    def test_benchmark_functionality(self, annealer):
        """Test benchmarking functionality."""
        model_sizes = [5, 10, 15]
        n_trials = 2
        
        results = annealer.benchmark(model_sizes, n_trials)
        
        assert len(results) == len(model_sizes)
        
        for size in model_sizes:
            assert size in results
            size_result = results[size]
            
            assert "mean_time" in size_result
            assert "std_time" in size_result
            assert "mean_energy" in size_result
            assert "mean_sps" in size_result  # sweeps per second
            
            assert size_result["mean_time"] > 0
            assert size_result["mean_sps"] > 0
    
    def test_reproducibility(self, simple_model):
        """Test reproducibility with fixed random seed."""
        config1 = GPUAnnealerConfig(n_sweeps=50, random_seed=42)
        config2 = GPUAnnealerConfig(n_sweeps=50, random_seed=42)
        
        annealer1 = GPUAnnealer(config1)
        annealer2 = GPUAnnealer(config2)
        
        result1 = annealer1.anneal(simple_model.copy())
        result2 = annealer2.anneal(simple_model.copy())
        
        # Results should be identical with same seed
        assert result1.best_energy == result2.best_energy
        assert torch.allclose(result1.best_configuration, result2.best_configuration)
    
    def test_different_seeds(self, simple_model):
        """Test different results with different seeds."""
        config1 = GPUAnnealerConfig(n_sweeps=50, random_seed=42)
        config2 = GPUAnnealerConfig(n_sweeps=50, random_seed=123)
        
        annealer1 = GPUAnnealer(config1)
        annealer2 = GPUAnnealer(config2)
        
        result1 = annealer1.anneal(simple_model.copy())
        result2 = annealer2.anneal(simple_model.copy())
        
        # Results should likely be different with different seeds
        # (though there's a small chance they could be the same)
        assert (result1.best_energy != result2.best_energy or 
                not torch.allclose(result1.best_configuration, result2.best_configuration))
    
    def test_progress_recording(self, annealer, simple_model):
        """Test progress recording during annealing."""
        annealer.config.record_interval = 10
        
        result = annealer.anneal(simple_model)
        
        # Should have recorded progress at intervals
        expected_records = annealer.config.n_sweeps // annealer.config.record_interval
        assert len(result.energy_history) >= expected_records
        assert len(result.temperature_history) >= expected_records
        assert len(result.acceptance_rate_history) >= expected_records
    
    def test_error_handling(self):
        """Test error handling in annealer."""
        config = GPUAnnealerConfig()
        annealer = GPUAnnealer(config)
        
        # Test with invalid model
        with pytest.raises((AttributeError, TypeError)):
            annealer.anneal("not a model")
    
    def test_cpu_vs_gpu_consistency(self, simple_model):
        """Test CPU vs GPU consistency when both available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # CPU annealer
        cpu_config = GPUAnnealerConfig(n_sweeps=50, random_seed=42)
        cpu_model = simple_model.copy()
        cpu_model.spins = cpu_model.spins.to("cpu")
        cpu_model.couplings = cpu_model.couplings.to("cpu")
        cpu_model.external_fields = cpu_model.external_fields.to("cpu")
        
        cpu_annealer = GPUAnnealer(cpu_config)
        cpu_annealer.device = torch.device("cpu")
        cpu_annealer.use_cuda = False
        
        cpu_result = cpu_annealer.anneal(cpu_model)
        
        # GPU annealer
        gpu_config = GPUAnnealerConfig(n_sweeps=50, random_seed=42)
        gpu_annealer = GPUAnnealer(gpu_config)
        
        gpu_result = gpu_annealer.anneal(simple_model.copy())
        
        # Results should be similar (though not identical due to different precision)
        assert abs(cpu_result.best_energy - gpu_result.best_energy) < 1.0
    
    def test_repr(self, annealer):
        """Test string representation."""
        repr_str = repr(annealer)
        
        assert "GPUAnnealer" in repr_str
        assert str(annealer.device) in repr_str
        assert str(annealer.config.n_sweeps) in repr_str
        assert annealer.config.schedule_type.value in repr_str


class TestGPUAnnealerEdgeCases:
    """Test edge cases for GPU annealer."""
    
    def test_zero_sweeps(self, simple_model):
        """Test annealing with zero sweeps."""
        config = GPUAnnealerConfig(n_sweeps=0)
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        assert result.n_sweeps == 0
        assert result.total_time >= 0
        assert len(result.energy_history) >= 1  # Initial energy
    
    def test_single_sweep(self, simple_model):
        """Test annealing with single sweep."""
        config = GPUAnnealerConfig(n_sweeps=1)
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        assert result.n_sweeps == 1
        assert len(result.energy_history) >= 1
    
    def test_very_high_temperature(self, simple_model):
        """Test annealing with very high initial temperature."""
        config = GPUAnnealerConfig(
            n_sweeps=50,
            initial_temp=1000.0,
            final_temp=1.0
        )
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        # Should complete without errors
        assert torch.isfinite(torch.tensor(result.best_energy))
        assert result.n_sweeps == 50
    
    def test_very_low_temperature(self, simple_model):
        """Test annealing with very low temperature."""
        config = GPUAnnealerConfig(
            n_sweeps=50,
            initial_temp=1e-6,
            final_temp=1e-8
        )
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        # Should complete without errors (though acceptance will be very low)
        assert torch.isfinite(torch.tensor(result.best_energy))
    
    def test_large_model(self):
        """Test annealing with large model."""
        large_config = IsingModelConfig(n_spins=500, device="cpu")
        large_model = IsingModel(large_config)
        
        # Add some random couplings
        for _ in range(100):
            i, j = np.random.randint(0, 500, 2)
            if i != j:
                large_model.set_coupling(i, j, np.random.randn())
        
        annealer_config = GPUAnnealerConfig(n_sweeps=10)  # Keep short for testing
        annealer = GPUAnnealer(annealer_config)
        
        result = annealer.anneal(large_model)
        
        assert result.best_configuration.shape == (500,)
        assert torch.isfinite(torch.tensor(result.best_energy))
    
    def test_inverted_temperature_schedule(self, simple_model):
        """Test with final temperature higher than initial (should handle gracefully)."""
        config = GPUAnnealerConfig(
            n_sweeps=50,
            initial_temp=1.0,
            final_temp=10.0  # Higher than initial
        )
        annealer = GPUAnnealer(config)
        
        # Should handle this gracefully (might behave like heating)
        result = annealer.anneal(simple_model)
        assert torch.isfinite(torch.tensor(result.best_energy))
    
    @pytest.mark.slow
    def test_very_long_annealing(self, simple_model):
        """Test very long annealing run."""
        config = GPUAnnealerConfig(n_sweeps=10000)
        annealer = GPUAnnealer(config)
        
        result = annealer.anneal(simple_model)
        
        # Should eventually converge or complete
        assert result.n_sweeps <= 10000
        assert torch.isfinite(torch.tensor(result.best_energy))