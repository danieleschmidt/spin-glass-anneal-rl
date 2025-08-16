"""Unit tests for optimization components."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import time
import tempfile
import json
from pathlib import Path

# Mock imports to avoid dependency issues
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'numpy': MagicMock(),
}):
    from spin_glass_rl.optimization.adaptive_optimization import (
        AdaptiveSimulatedAnnealing, AdaptiveConfig
    )
    from spin_glass_rl.optimization.performance_cache import (
        LRUCache, ComputationCache, CacheStats
    )
    from spin_glass_rl.optimization.high_performance_computing import (
        BatchProcessor, PerformanceOptimizer, GPUAccelerator
    )


class TestAdaptiveSimulatedAnnealing:
    """Test adaptive simulated annealing optimization."""
    
    @pytest.fixture
    def adaptive_config(self):
        """Create adaptive configuration."""
        return AdaptiveConfig(
            initial_temperature=10.0,
            final_temperature=0.01,
            adaptation_interval=100,
            acceptance_target=0.3,
            learning_rate=0.1
        )
    
    @pytest.fixture
    def adaptive_annealer(self, adaptive_config):
        """Create adaptive annealer."""
        return AdaptiveSimulatedAnnealing(adaptive_config)
    
    def test_initialization(self, adaptive_annealer, adaptive_config):
        """Test adaptive annealer initialization."""
        assert adaptive_annealer.config == adaptive_config
        assert adaptive_annealer.current_temperature == adaptive_config.initial_temperature
        assert hasattr(adaptive_annealer, 'acceptance_history')
    
    def test_temperature_adaptation(self, adaptive_annealer):
        """Test temperature adaptation based on acceptance rate."""
        # Mock acceptance history
        adaptive_annealer.acceptance_history = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        old_temp = adaptive_annealer.current_temperature
        adaptive_annealer.adapt_temperature()
        new_temp = adaptive_annealer.current_temperature
        
        # Temperature should adapt based on acceptance rate
        assert new_temp != old_temp
        assert new_temp > 0
    
    def test_acceptance_rate_calculation(self, adaptive_annealer):
        """Test acceptance rate calculation."""
        adaptive_annealer.accepted_moves = 30
        adaptive_annealer.total_moves = 100
        
        rate = adaptive_annealer.get_acceptance_rate()
        assert rate == 0.3
    
    def test_adaptive_optimization_step(self, adaptive_annealer):
        """Test single adaptive optimization step."""
        # Mock problem
        mock_problem = Mock()
        mock_problem.compute_energy.return_value = 1.5
        mock_problem.n_spins = 10
        
        result = adaptive_annealer.optimization_step(mock_problem)
        
        assert 'energy' in result
        assert 'accepted' in result
        assert 'temperature' in result


class TestLRUCache:
    """Test LRU cache implementation."""
    
    @pytest.fixture
    def lru_cache(self):
        """Create LRU cache."""
        return LRUCache(capacity=3)
    
    def test_cache_insertion(self, lru_cache):
        """Test cache insertion."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        
        assert lru_cache.get("key1") == "value1"
        assert lru_cache.get("key2") == "value2"
        assert lru_cache.size() == 2
    
    def test_cache_eviction(self, lru_cache):
        """Test LRU eviction policy."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2") 
        lru_cache.put("key3", "value3")
        lru_cache.put("key4", "value4")  # Should evict key1
        
        assert lru_cache.get("key1") is None
        assert lru_cache.get("key2") == "value2"
        assert lru_cache.get("key3") == "value3"
        assert lru_cache.get("key4") == "value4"
    
    def test_cache_access_order(self, lru_cache):
        """Test that access updates order."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.put("key3", "value3")
        
        # Access key1 to make it most recent
        lru_cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        lru_cache.put("key4", "value4")
        
        assert lru_cache.get("key1") == "value1"  # Still there
        assert lru_cache.get("key2") is None       # Evicted
        assert lru_cache.get("key3") == "value3"
        assert lru_cache.get("key4") == "value4"


class TestComputationCache:
    """Test computation caching system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def computation_cache(self, temp_cache_dir):
        """Create computation cache."""
        return ComputationCache(
            memory_cache_size=100,
            disk_cache_dir=temp_cache_dir,
            enable_disk_cache=True
        )
    
    def test_function_caching(self, computation_cache):
        """Test function result caching."""
        @computation_cache.cached_function()
        def expensive_function(x, y):
            time.sleep(0.01)  # Simulate computation
            return x + y
        
        # First call should be slow
        start_time = time.time()
        result1 = expensive_function(2, 3)
        first_call_time = time.time() - start_time
        
        # Second call should be fast (cached)
        start_time = time.time()
        result2 = expensive_function(2, 3)
        second_call_time = time.time() - start_time
        
        assert result1 == result2 == 5
        assert second_call_time < first_call_time
    
    def test_cache_key_generation(self, computation_cache):
        """Test cache key generation."""
        def test_func(a, b, c=10):
            return a + b + c
        
        key1 = computation_cache._generate_key(test_func, (1, 2), {'c': 3})
        key2 = computation_cache._generate_key(test_func, (1, 2), {'c': 3})
        key3 = computation_cache._generate_key(test_func, (1, 2), {'c': 4})
        
        assert key1 == key2  # Same parameters
        assert key1 != key3  # Different parameters
    
    def test_cache_statistics(self, computation_cache):
        """Test cache statistics tracking."""
        @computation_cache.cached_function()
        def test_func(x):
            return x * 2
        
        # Generate some cache hits and misses
        test_func(1)  # miss
        test_func(2)  # miss
        test_func(1)  # hit
        test_func(3)  # miss
        test_func(2)  # hit
        
        stats = computation_cache.get_statistics()
        
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        assert stats['cache_hits'] == 2
        assert stats['cache_misses'] == 3


class TestBatchProcessor:
    """Test batch processing optimization."""
    
    @pytest.fixture
    def batch_processor(self):
        """Create batch processor."""
        return BatchProcessor(batch_size=4, max_workers=2)
    
    def test_batch_creation(self, batch_processor):
        """Test batch creation from items."""
        items = list(range(10))
        batches = batch_processor.create_batches(items)
        
        assert len(batches) == 3  # 10 items / 4 batch_size = 2.5 -> 3 batches
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4  
        assert len(batches[2]) == 2
    
    def test_parallel_batch_processing(self, batch_processor):
        """Test parallel processing of batches."""
        def process_item(x):
            return x * 2
        
        items = list(range(8))
        results = batch_processor.process_parallel(items, process_item)
        
        expected = [x * 2 for x in items]
        assert results == expected
    
    def test_batch_memory_optimization(self, batch_processor):
        """Test memory-efficient batch processing."""
        def memory_intensive_function(data_batch):
            # Simulate memory-intensive operation
            return [sum(data_batch)] * len(data_batch)
        
        large_dataset = [list(range(100)) for _ in range(10)]
        
        # Process with memory optimization
        results = batch_processor.process_with_memory_optimization(
            large_dataset, 
            memory_intensive_function
        )
        
        assert len(results) == 10
        assert all(isinstance(result, list) for result in results)


class TestPerformanceOptimizer:
    """Test performance optimization utilities."""
    
    @pytest.fixture
    def optimizer(self):
        """Create performance optimizer."""
        return PerformanceOptimizer()
    
    def test_vectorization_optimization(self, optimizer):
        """Test automatic vectorization."""
        def scalar_function(x):
            return x ** 2 + 2 * x + 1
        
        vectorized_func = optimizer.vectorize_function(scalar_function)
        
        # Test with numpy array
        input_array = np.array([1, 2, 3, 4, 5])
        result = vectorized_func(input_array)
        
        expected = input_array ** 2 + 2 * input_array + 1
        np.testing.assert_array_equal(result, expected)
    
    def test_memory_usage_profiling(self, optimizer):
        """Test memory usage profiling."""
        def memory_function():
            # Create some memory usage
            large_array = np.random.rand(1000, 1000)
            return np.sum(large_array)
        
        profile = optimizer.profile_memory_usage(memory_function)
        
        assert 'peak_memory' in profile
        assert 'memory_delta' in profile
        assert profile['peak_memory'] > 0
    
    def test_performance_timing(self, optimizer):
        """Test execution time measurement."""
        def timed_function():
            time.sleep(0.01)
            return "completed"
        
        result, timing_info = optimizer.time_function(timed_function)
        
        assert result == "completed"
        assert 'execution_time' in timing_info
        assert timing_info['execution_time'] > 0.01
    
    def test_algorithm_complexity_analysis(self, optimizer):
        """Test algorithm complexity analysis."""
        def test_algorithm(n):
            # O(n^2) algorithm
            total = 0
            for i in range(n):
                for j in range(n):
                    total += i + j
            return total
        
        analysis = optimizer.analyze_complexity(
            test_algorithm, 
            input_sizes=[10, 20, 30, 40, 50]
        )
        
        assert 'complexity_estimate' in analysis
        assert 'scaling_factor' in analysis
        assert 'r_squared' in analysis


class TestGPUAccelerator:
    """Test GPU acceleration utilities."""
    
    @pytest.fixture
    def gpu_accelerator(self):
        """Create GPU accelerator."""
        return GPUAccelerator()
    
    def test_gpu_availability_check(self, gpu_accelerator):
        """Test GPU availability detection."""
        # This will work regardless of actual GPU presence
        available = gpu_accelerator.is_gpu_available()
        assert isinstance(available, bool)
    
    def test_memory_management(self, gpu_accelerator):
        """Test GPU memory management."""
        if not gpu_accelerator.is_gpu_available():
            pytest.skip("GPU not available")
        
        memory_info = gpu_accelerator.get_memory_info()
        
        assert 'total_memory' in memory_info
        assert 'allocated_memory' in memory_info
        assert 'free_memory' in memory_info
    
    def test_tensor_optimization(self, gpu_accelerator):
        """Test tensor operation optimization."""
        # Mock tensor operations
        def tensor_operation(a, b):
            return a @ b + a
        
        # Create mock tensors
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        
        optimized_op = gpu_accelerator.optimize_tensor_operation(tensor_operation)
        result = optimized_op(a, b)
        
        assert result.shape == (100, 100)
    
    def test_kernel_compilation(self, gpu_accelerator):
        """Test CUDA kernel compilation."""
        if not gpu_accelerator.is_gpu_available():
            pytest.skip("GPU not available")
        
        # Mock kernel source code
        kernel_source = """
        __global__ void simple_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
        """
        
        try:
            kernel = gpu_accelerator.compile_kernel("simple_kernel", kernel_source)
            assert kernel is not None
        except Exception:
            pytest.skip("CUDA compilation not available")


# Integration tests
class TestOptimizationIntegration:
    """Test integration of optimization components."""
    
    def test_adaptive_annealing_with_caching(self):
        """Test adaptive annealing with computation caching."""
        # Mock problem
        mock_problem = Mock()
        mock_problem.compute_energy = Mock(return_value=1.5)
        mock_problem.n_spins = 20
        
        # Create adaptive annealer with caching
        config = AdaptiveConfig()
        annealer = AdaptiveSimulatedAnnealing(config)
        cache = ComputationCache(memory_cache_size=50)
        
        # Cached optimization function
        @cache.cached_function()
        def cached_optimization_step(problem, temperature):
            return annealer.optimization_step(problem)
        
        # Run multiple steps
        results = []
        for _ in range(5):
            result = cached_optimization_step(mock_problem, 1.0)
            results.append(result)
        
        assert len(results) == 5
        assert all('energy' in result for result in results)
    
    def test_batch_processing_with_gpu_acceleration(self):
        """Test batch processing with GPU acceleration."""
        batch_processor = BatchProcessor(batch_size=2)
        gpu_accelerator = GPUAccelerator()
        
        def process_batch(batch):
            # Mock GPU-accelerated batch processing
            return [item * 2 for item in batch]
        
        data = list(range(10))
        results = batch_processor.process_parallel(data, process_batch)
        
        expected = [x * 2 for x in data]
        assert results == expected
    
    def test_performance_optimization_pipeline(self):
        """Test complete performance optimization pipeline."""
        optimizer = PerformanceOptimizer()
        cache = ComputationCache(memory_cache_size=100)
        
        # Define computation-heavy function
        def complex_computation(data):
            result = 0
            for item in data:
                result += item ** 2
            return result
        
        # Optimize function
        vectorized_func = optimizer.vectorize_function(complex_computation)
        cached_func = cache.cached_function()(vectorized_func)
        
        # Test with data
        test_data = np.array([1, 2, 3, 4, 5])
        
        # First call
        result1 = cached_func(test_data)
        
        # Second call (should be cached)
        result2 = cached_func(test_data)
        
        assert result1 == result2
        
        # Check cache statistics
        stats = cache.get_statistics()
        assert stats['cache_hits'] >= 1


if __name__ == '__main__':
    pytest.main([__file__])