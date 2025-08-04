"""Memory-efficient batch processing for large-scale annealing operations."""

import torch
import numpy as np
from typing import List, Iterator, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque
import psutil
import gc

from ..core.ising_model import IsingModel, IsingModelConfig
from ..utils.exceptions import MemoryError, BatchProcessingError
from .result import AnnealingResult
from .gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from .cuda_kernels import GPUMemoryOptimizer


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = 32
    max_memory_usage: float = 0.8  # Fraction of available memory
    prefetch_batches: int = 2
    use_mixed_precision: bool = False
    enable_gradient_checkpointing: bool = True
    memory_optimization_level: int = 1  # 0=none, 1=basic, 2=aggressive
    streaming_mode: bool = False
    checkpoint_interval: int = 100  # Save progress every N batches
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("Max memory usage must be between 0 and 1")
        if self.memory_optimization_level not in [0, 1, 2]:
            raise ValueError("Memory optimization level must be 0, 1, or 2")


class MemoryTracker:
    """Track memory usage during batch processing."""
    
    def __init__(self, device: torch.device):
        """Initialize memory tracker.
        
        Args:
            device: Device to track memory for
        """
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.peak_memory = 0
        self.memory_timeline = []
        self.start_time = time.time()
        
    def record_memory(self, label: str = ""):
        """Record current memory usage."""
        current_time = time.time() - self.start_time
        
        if self.is_cuda and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            self.peak_memory = max(self.peak_memory, allocated)
        else:
            # Use system memory for CPU
            process = psutil.Process()
            allocated = process.memory_info().rss
            reserved = process.memory_info().vms
            self.peak_memory = max(self.peak_memory, allocated)
        
        self.memory_timeline.append({
            'time': current_time,
            'allocated': allocated,
            'reserved': reserved,
            'label': label
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_timeline:
            return {}
        
        allocations = [entry['allocated'] for entry in self.memory_timeline]
        
        return {
            'peak_memory': self.peak_memory,
            'mean_memory': np.mean(allocations),
            'std_memory': np.std(allocations),
            'timeline': self.memory_timeline,
            'device': str(self.device)
        }


class BatchIterator:
    """Iterator for processing data in batches."""
    
    def __init__(
        self,
        data: Union[List[Any], Iterator[Any]],
        batch_size: int,
        prefetch_size: int = 0,
        shuffle: bool = False
    ):
        """Initialize batch iterator.
        
        Args:
            data: Data to iterate over
            batch_size: Size of each batch
            prefetch_size: Number of batches to prefetch
            shuffle: Whether to shuffle data
        """
        self.data = list(data) if not isinstance(data, list) else data
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle = shuffle
        
        if self.shuffle:
            np.random.shuffle(self.data)
        
        self.current_idx = 0
        self.prefetch_queue = deque(maxlen=prefetch_size) if prefetch_size > 0 else None
        self._prefetch_lock = threading.Lock() if prefetch_size > 0 else None
        
        if prefetch_size > 0:
            self._start_prefetching()
    
    def _start_prefetching(self):
        """Start background prefetching."""
        def prefetch_worker():
            while self.current_idx < len(self.data):
                if len(self.prefetch_queue) < self.prefetch_size:
                    batch = self._get_next_batch()
                    if batch:
                        with self._prefetch_lock:
                            self.prefetch_queue.append(batch)
                time.sleep(0.001)  # Small delay to prevent busy waiting
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _get_next_batch(self) -> Optional[List[Any]]:
        """Get next batch from data."""
        if self.current_idx >= len(self.data):
            return None
        
        end_idx = min(self.current_idx + self.batch_size, len(self.data))
        batch = self.data[self.current_idx:end_idx]
        self.current_idx = end_idx
        
        return batch
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self) -> List[Any]:
        """Get next batch."""
        if self.prefetch_queue:
            # Use prefetched batch if available
            with self._prefetch_lock:
                if self.prefetch_queue:
                    return self.prefetch_queue.popleft()
        
        # Get batch directly
        batch = self._get_next_batch()
        if batch is None:
            raise StopIteration
        
        return batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class BatchProcessor:
    """Efficient batch processor for large-scale annealing operations."""
    
    def __init__(
        self,
        config: BatchConfig,
        annealer_config: GPUAnnealerConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize batch processor.
        
        Args:
            config: Batch processing configuration
            annealer_config: Annealer configuration
            device: Device to use for processing
        """
        self.config = config
        self.annealer_config = annealer_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.memory_tracker = MemoryTracker(self.device)
        self.memory_optimizer = GPUMemoryOptimizer(self.device)
        self.annealer = GPUAnnealer(annealer_config)
        
        # Batch processing state
        self.processed_batches = 0
        self.total_processing_time = 0.0
        self.checkpoint_data = {}
        
        # Memory management
        self._setup_memory_management()
    
    def _setup_memory_management(self):
        """Setup memory management based on configuration."""
        if self.config.memory_optimization_level >= 1:
            # Enable basic memory optimizations
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
        
        if self.config.memory_optimization_level >= 2:
            # Enable aggressive memory optimizations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Enable memory pool fragmentation reduction
                torch.cuda.memory._set_allocator_settings(
                    'expandable_segments:True'
                )
    
    def process_models_batch(
        self,
        models: List[IsingModel],
        callback: Optional[Callable[[List[AnnealingResult]], None]] = None
    ) -> List[AnnealingResult]:
        """Process a batch of Ising models.
        
        Args:
            models: List of Ising models to process
            callback: Optional callback for processing results
            
        Returns:
            List of annealing results
        """
        self.memory_tracker.record_memory("batch_start")
        batch_start_time = time.time()
        
        try:
            # Move models to device and optimize memory usage
            gpu_models = []
            for model in models:
                gpu_model = self._prepare_model_for_gpu(model)
                gpu_models.append(gpu_model)
            
            self.memory_tracker.record_memory("models_loaded")
            
            # Process models in parallel if possible
            results = []
            if len(gpu_models) <= self.config.batch_size:
                # Small batch - process all at once
                results = self._process_batch_parallel(gpu_models)
            else:
                # Large batch - split into sub-batches
                results = self._process_batch_sequential(gpu_models)
            
            self.memory_tracker.record_memory("batch_processed")
            
            # Apply callback if provided
            if callback:
                callback(results)
            
            # Update statistics
            batch_time = time.time() - batch_start_time
            self.total_processing_time += batch_time
            self.processed_batches += 1
            
            # Checkpoint if needed
            if self.processed_batches % self.config.checkpoint_interval == 0:
                self._save_checkpoint(results)
            
            # Memory cleanup
            self._cleanup_batch_memory(gpu_models)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Failed to process batch: {e}")
    
    def process_models_stream(
        self,
        model_generator: Iterator[IsingModel],
        output_callback: Callable[[AnnealingResult], None]
    ) -> Dict[str, Any]:
        """Process models in streaming mode with minimal memory usage.
        
        Args:
            model_generator: Generator yielding Ising models
            output_callback: Callback for each processed result
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        try:
            for model in model_generator:
                try:
                    # Process single model
                    gpu_model = self._prepare_model_for_gpu(model)
                    result = self.annealer.anneal(gpu_model)
                    
                    # Callback with result
                    output_callback(result)
                    
                    # Cleanup immediately
                    del gpu_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count} models in streaming mode")
                
                except Exception as e:
                    self.logger.error(f"Failed to process model in stream: {e}")
                    error_count += 1
                    continue
        
        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            raise BatchProcessingError(f"Streaming failed: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'total_time': total_time,
            'throughput': processed_count / total_time if total_time > 0 else 0,
            'memory_stats': self.memory_tracker.get_stats()
        }
    
    def process_large_dataset(
        self,
        dataset: List[IsingModel],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[AnnealingResult]:
        """Process large dataset with automatic batching and memory management.
        
        Args:
            dataset: Large dataset of Ising models
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            List of all annealing results
        """
        total_models = len(dataset)
        all_results = []
        
        # Calculate optimal batch size based on memory
        optimal_batch_size = self._calculate_optimal_batch_size(dataset[0] if dataset else None)
        
        self.logger.info(f"Processing {total_models} models with batch size {optimal_batch_size}")
        
        # Create batch iterator
        batch_iterator = BatchIterator(
            dataset,
            batch_size=optimal_batch_size,
            prefetch_size=self.config.prefetch_batches,
            shuffle=False
        )
        
        try:
            for batch_idx, batch_models in enumerate(batch_iterator):
                self.logger.debug(f"Processing batch {batch_idx + 1}/{len(batch_iterator)}")
                
                # Process batch
                batch_results = self.process_models_batch(batch_models)
                all_results.extend(batch_results)
                
                # Progress callback
                if progress_callback:
                    processed_count = len(all_results)
                    progress_callback(processed_count, total_models)
                
                # Memory management between batches
                if batch_idx % 10 == 0:  # Every 10 batches
                    self._aggressive_memory_cleanup()
        
        except Exception as e:
            self.logger.error(f"Large dataset processing failed: {e}")
            raise BatchProcessingError(f"Dataset processing failed: {e}")
        
        return all_results
    
    def _prepare_model_for_gpu(self, model: IsingModel) -> IsingModel:
        """Prepare model for GPU processing with memory optimizations."""
        gpu_model = model.copy()
        
        # Move to device
        gpu_model.spins = gpu_model.spins.to(self.device)
        gpu_model.external_fields = gpu_model.external_fields.to(self.device)
        
        if hasattr(gpu_model, 'couplings'):
            # Optimize coupling matrix storage
            gpu_model.couplings = self.memory_optimizer.optimize_coupling_matrix_storage(
                gpu_model.couplings.to(self.device)
            )
        
        gpu_model.device = self.device
        
        # Use mixed precision if enabled
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            gpu_model.spins = gpu_model.spins.half()
            gpu_model.external_fields = gpu_model.external_fields.half()
        
        return gpu_model
    
    def _process_batch_parallel(self, models: List[IsingModel]) -> List[AnnealingResult]:
        """Process batch in parallel using multiple threads."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
            # Submit all annealing tasks
            futures = []
            for model in models:
                future = executor.submit(self.annealer.anneal, model)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel processing task failed: {e}")
                    # Create dummy result to maintain batch size
                    dummy_result = AnnealingResult(
                        best_configuration=torch.zeros(models[0].n_spins),
                        best_energy=float('inf'),
                        energy_history=[],
                        temperature_history=[],
                        acceptance_rate_history=[],
                        total_time=0.0,
                        n_sweeps=0,
                        metadata={'error': str(e)}
                    )
                    results.append(dummy_result)
        
        return results
    
    def _process_batch_sequential(self, models: List[IsingModel]) -> List[AnnealingResult]:
        """Process batch sequentially for memory efficiency."""
        results = []
        
        for i, model in enumerate(models):
            try:
                result = self.annealer.anneal(model)
                results.append(result)
                
                # Memory cleanup after each model if needed
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Sequential processing failed for model {i}: {e}")
                # Create dummy result
                dummy_result = AnnealingResult(
                    best_configuration=torch.zeros(model.n_spins),
                    best_energy=float('inf'),
                    energy_history=[],
                    temperature_history=[],
                    acceptance_rate_history=[],
                    total_time=0.0,
                    n_sweeps=0,
                    metadata={'error': str(e)}
                )
                results.append(dummy_result)
        
        return results
    
    def _calculate_optimal_batch_size(self, sample_model: Optional[IsingModel]) -> int:
        """Calculate optimal batch size based on available memory."""
        if sample_model is None:
            return self.config.batch_size
        
        # Get optimal batch size from memory optimizer
        optimal_size = self.memory_optimizer.get_optimal_batch_size(
            sample_model.n_spins
        )
        
        # Use configured batch size as upper bound
        return min(optimal_size, self.config.batch_size)
    
    def _cleanup_batch_memory(self, models: List[IsingModel]):
        """Clean up memory after processing batch."""
        # Delete model references
        for model in models:
            del model
        
        # Force garbage collection
        if self.config.memory_optimization_level >= 1:
            gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup."""
        gc.collect()
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _save_checkpoint(self, recent_results: List[AnnealingResult]):
        """Save processing checkpoint."""
        self.checkpoint_data = {
            'processed_batches': self.processed_batches,
            'total_processing_time': self.total_processing_time,
            'memory_stats': self.memory_tracker.get_stats(),
            'recent_results_count': len(recent_results),
            'timestamp': time.time()
        }
        
        self.logger.info(f"Checkpoint saved: {self.processed_batches} batches processed")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            'processed_batches': self.processed_batches,
            'total_processing_time': self.total_processing_time,
            'average_batch_time': (
                self.total_processing_time / self.processed_batches
                if self.processed_batches > 0 else 0
            ),
            'memory_stats': self.memory_tracker.get_stats(),
            'memory_optimizer_stats': self.memory_optimizer.get_memory_stats(),
            'config': self.config.__dict__,
            'checkpoint_data': self.checkpoint_data
        }
    
    def reset(self):
        """Reset processing state for new batch job."""
        self.processed_batches = 0
        self.total_processing_time = 0.0
        self.checkpoint_data = {}
        self.memory_tracker = MemoryTracker(self.device)
        self._aggressive_memory_cleanup()


class AdaptiveBatchProcessor(BatchProcessor):
    """Adaptive batch processor that adjusts parameters based on performance."""
    
    def __init__(
        self,
        config: BatchConfig,
        annealer_config: GPUAnnealerConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize adaptive batch processor."""
        super().__init__(config, annealer_config, device)
        
        self.performance_history = deque(maxlen=100)
        self.batch_size_history = deque(maxlen=50)
        self.adaptive_enabled = True
        self.adaptation_interval = 10  # Adapt every N batches
    
    def process_models_batch(
        self,
        models: List[IsingModel],
        callback: Optional[Callable[[List[AnnealingResult]], None]] = None
    ) -> List[AnnealingResult]:
        """Process batch with adaptive optimization."""
        batch_start_time = time.time()
        
        # Process batch normally
        results = super().process_models_batch(models, callback)
        
        # Record performance
        batch_time = time.time() - batch_start_time
        throughput = len(models) / batch_time if batch_time > 0 else 0
        
        self.performance_history.append({
            'batch_size': len(models),
            'batch_time': batch_time,
            'throughput': throughput,
            'memory_peak': self.memory_tracker.peak_memory
        })
        
        # Adapt parameters if needed
        if (self.adaptive_enabled and 
            self.processed_batches % self.adaptation_interval == 0 and
            len(self.performance_history) >= 5):
            self._adapt_parameters()
        
        return results
    
    def _adapt_parameters(self):
        """Adapt processing parameters based on performance history."""
        recent_performance = list(self.performance_history)[-5:]
        
        # Calculate performance metrics
        avg_throughput = np.mean([p['throughput'] for p in recent_performance])
        memory_pressure = np.mean([p['memory_peak'] for p in recent_performance])
        
        # Get available memory
        if torch.cuda.is_available() and self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            memory_utilization = memory_pressure / total_memory
        else:
            memory_utilization = 0.5  # Assume moderate utilization for CPU
        
        # Adapt batch size
        current_batch_size = self.config.batch_size
        
        if memory_utilization > 0.9:
            # High memory pressure - reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            self.logger.info(f"Reducing batch size due to memory pressure: {current_batch_size} -> {new_batch_size}")
        elif memory_utilization < 0.6 and avg_throughput > 0:
            # Low memory pressure and good throughput - try increasing batch size
            new_batch_size = min(128, int(current_batch_size * 1.2))
            self.logger.info(f"Increasing batch size: {current_batch_size} -> {new_batch_size}")
        else:
            new_batch_size = current_batch_size
        
        self.config.batch_size = new_batch_size
        self.batch_size_history.append(new_batch_size)
        
        # Adapt memory optimization level
        if memory_utilization > 0.85:
            self.config.memory_optimization_level = 2
        elif memory_utilization > 0.7:
            self.config.memory_optimization_level = 1
        else:
            self.config.memory_optimization_level = 0
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            'performance_history': list(self.performance_history),
            'batch_size_history': list(self.batch_size_history),
            'current_batch_size': self.config.batch_size,
            'adaptive_enabled': self.adaptive_enabled,
            'adaptation_interval': self.adaptation_interval
        }