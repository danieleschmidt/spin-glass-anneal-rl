"""High-performance computing features for large-scale optimization."""

import torch
import torch.multiprocessing as mp
import numpy as np
import concurrent.futures
import queue
import threading
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import psutil


@dataclass
class ComputeConfig:
    """Configuration for high-performance computing."""
    enable_multiprocessing: bool = True
    max_workers: int = None  # Auto-detect
    enable_gpu_acceleration: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0
    enable_distributed: bool = False
    chunk_size: int = 100
    enable_vectorization: bool = True


class WorkloadDistributor:
    """Distribute optimization workloads across multiple processors."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.n_workers = config.max_workers or min(mp.cpu_count(), 8)
        self.executor = None
        self.gpu_available = torch.cuda.is_available() and config.enable_gpu_acceleration
        
    def __enter__(self):
        if self.config.enable_multiprocessing:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_workers
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def distribute_work(
        self,
        work_func: Callable,
        work_items: List[Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Distribute work across workers."""
        if not self.config.enable_multiprocessing or len(work_items) < 2:
            # Single-threaded execution
            results = []
            for i, item in enumerate(work_items):
                result = work_func(item)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(work_items))
            return results
        
        # Multi-threaded execution
        futures = []
        results = [None] * len(work_items)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all work
            for i, item in enumerate(work_items):
                future = executor.submit(work_func, item)
                futures.append((i, future))
            
            # Collect results
            completed = 0
            for i, future in futures:
                try:
                    results[i] = future.result(timeout=300)  # 5 minute timeout
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(work_items))
                except Exception as e:
                    print(f"Worker {i} failed: {e}")
                    results[i] = None
        
        return results


class BatchProcessor:
    """Process optimization tasks in optimized batches."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.batch_size = config.batch_size
        
    def process_batch_energies(
        self,
        spin_configurations: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor
    ) -> torch.Tensor:
        """Compute energies for batch of spin configurations."""
        if spin_configurations.dim() == 1:
            # Single configuration
            return self._single_energy(spin_configurations, couplings, external_fields)
        
        # Batch computation
        batch_size, n_spins = spin_configurations.shape
        
        # Vectorized energy computation
        # E = -0.5 * Σ_ij J_ij * s_i * s_j - Σ_i h_i * s_i
        
        # External field energy: batch_size x n_spins @ n_spins -> batch_size
        field_energy = -torch.sum(spin_configurations * external_fields, dim=1)
        
        # Coupling energy: more complex for batch
        if couplings.is_sparse:
            # Sparse coupling computation
            coupling_energy = self._batch_sparse_coupling_energy(
                spin_configurations, couplings
            )
        else:
            # Dense coupling computation
            coupling_energy = self._batch_dense_coupling_energy(
                spin_configurations, couplings
            )
        
        return field_energy + coupling_energy
    
    def _single_energy(
        self,
        spins: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy for single configuration."""
        field_energy = -torch.sum(external_fields * spins)
        
        if couplings.is_sparse:
            coupling_energy = -0.5 * torch.sparse.mm(
                couplings.unsqueeze(0), spins.unsqueeze(1)
            ).squeeze()
        else:
            coupling_energy = -0.5 * torch.sum(
                spins.unsqueeze(0) * couplings * spins.unsqueeze(1)
            )
        
        return field_energy + coupling_energy
    
    def _batch_dense_coupling_energy(
        self,
        spin_configurations: torch.Tensor,
        couplings: torch.Tensor
    ) -> torch.Tensor:
        """Compute coupling energy for batch with dense couplings."""
        batch_size = spin_configurations.shape[0]
        
        # Efficient batch computation: spins @ couplings @ spins.T
        # Shape: (batch_size, n_spins) @ (n_spins, n_spins) @ (n_spins, batch_size)
        temp = torch.mm(spin_configurations, couplings)  # batch_size x n_spins
        coupling_energy = -0.5 * torch.sum(temp * spin_configurations, dim=1)
        
        return coupling_energy
    
    def _batch_sparse_coupling_energy(
        self,
        spin_configurations: torch.Tensor,
        couplings: torch.Tensor
    ) -> torch.Tensor:
        """Compute coupling energy for batch with sparse couplings."""
        batch_size = spin_configurations.shape[0]
        
        # Convert to dense for batch operations (if not too large)
        if couplings.nnz() / couplings.numel() > 0.1:  # > 10% dense
            dense_couplings = couplings.to_dense()
            return self._batch_dense_coupling_energy(spin_configurations, dense_couplings)
        
        # Keep sparse - process each configuration separately
        energies = []
        for i in range(batch_size):
            spins = spin_configurations[i]
            temp = torch.sparse.mm(couplings, spins.unsqueeze(1)).squeeze()
            energy = -0.5 * torch.sum(spins * temp)
            energies.append(energy)
        
        return torch.stack(energies)


class GPUAccelerator:
    """GPU acceleration for optimization computations."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.device = self._get_best_device()
        self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        
    def _get_best_device(self) -> torch.device:
        """Get the best available compute device."""
        if not self.config.enable_gpu_acceleration or not torch.cuda.is_available():
            return torch.device('cpu')
        
        # Select GPU with most memory
        best_gpu = 0
        max_memory = 0
        
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory
            if memory > max_memory:
                max_memory = memory
                best_gpu = i
        
        return torch.device(f'cuda:{best_gpu}')
    
    def optimize_tensor_operations(self):
        """Optimize tensor operations for current device."""
        if self.device.type == 'cuda':
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set memory format optimization
            torch.backends.cudnn.deterministic = False  # For performance
            
    def batch_energy_computation(
        self,
        spin_batches: List[torch.Tensor],
        model_params: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Compute energies for multiple batches on GPU."""
        results = []
        
        # Move model parameters to device once
        device_params = {
            key: tensor.to(self.device) for key, tensor in model_params.items()
        }
        
        for batch in spin_batches:
            # Move batch to device
            device_batch = batch.to(self.device)
            
            # Compute energy on GPU
            with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
                energy = self._compute_batch_energy(device_batch, device_params)
                
            # Move result back to CPU
            results.append(energy.cpu())
        
        return results
    
    def _compute_batch_energy(
        self,
        spin_batch: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute energy for a batch on current device."""
        couplings = params['couplings']
        external_fields = params['external_fields']
        
        # Vectorized computation
        field_energy = -torch.sum(spin_batch * external_fields, dim=1)
        
        # Coupling energy using optimized matrix operations
        temp = torch.mm(spin_batch, couplings)
        coupling_energy = -0.5 * torch.sum(temp * spin_batch, dim=1)
        
        return field_energy + coupling_energy


class MemoryManager:
    """Manage memory usage for large-scale optimizations."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.memory_limit = config.memory_limit_gb * 1024**3  # Convert to bytes
        self.active_tensors: Dict[str, torch.Tensor] = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            "system_memory_gb": psutil.virtual_memory().used / 1024**3,
            "available_memory_gb": psutil.virtual_memory().available / 1024**3,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "gpu_max_memory_mb": torch.cuda.max_memory_allocated() / 1024**2
            })
        
        return stats
    
    def optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout."""
        if tensor.is_sparse:
            # Coalesce sparse tensors
            return tensor.coalesce()
        
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Use appropriate data type
        if tensor.dtype == torch.float64 and not tensor.requires_grad:
            # Use float32 for better performance
            tensor = tensor.float()
        
        return tensor
    
    def manage_tensor_lifecycle(self, tensor_id: str, tensor: torch.Tensor) -> None:
        """Manage tensor lifecycle to prevent memory leaks."""
        self.active_tensors[tensor_id] = tensor
        
        # Check if we're approaching memory limit
        current_usage = self.get_memory_usage()
        if current_usage["memory_percent"] > 85:  # > 85% memory usage
            self._cleanup_unused_tensors()
    
    def _cleanup_unused_tensors(self) -> None:
        """Clean up unused tensors to free memory."""
        # Remove tensors that are no longer referenced
        to_remove = []
        for tensor_id, tensor in self.active_tensors.items():
            if tensor.ref_count() <= 1:  # Only this reference
                to_remove.append(tensor_id)
        
        for tensor_id in to_remove:
            del self.active_tensors[tensor_id]
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VectorizedOperations:
    """Highly optimized vectorized operations."""
    
    @staticmethod
    def vectorized_spin_flips(
        spin_configurations: torch.Tensor,
        flip_indices: torch.Tensor
    ) -> torch.Tensor:
        """Perform vectorized spin flips."""
        # Create a copy to avoid in-place modification
        result = spin_configurations.clone()
        
        # Vectorized flip operation
        batch_indices = torch.arange(result.shape[0]).unsqueeze(1)
        result[batch_indices, flip_indices] *= -1
        
        return result
    
    @staticmethod
    def vectorized_local_fields(
        spin_configurations: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor
    ) -> torch.Tensor:
        """Compute local fields for all spins in batch."""
        # Local field h_i = Σ_j J_ij * s_j + h_i
        
        if couplings.is_sparse:
            # Sparse matrix multiplication
            coupling_fields = torch.sparse.mm(couplings, spin_configurations.t()).t()
        else:
            # Dense matrix multiplication
            coupling_fields = torch.mm(spin_configurations, couplings)
        
        return coupling_fields + external_fields
    
    @staticmethod
    def vectorized_energy_differences(
        spin_configurations: torch.Tensor,
        local_fields: torch.Tensor,
        flip_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy differences for vectorized spin flips."""
        batch_indices = torch.arange(spin_configurations.shape[0]).unsqueeze(1)
        current_spins = spin_configurations[batch_indices, flip_indices]
        fields = local_fields[batch_indices, flip_indices]
        
        # ΔE = 2 * s_i * h_i (for flipping spin i)
        return 2.0 * current_spins * fields


class DistributedComputing:
    """Distributed computing capabilities for multi-node optimization."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.enabled = config.enable_distributed
        self.rank = 0
        self.world_size = 1
        
    def initialize_distributed(self, rank: int, world_size: int) -> None:
        """Initialize distributed computing environment."""
        if not self.enabled:
            return
        
        self.rank = rank
        self.world_size = world_size
        
        # Initialize process group (placeholder - would use actual distributed framework)
        print(f"Initialized distributed node {rank}/{world_size}")
    
    def distribute_work_across_nodes(
        self,
        work_items: List[Any],
        work_func: Callable
    ) -> List[Any]:
        """Distribute work across multiple nodes."""
        if not self.enabled or self.world_size == 1:
            return [work_func(item) for item in work_items]
        
        # Split work across nodes
        items_per_node = len(work_items) // self.world_size
        start_idx = self.rank * items_per_node
        
        if self.rank == self.world_size - 1:
            # Last node gets any remaining items
            end_idx = len(work_items)
        else:
            end_idx = start_idx + items_per_node
        
        # Process assigned work
        local_items = work_items[start_idx:end_idx]
        local_results = [work_func(item) for item in local_items]
        
        # In real implementation, would gather results from all nodes
        return local_results


class PerformanceOptimizer:
    """Optimize performance based on runtime characteristics."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.batch_processor = BatchProcessor(config)
        self.gpu_accelerator = GPUAccelerator(config) if config.enable_gpu_acceleration else None
        self.memory_manager = MemoryManager(config)
        self.workload_distributor = None
        
    def optimize_for_problem_size(self, n_spins: int, n_samples: int) -> ComputeConfig:
        """Optimize configuration based on problem size."""
        optimized_config = ComputeConfig()
        
        # Adjust batch size based on problem size
        if n_spins < 100:
            optimized_config.batch_size = min(1000, n_samples)
        elif n_spins < 1000:
            optimized_config.batch_size = min(500, n_samples)
        else:
            optimized_config.batch_size = min(100, n_samples)
        
        # Enable GPU for larger problems
        optimized_config.enable_gpu_acceleration = n_spins > 50 and torch.cuda.is_available()
        
        # Enable multiprocessing for many samples
        optimized_config.enable_multiprocessing = n_samples > 10
        
        # Adjust memory limits
        available_memory = psutil.virtual_memory().available / 1024**3
        optimized_config.memory_limit_gb = min(available_memory * 0.8, 16.0)
        
        return optimized_config
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        memory_stats = self.memory_manager.get_memory_usage()
        
        if memory_stats["memory_percent"] > 80:
            recommendations.append("Consider reducing batch size to lower memory usage")
        
        if torch.cuda.is_available() and not self.config.enable_gpu_acceleration:
            recommendations.append("Enable GPU acceleration for better performance")
        
        if psutil.cpu_count() > 4 and not self.config.enable_multiprocessing:
            recommendations.append("Enable multiprocessing to utilize multiple CPU cores")
        
        if memory_stats.get("gpu_memory_mb", 0) < 1000 and self.config.enable_gpu_acceleration:
            recommendations.append("GPU memory usage is low - consider increasing batch size")
        
        return recommendations


# Global instances for high-performance computing
global_workload_distributor = None
global_performance_optimizer = None

def initialize_hpc(config: ComputeConfig = None) -> Tuple[WorkloadDistributor, PerformanceOptimizer]:
    """Initialize high-performance computing components."""
    global global_workload_distributor, global_performance_optimizer
    
    if config is None:
        config = ComputeConfig()
    
    global_workload_distributor = WorkloadDistributor(config)
    global_performance_optimizer = PerformanceOptimizer(config)
    
    return global_workload_distributor, global_performance_optimizer