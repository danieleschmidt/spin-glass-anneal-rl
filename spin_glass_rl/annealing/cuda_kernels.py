"""Custom CUDA kernels for GPU-accelerated annealing operations."""

import torch
from typing import Optional, Tuple
import numpy as np

# Raw CUDA kernel source code for optimized operations
SPIN_UPDATE_KERNEL_SOURCE = """
extern "C" __global__ void metropolis_update_kernel(
    float* spins,
    float* couplings,
    float* external_fields,
    float* random_values,
    float* energy_changes,
    int* accepted_flips,
    float temperature,
    int n_spins,
    int n_updates_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread handles multiple spin updates
    for (int update = 0; update < n_updates_per_thread; update++) {
        int spin_idx = (tid + update * total_threads) % n_spins;
        
        // Calculate local field (sum of coupling * neighbor_spins + external_field)
        float local_field = external_fields[spin_idx];
        for (int j = 0; j < n_spins; j++) {
            if (j != spin_idx) {
                local_field += couplings[spin_idx * n_spins + j] * spins[j];
            }
        }
        
        // Calculate energy change for flipping this spin
        float delta_energy = 2.0f * spins[spin_idx] * local_field;
        
        // Metropolis acceptance criterion
        float acceptance_prob = (delta_energy <= 0.0f) ? 1.0f : expf(-delta_energy / temperature);
        
        // Use random value to decide acceptance
        int rand_idx = (spin_idx + update * n_spins) % (n_spins * n_updates_per_thread);
        if (random_values[rand_idx] < acceptance_prob) {
            spins[spin_idx] *= -1.0f;  // Flip the spin
            energy_changes[spin_idx] += delta_energy;
            atomicAdd(&accepted_flips[0], 1);
        }
    }
}
"""

ENERGY_KERNEL_SOURCE = """
extern "C" __global__ void compute_energy_kernel(
    float* spins,
    float* couplings,
    float* external_fields,
    float* partial_energies,
    int n_spins
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    float local_energy = 0.0f;
    
    // Each thread computes partial energy for assigned spin range
    for (int i = tid; i < n_spins; i += stride) {
        // External field contribution
        local_energy -= external_fields[i] * spins[i];
        
        // Coupling contribution (only upper triangle to avoid double counting)
        for (int j = i + 1; j < n_spins; j++) {
            local_energy -= 0.5f * couplings[i * n_spins + j] * spins[i] * spins[j];
        }
    }
    
    partial_energies[tid] = local_energy;
}
"""

PARALLEL_TEMPERING_KERNEL_SOURCE = """
extern "C" __global__ void parallel_tempering_exchange_kernel(
    float* spins_arrays,
    float* energies,
    float* temperatures,
    float* random_values,
    int* exchange_accepted,
    int n_replicas,
    int n_spins
) {
    int replica_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (replica_idx >= n_replicas - 1) return;
    
    int replica1 = replica_idx;
    int replica2 = replica_idx + 1;
    
    // Calculate exchange probability
    float beta1 = 1.0f / temperatures[replica1];
    float beta2 = 1.0f / temperatures[replica2];
    float energy1 = energies[replica1];
    float energy2 = energies[replica2];
    
    float delta_beta = beta2 - beta1;
    float delta_energy = energy1 - energy2;
    float exchange_prob = expf(delta_beta * delta_energy);
    
    // Accept or reject exchange
    if (random_values[replica_idx] < exchange_prob) {
        // Swap spin configurations
        for (int i = 0; i < n_spins; i++) {
            float temp_spin = spins_arrays[replica1 * n_spins + i];
            spins_arrays[replica1 * n_spins + i] = spins_arrays[replica2 * n_spins + i];
            spins_arrays[replica2 * n_spins + i] = temp_spin;
        }
        
        // Swap energies
        float temp_energy = energies[replica1];
        energies[replica1] = energies[replica2];
        energies[replica2] = temp_energy;
        
        atomicAdd(&exchange_accepted[0], 1);
    }
}
"""


class CUDAKernelManager:
    """Manager for compiling and executing custom CUDA kernels."""
    
    def __init__(self, device: torch.device):
        """Initialize CUDA kernel manager.
        
        Args:
            device: CUDA device to compile kernels for
        """
        self.device = device
        self.compiled_kernels = {}
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile CUDA kernels if CUDA is available."""
        if not torch.cuda.is_available() or self.device.type != 'cuda':
            return
        
        try:
            # Check for NVCC compiler with safe subprocess usage
            import subprocess
            import shutil
            
            # Use shutil.which for safer executable detection
            nvcc_path = shutil.which('nvcc')
            if nvcc_path is None:
                print("Warning: NVCC compiler not found. CUDA kernels will not be compiled.")
                print("Falling back to PyTorch operations.")
                return
            
            try:
                # Safe subprocess call with timeout and explicit arguments
                result = subprocess.run(
                    [nvcc_path, '--version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10,
                    check=False
                )
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, 'nvcc')
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                print("Warning: NVCC compiler not available. CUDA kernels will not be compiled.")
                print("Falling back to PyTorch operations.")
                return
            
            # Try to compile kernels using torch.utils.cpp_extension
            from torch.utils.cpp_extension import load_inline
            
            print("Compiling CUDA kernels... (this may take a moment)")
            
            # Compile spin update kernel with better error handling
            try:
                self.compiled_kernels['metropolis_update'] = load_inline(
                    name='metropolis_update',
                    cpp_sources=[],
                    cuda_sources=[SPIN_UPDATE_KERNEL_SOURCE],
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math'],
                    extra_ldflags=['-lcurand']
                )
                print("✓ Metropolis update kernel compiled successfully")
            except Exception as e:
                print(f"✗ Failed to compile Metropolis kernel: {e}")
                raise
            
            # Compile energy computation kernel
            try:
                self.compiled_kernels['compute_energy'] = load_inline(
                    name='compute_energy',
                    cpp_sources=[],
                    cuda_sources=[ENERGY_KERNEL_SOURCE],
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
                print("✓ Energy computation kernel compiled successfully")
            except Exception as e:
                print(f"✗ Failed to compile energy kernel: {e}")
                raise
            
            # Compile parallel tempering kernel
            try:
                self.compiled_kernels['parallel_tempering'] = load_inline(
                    name='parallel_tempering',
                    cpp_sources=[],
                    cuda_sources=[PARALLEL_TEMPERING_KERNEL_SOURCE],
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
                print("✓ Parallel tempering kernel compiled successfully")
            except Exception as e:
                print(f"✗ Failed to compile parallel tempering kernel: {e}")
                raise
            
            print("✓ All CUDA kernels compiled successfully")
            
        except Exception as e:
            # Fall back to PyTorch operations if kernel compilation fails
            print(f"Warning: CUDA kernel compilation failed: {e}")
            print("Falling back to PyTorch operations")
    
    def metropolis_update_optimized(
        self,
        spins: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor,
        temperature: float,
        n_updates: int = 1
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Optimized Metropolis update using custom CUDA kernel.
        
        Args:
            spins: Current spin configuration [n_spins]
            couplings: Coupling matrix [n_spins, n_spins]
            external_fields: External field values [n_spins]
            temperature: Current temperature
            n_updates: Number of update attempts per spin
            
        Returns:
            Updated spins, number of accepted flips, energy changes
        """
        if 'metropolis_update' not in self.compiled_kernels:
            return self._metropolis_update_fallback(
                spins, couplings, external_fields, temperature, n_updates
            )
        
        n_spins = spins.shape[0]
        
        # Generate random values for acceptance decisions
        random_values = torch.rand(n_spins * n_updates, device=self.device)
        
        # Prepare output tensors
        energy_changes = torch.zeros_like(spins)
        accepted_flips = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        # Launch kernel
        block_size = 256
        grid_size = min(65535, (n_spins + block_size - 1) // block_size)
        
        self.compiled_kernels['metropolis_update'].metropolis_update_kernel(
            grid=(grid_size,),
            block=(block_size,),
            args=[
                spins.data_ptr(),
                couplings.data_ptr(),
                external_fields.data_ptr(),
                random_values.data_ptr(),
                energy_changes.data_ptr(),
                accepted_flips.data_ptr(),
                temperature,
                n_spins,
                n_updates
            ]
        )
        
        return spins, accepted_flips.item(), energy_changes
    
    def compute_energy_optimized(
        self,
        spins: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor
    ) -> float:
        """Optimized energy computation using custom CUDA kernel.
        
        Args:
            spins: Spin configuration [n_spins]
            couplings: Coupling matrix [n_spins, n_spins]
            external_fields: External field values [n_spins]
            
        Returns:
            Total energy of the configuration
        """
        if 'compute_energy' not in self.compiled_kernels:
            return self._compute_energy_fallback(spins, couplings, external_fields)
        
        n_spins = spins.shape[0]
        
        # Prepare partial energy array
        block_size = 256
        grid_size = min(65535, (n_spins + block_size - 1) // block_size)
        partial_energies = torch.zeros(grid_size * block_size, device=self.device)
        
        # Launch kernel
        self.compiled_kernels['compute_energy'].compute_energy_kernel(
            grid=(grid_size,),
            block=(block_size,),
            args=[
                spins.data_ptr(),
                couplings.data_ptr(),
                external_fields.data_ptr(),
                partial_energies.data_ptr(),
                n_spins
            ]
        )
        
        # Sum partial energies
        return partial_energies.sum().item()
    
    def parallel_tempering_exchange_optimized(
        self,
        spins_arrays: torch.Tensor,
        energies: torch.Tensor,
        temperatures: torch.Tensor
    ) -> int:
        """Optimized parallel tempering exchange using custom CUDA kernel.
        
        Args:
            spins_arrays: Spin configurations for all replicas [n_replicas, n_spins]
            energies: Energies for all replicas [n_replicas]
            temperatures: Temperatures for all replicas [n_replicas]
            
        Returns:
            Number of successful exchanges
        """
        if 'parallel_tempering' not in self.compiled_kernels:
            return self._parallel_tempering_fallback(spins_arrays, energies, temperatures)
        
        n_replicas, n_spins = spins_arrays.shape
        
        # Generate random values for exchange decisions
        random_values = torch.rand(n_replicas - 1, device=self.device)
        exchange_accepted = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        # Launch kernel
        block_size = min(256, n_replicas - 1)
        grid_size = (n_replicas - 1 + block_size - 1) // block_size
        
        self.compiled_kernels['parallel_tempering'].parallel_tempering_exchange_kernel(
            grid=(grid_size,),
            block=(block_size,),
            args=[
                spins_arrays.data_ptr(),
                energies.data_ptr(),
                temperatures.data_ptr(),
                random_values.data_ptr(),
                exchange_accepted.data_ptr(),
                n_replicas,
                n_spins
            ]
        )
        
        return exchange_accepted.item()
    
    def _metropolis_update_fallback(
        self,
        spins: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor,
        temperature: float,
        n_updates: int = 1
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Fallback Metropolis update using PyTorch operations."""
        n_spins = spins.shape[0]
        accepted_flips = 0
        energy_changes = torch.zeros_like(spins)
        
        for _ in range(n_updates):
            for i in range(n_spins):
                # Calculate local field
                local_field = external_fields[i] + torch.sum(couplings[i] * spins) - couplings[i, i] * spins[i]
                
                # Calculate energy change
                delta_energy = 2.0 * spins[i] * local_field
                
                # Metropolis acceptance
                if delta_energy <= 0 or torch.rand(1).item() < torch.exp(-delta_energy / temperature):
                    spins[i] *= -1
                    energy_changes[i] += delta_energy
                    accepted_flips += 1
        
        return spins, accepted_flips, energy_changes
    
    def _compute_energy_fallback(
        self,
        spins: torch.Tensor,
        couplings: torch.Tensor,
        external_fields: torch.Tensor
    ) -> float:
        """Fallback energy computation using PyTorch operations."""
        # Coupling energy (avoiding double counting)
        coupling_energy = -0.5 * torch.sum(spins.unsqueeze(0) * couplings * spins.unsqueeze(1))
        
        # External field energy
        field_energy = -torch.sum(external_fields * spins)
        
        return (coupling_energy + field_energy).item()
    
    def _parallel_tempering_fallback(
        self,
        spins_arrays: torch.Tensor,
        energies: torch.Tensor,
        temperatures: torch.Tensor
    ) -> int:
        """Fallback parallel tempering exchange using PyTorch operations."""
        n_replicas = spins_arrays.shape[0]
        exchanges_accepted = 0
        
        for i in range(n_replicas - 1):
            # Calculate exchange probability
            beta1 = 1.0 / temperatures[i]
            beta2 = 1.0 / temperatures[i + 1]
            energy1 = energies[i]
            energy2 = energies[i + 1]
            
            delta_beta = beta2 - beta1
            delta_energy = energy1 - energy2
            exchange_prob = torch.exp(delta_beta * delta_energy)
            
            # Accept or reject exchange
            if torch.rand(1).item() < exchange_prob:
                # Swap configurations
                spins_arrays[[i, i + 1]] = spins_arrays[[i + 1, i]]
                energies[[i, i + 1]] = energies[[i + 1, i]]
                exchanges_accepted += 1
        
        return exchanges_accepted


class GPUMemoryOptimizer:
    """Optimizer for GPU memory usage in large-scale annealing."""
    
    def __init__(self, device: torch.device):
        """Initialize GPU memory optimizer.
        
        Args:
            device: CUDA device to optimize for
        """
        self.device = device
        self.memory_pool = {}
        
    def get_optimal_batch_size(self, n_spins: int, available_memory: Optional[int] = None) -> int:
        """Calculate optimal batch size for given model size and available memory.
        
        Args:
            n_spins: Number of spins in the model
            available_memory: Available GPU memory in bytes (auto-detect if None)
            
        Returns:
            Optimal batch size for processing
        """
        if available_memory is None:
            if torch.cuda.is_available() and self.device.type == 'cuda':
                # Get available GPU memory
                available_memory = torch.cuda.get_device_properties(self.device).total_memory
                allocated_memory = torch.cuda.memory_allocated(self.device)
                available_memory = available_memory - allocated_memory
            else:
                # Assume 4GB for CPU
                available_memory = 4 * 1024**3
        
        # Estimate memory usage per spin configuration
        # - Spins: n_spins * 4 bytes (float32)
        # - Couplings: n_spins^2 * 4 bytes (float32)
        # - Working memory: ~2x for intermediate calculations
        memory_per_config = (n_spins + n_spins**2) * 4 * 2
        
        # Reserve 20% of memory for other operations
        usable_memory = int(available_memory * 0.8)
        
        # Calculate batch size
        batch_size = max(1, usable_memory // memory_per_config)
        
        return min(batch_size, 64)  # Cap at reasonable maximum
    
    def create_memory_efficient_tensors(
        self,
        n_spins: int,
        batch_size: int,
        use_half_precision: bool = False
    ) -> dict:
        """Create memory-efficient tensors for batch processing.
        
        Args:
            n_spins: Number of spins
            batch_size: Batch size for processing
            use_half_precision: Whether to use half precision (float16)
            
        Returns:
            Dictionary of pre-allocated tensors
        """
        dtype = torch.float16 if use_half_precision else torch.float32
        
        tensors = {
            'spins_batch': torch.zeros(batch_size, n_spins, dtype=dtype, device=self.device),
            'energies_batch': torch.zeros(batch_size, dtype=dtype, device=self.device),
            'temp_spins': torch.zeros(n_spins, dtype=dtype, device=self.device),
            'random_values': torch.zeros(batch_size * n_spins, dtype=dtype, device=self.device),
            'local_fields': torch.zeros(batch_size, n_spins, dtype=dtype, device=self.device)
        }
        
        return tensors
    
    def optimize_coupling_matrix_storage(self, couplings: torch.Tensor, sparsity_threshold: float = 0.1) -> torch.Tensor:
        """Optimize coupling matrix storage format based on sparsity.
        
        Args:
            couplings: Dense coupling matrix
            sparsity_threshold: Threshold for converting to sparse format
            
        Returns:
            Optimized coupling matrix (sparse or dense)
        """
        # Calculate sparsity
        non_zero = torch.count_nonzero(couplings)
        total_elements = couplings.numel()
        sparsity = 1.0 - (non_zero.float() / total_elements)
        
        if sparsity > sparsity_threshold:
            # Convert to sparse COO format for memory efficiency
            return couplings.to_sparse_coo()
        else:
            # Keep dense format for faster access
            return couplings
    
    def clear_memory_cache(self):
        """Clear GPU memory cache and reset memory pool."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.memory_pool.clear()
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        stats = {
            'device': str(self.device),
            'memory_allocated': 0,
            'memory_reserved': 0,
            'max_memory_allocated': 0,
            'memory_stats': {}
        }
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            stats.update({
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'memory_reserved': torch.cuda.memory_reserved(self.device),
                'max_memory_allocated': torch.cuda.max_memory_allocated(self.device),
                'memory_stats': torch.cuda.memory_stats(self.device)
            })
        
        return stats