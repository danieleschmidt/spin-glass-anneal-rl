"""Multi-GPU scaling support for large-scale annealing."""

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from typing import List, Dict, Optional, Tuple, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import queue
import threading

from ..core.ising_model import IsingModel
from ..utils.exceptions import DeviceError, AnnealingError
from .result import AnnealingResult
from .gpu_annealer import GPUAnnealer, GPUAnnealerConfig


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU annealing."""
    
    gpu_ids: List[int]  # List of GPU IDs to use
    strategy: str = "data_parallel"  # "data_parallel", "model_parallel", "replica_exchange"
    communication_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    synchronization_interval: int = 10  # Sync every N sweeps
    load_balancing: bool = True
    fault_tolerance: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.gpu_ids:
            raise ValueError("At least one GPU ID must be specified")
        
        valid_strategies = ["data_parallel", "model_parallel", "replica_exchange"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        valid_backends = ["nccl", "gloo", "mpi"]
        if self.communication_backend not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}")


class MultiGPUAnnealer:
    """Multi-GPU annealer with support for various parallelization strategies."""
    
    def __init__(self, config: MultiGPUConfig, annealer_config: GPUAnnealerConfig):
        """Initialize multi-GPU annealer.
        
        Args:
            config: Multi-GPU configuration
            annealer_config: Individual annealer configuration
        """
        self.config = config
        self.annealer_config = annealer_config
        self.devices = self._validate_devices()
        self.master_device = self.devices[0]
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy-specific components
        self._setup_strategy()
        
    def _validate_devices(self) -> List[torch.device]:
        """Validate and create device objects."""
        if not torch.cuda.is_available():
            raise DeviceError("CUDA is not available for multi-GPU annealing")
        
        available_gpus = torch.cuda.device_count()
        devices = []
        
        for gpu_id in self.config.gpu_ids:
            if gpu_id >= available_gpus:
                raise DeviceError(f"GPU {gpu_id} not available (only {available_gpus} GPUs found)")
            devices.append(torch.device(f"cuda:{gpu_id}"))
        
        return devices
    
    def _setup_strategy(self):
        """Setup strategy-specific components."""
        if self.config.strategy == "data_parallel":
            self._setup_data_parallel()
        elif self.config.strategy == "model_parallel":
            self._setup_model_parallel()
        elif self.config.strategy == "replica_exchange":
            self._setup_replica_exchange()
    
    def _setup_data_parallel(self):
        """Setup data parallel strategy components."""
        self.annealers = []
        for device in self.devices:
            annealer_config = GPUAnnealerConfig(**self.annealer_config.__dict__)
            annealer = GPUAnnealer(annealer_config)
            annealer.device = device
            annealer.use_cuda = True
            self.annealers.append(annealer)
    
    def _setup_model_parallel(self):
        """Setup model parallel strategy components."""
        # Model parallelism: split coupling matrix across GPUs
        self.annealers = []
        for device in self.devices:
            annealer_config = GPUAnnealerConfig(**self.annealer_config.__dict__)
            annealer = GPUAnnealer(annealer_config)
            annealer.device = device
            annealer.use_cuda = True
            self.annealers.append(annealer)
    
    def _setup_replica_exchange(self):
        """Setup replica exchange strategy components."""
        self.annealers = []
        n_replicas = len(self.devices)
        
        # Create temperature schedule for replica exchange
        temp_min = self.annealer_config.final_temp
        temp_max = self.annealer_config.initial_temp
        temperatures = torch.logspace(
            torch.log10(torch.tensor(temp_min)),
            torch.log10(torch.tensor(temp_max)),
            n_replicas
        )
        
        for i, device in enumerate(self.devices):
            annealer_config = GPUAnnealerConfig(**self.annealer_config.__dict__)
            annealer_config.initial_temp = temperatures[i].item()
            annealer_config.final_temp = temperatures[i].item()  # Fixed temperature
            
            annealer = GPUAnnealer(annealer_config)
            annealer.device = device
            annealer.use_cuda = True
            self.annealers.append(annealer)
    
    def anneal_data_parallel(self, models: List[IsingModel]) -> List[AnnealingResult]:
        """Run data parallel annealing across multiple GPUs.
        
        Args:
            models: List of Ising models to anneal
            
        Returns:
            List of annealing results
        """
        if len(models) != len(self.devices):
            raise ValueError(f"Number of models ({len(models)}) must match number of GPUs ({len(self.devices)})")
        
        # Distribute models across GPUs
        futures = []
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            for i, (model, annealer) in enumerate(zip(models, self.annealers)):
                # Move model to appropriate device
                model_gpu = self._move_model_to_device(model, self.devices[i])
                
                # Submit annealing task
                future = executor.submit(annealer.anneal, model_gpu)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Annealing failed on GPU: {e}")
                    raise AnnealingError(f"Multi-GPU annealing failed: {e}")
        
        return results
    
    def anneal_model_parallel(self, model: IsingModel) -> AnnealingResult:
        """Run model parallel annealing using multiple GPUs.
        
        Args:
            model: Large Ising model to distribute across GPUs
            
        Returns:
            Combined annealing result
        """
        n_spins = model.n_spins
        n_gpus = len(self.devices)
        spins_per_gpu = n_spins // n_gpus
        
        # Split model across GPUs
        model_parts = []
        for i in range(n_gpus):
            start_idx = i * spins_per_gpu
            end_idx = start_idx + spins_per_gpu if i < n_gpus - 1 else n_spins
            
            # Create sub-model for this GPU
            sub_model = self._create_sub_model(model, start_idx, end_idx)
            sub_model = self._move_model_to_device(sub_model, self.devices[i])
            model_parts.append(sub_model)
        
        # Run coordinated annealing with synchronization
        best_energy = float('inf')
        best_configuration = None
        energy_history = []
        
        for sweep in range(self.annealer_config.n_sweeps):
            # Perform local updates on each GPU
            futures = []
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                for sub_model, annealer in zip(model_parts, self.annealers):
                    future = executor.submit(self._single_sweep_update, annealer, sub_model)
                    futures.append(future)
                
                # Wait for all updates to complete
                for future in futures:
                    future.result()
            
            # Synchronize boundary spins if needed
            if sweep % self.config.synchronization_interval == 0:
                self._synchronize_boundaries(model_parts)
            
            # Calculate total energy and update best configuration
            total_energy = sum(part.compute_energy() for part in model_parts)
            energy_history.append(total_energy)
            
            if total_energy < best_energy:
                best_energy = total_energy
                best_configuration = self._combine_configurations(model_parts)
        
        # Create combined result
        return AnnealingResult(
            best_configuration=best_configuration,
            best_energy=best_energy,
            energy_history=energy_history,
            temperature_history=[],
            acceptance_rate_history=[],
            total_time=0.0,  # Would need proper timing
            n_sweeps=self.annealer_config.n_sweeps,
            metadata={"strategy": "model_parallel", "n_gpus": n_gpus}
        )
    
    def anneal_replica_exchange(self, model: IsingModel, n_replicas: Optional[int] = None) -> AnnealingResult:
        """Run parallel tempering/replica exchange across multiple GPUs.
        
        Args:
            model: Ising model to anneal
            n_replicas: Number of replicas (defaults to number of GPUs)
            
        Returns:
            Best result from all replicas
        """
        if n_replicas is None:
            n_replicas = len(self.devices)
        
        # Create replica models
        replicas = []
        for i in range(n_replicas):
            replica = model.copy()
            device_idx = i % len(self.devices)
            replica = self._move_model_to_device(replica, self.devices[device_idx])
            replicas.append(replica)
        
        # Track best result across all replicas
        best_energy = float('inf')
        best_configuration = None
        energy_histories = [[] for _ in range(n_replicas)]
        exchange_history = []
        
        for sweep in range(self.annealer_config.n_sweeps):
            # Perform local updates on all replicas
            futures = []
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                for i, (replica, annealer) in enumerate(zip(replicas, self.annealers)):
                    future = executor.submit(self._single_sweep_update, annealer, replica)
                    futures.append(future)
                
                # Wait for all updates
                for future in futures:
                    future.result()
            
            # Record energies
            current_energies = []
            for i, replica in enumerate(replicas):
                energy = replica.compute_energy()
                energy_histories[i].append(energy)
                current_energies.append(energy)
                
                # Update global best
                if energy < best_energy:
                    best_energy = energy
                    best_configuration = replica.spins.clone()
            
            # Attempt replica exchanges
            if sweep % self.config.synchronization_interval == 0:
                exchanges = self._attempt_replica_exchanges(replicas, current_energies)
                exchange_history.append(exchanges)
        
        # Find the best replica
        final_energies = [replica.compute_energy() for replica in replicas]
        best_replica_idx = min(range(len(final_energies)), key=lambda i: final_energies[i])
        
        return AnnealingResult(
            best_configuration=best_configuration,
            best_energy=best_energy,
            energy_history=energy_histories[best_replica_idx],
            temperature_history=[],
            acceptance_rate_history=[],
            total_time=0.0,  # Would need proper timing
            n_sweeps=self.annealer_config.n_sweeps,
            metadata={
                "strategy": "replica_exchange",
                "n_replicas": n_replicas,
                "exchanges": exchange_history
            }
        )
    
    def anneal(self, models: Any) -> Any:
        """Main annealing interface that dispatches to appropriate strategy.
        
        Args:
            models: Model(s) to anneal (format depends on strategy)
            
        Returns:
            Result(s) from annealing
        """
        start_time = time.time()
        
        try:
            if self.config.strategy == "data_parallel":
                if not isinstance(models, list):
                    raise ValueError("Data parallel strategy requires list of models")
                results = self.anneal_data_parallel(models)
            
            elif self.config.strategy == "model_parallel":
                if isinstance(models, list):
                    raise ValueError("Model parallel strategy requires single model")
                results = self.anneal_model_parallel(models)
            
            elif self.config.strategy == "replica_exchange":
                if isinstance(models, list):
                    raise ValueError("Replica exchange strategy requires single model")
                results = self.anneal_replica_exchange(models)
            
            else:
                raise ValueError(f"Unknown strategy: {self.config.strategy}")
            
            # Update timing
            total_time = time.time() - start_time
            if isinstance(results, list):
                for result in results:
                    result.total_time = total_time / len(results)
            else:
                results.total_time = total_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-GPU annealing failed: {e}")
            raise AnnealingError(f"Multi-GPU annealing failed: {e}")
    
    def _move_model_to_device(self, model: IsingModel, device: torch.device) -> IsingModel:
        """Move model to specified device."""
        model.spins = model.spins.to(device)
        model.external_fields = model.external_fields.to(device)
        
        if hasattr(model, 'couplings'):
            model.couplings = model.couplings.to(device)
        
        model.device = device
        model._cached_energy = None  # Invalidate cache
        
        return model
    
    def _create_sub_model(self, model: IsingModel, start_idx: int, end_idx: int) -> IsingModel:
        """Create sub-model for model parallel processing."""
        n_sub_spins = end_idx - start_idx
        
        # Create new config for sub-model
        from ..core.ising_model import IsingModelConfig
        sub_config = IsingModelConfig(
            n_spins=n_sub_spins,
            coupling_strength=model.config.coupling_strength,
            external_field_strength=model.config.external_field_strength,
            use_sparse=model.config.use_sparse,
            device=model.config.device
        )
        
        sub_model = IsingModel(sub_config)
        
        # Copy relevant spins and fields
        sub_model.spins = model.spins[start_idx:end_idx].clone()
        sub_model.external_fields = model.external_fields[start_idx:end_idx].clone()
        
        # Copy relevant coupling submatrix
        if model.config.use_sparse:
            # Handle sparse coupling matrix
            dense_couplings = model.couplings.to_dense()
            sub_couplings = dense_couplings[start_idx:end_idx, start_idx:end_idx]
            sub_model.couplings = sub_couplings.to_sparse_coo()
        else:
            sub_model.couplings = model.couplings[start_idx:end_idx, start_idx:end_idx].clone()
        
        return sub_model
    
    def _single_sweep_update(self, annealer: GPUAnnealer, model: IsingModel) -> None:
        """Perform single sweep update on a model."""
        # Simplified single sweep - would need more sophisticated implementation
        n_spins = model.n_spins
        for _ in range(n_spins):
            spin_idx = torch.randint(0, n_spins, (1,)).item()
            model.flip_spin(spin_idx)
    
    def _synchronize_boundaries(self, model_parts: List[IsingModel]) -> None:
        """Synchronize boundary spins between model parts."""
        # Simple boundary synchronization - average boundary spins
        for i in range(len(model_parts) - 1):
            # Get boundary spins
            right_boundary = model_parts[i].spins[-1]
            left_boundary = model_parts[i + 1].spins[0]
            
            # Average and round to nearest ±1
            avg_spin = (right_boundary + left_boundary) / 2
            new_spin = torch.sign(avg_spin)
            
            # Update both models
            model_parts[i].spins[-1] = new_spin
            model_parts[i + 1].spins[0] = new_spin
    
    def _combine_configurations(self, model_parts: List[IsingModel]) -> torch.Tensor:
        """Combine spin configurations from model parts."""
        configurations = [part.spins for part in model_parts]
        return torch.cat(configurations, dim=0)
    
    def _attempt_replica_exchanges(self, replicas: List[IsingModel], energies: List[float]) -> int:
        """Attempt replica exchanges between adjacent temperatures."""
        exchanges = 0
        n_replicas = len(replicas)
        
        # Get temperatures from annealers
        temperatures = [annealer.config.initial_temp for annealer in self.annealers]
        
        for i in range(n_replicas - 1):
            # Calculate exchange probability
            beta1 = 1.0 / temperatures[i]
            beta2 = 1.0 / temperatures[i + 1]
            energy1 = energies[i]
            energy2 = energies[i + 1]
            
            delta_beta = beta2 - beta1
            delta_energy = energy1 - energy2
            exchange_prob = min(1.0, torch.exp(delta_beta * delta_energy).item())
            
            # Accept or reject exchange
            if torch.rand(1).item() < exchange_prob:
                # Swap spin configurations
                temp_spins = replicas[i].spins.clone()
                replicas[i].spins = replicas[i + 1].spins.clone()
                replicas[i + 1].spins = temp_spins
                
                exchanges += 1
        
        return exchanges
    
    def get_device_utilization(self) -> Dict[str, float]:
        """Get GPU utilization statistics."""
        utilization = {}
        
        for i, device in enumerate(self.devices):
            if torch.cuda.is_available():
                # Get memory usage
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total = torch.cuda.get_device_properties(device).total_memory
                
                utilization[f"gpu_{device.index}"] = {
                    "memory_allocated": allocated,
                    "memory_reserved": reserved,
                    "memory_total": total,
                    "memory_utilization": allocated / total * 100
                }
        
        return utilization
    
    def cleanup(self):
        """Clean up resources and clear GPU memory."""
        for device in self.devices:
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()


class LoadBalancer:
    """Load balancer for distributing work across GPUs."""
    
    def __init__(self, devices: List[torch.device]):
        """Initialize load balancer.
        
        Args:
            devices: List of available devices
        """
        self.devices = devices
        self.device_loads = {device: 0.0 for device in devices}
        self.device_capabilities = self._assess_device_capabilities()
    
    def _assess_device_capabilities(self) -> Dict[torch.device, float]:
        """Assess relative capabilities of each device."""
        capabilities = {}
        
        for device in self.devices:
            if torch.cuda.is_available() and device.type == 'cuda':
                props = torch.cuda.get_device_properties(device)
                # Simple capability score based on memory and compute units
                score = props.total_memory * props.multi_processor_count
                capabilities[device] = score
            else:
                capabilities[device] = 1.0  # Default for CPU
        
        # Normalize capabilities
        max_capability = max(capabilities.values())
        for device in capabilities:
            capabilities[device] /= max_capability
        
        return capabilities
    
    def select_device(self, workload_size: float) -> torch.device:
        """Select best device for given workload.
        
        Args:
            workload_size: Relative size of workload
            
        Returns:
            Selected device
        """
        # Calculate effective load (current load / capability)
        effective_loads = {}
        for device in self.devices:
            effective_load = self.device_loads[device] / self.device_capabilities[device]
            effective_loads[device] = effective_load
        
        # Select device with lowest effective load
        best_device = min(effective_loads.keys(), key=lambda d: effective_loads[d])
        
        # Update load
        self.device_loads[best_device] += workload_size
        
        return best_device
    
    def release_device(self, device: torch.device, workload_size: float):
        """Release device after completing workload.
        
        Args:
            device: Device to release
            workload_size: Size of completed workload
        """
        self.device_loads[device] = max(0.0, self.device_loads[device] - workload_size)
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across devices."""
        return {str(device): load for device, load in self.device_loads.items()}