"""GPU-accelerated simulated annealing implementation."""

from typing import Dict, List, Optional, Tuple
import time
import torch
import numpy as np
from dataclasses import dataclass

from spin_glass_rl.core.ising_model import IsingModel
from spin_glass_rl.core.spin_dynamics import SpinDynamics, UpdateRule
from spin_glass_rl.annealing.temperature_scheduler import TemperatureScheduler, ScheduleType
from spin_glass_rl.annealing.result import AnnealingResult


@dataclass
class GPUAnnealerConfig:
    """Configuration for GPU annealer."""
    n_sweeps: int = 1000
    initial_temp: float = 10.0
    final_temp: float = 0.01
    schedule_type: ScheduleType = ScheduleType.GEOMETRIC
    schedule_params: Dict = None
    
    # GPU-specific parameters
    block_size: int = 256
    shared_memory_size: int = 48 * 1024  # 48KB
    
    # Recording parameters
    record_interval: int = 10
    energy_tolerance: float = 1e-8
    
    # Random seed
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.schedule_params is None:
            self.schedule_params = {"alpha": 0.95}


class GPUAnnealer:
    """
    GPU-accelerated simulated annealing for Ising models.
    
    Implements simulated annealing with CUDA acceleration (when available)
    and various temperature schedules and update rules.
    """
    
    def __init__(self, config: GPUAnnealerConfig):
        self.config = config
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Initialize CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        
        if self.use_cuda:
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA not available, using CPU")
        
        # Performance tracking
        self.total_flips = 0
        self.total_time = 0.0
    
    def anneal(
        self, 
        model: IsingModel,
        update_rule: UpdateRule = UpdateRule.METROPOLIS
    ) -> AnnealingResult:
        """
        Run simulated annealing optimization.
        
        Args:
            model: Ising model to optimize
            update_rule: Monte Carlo update rule
            
        Returns:
            AnnealingResult with optimization statistics
        """
        start_time = time.time()
        
        # Move model to GPU if available
        if self.use_cuda and model.device.type != "cuda":
            model = self._move_model_to_gpu(model)
        
        # Create temperature schedule
        schedule = TemperatureScheduler.create_schedule(
            self.config.schedule_type,
            self.config.initial_temp,
            self.config.final_temp,
            self.config.n_sweeps,
            **self.config.schedule_params
        )
        
        # Initialize spin dynamics
        dynamics = SpinDynamics(model, self.config.initial_temp, update_rule)
        
        # Track best solution
        best_energy = model.compute_energy()
        best_configuration = model.get_spins().clone()
        
        # History tracking
        energy_history = [best_energy]
        temperature_history = [self.config.initial_temp]
        acceptance_rate_history = [0.0]
        
        # Main annealing loop
        for sweep in range(self.config.n_sweeps):
            # Update temperature
            current_temp = schedule.update(sweep, acceptance_rate=dynamics.get_acceptance_rate())
            dynamics.set_temperature(current_temp)
            
            # Perform Monte Carlo sweep
            if self.use_cuda:
                current_energy = self._gpu_sweep(model, dynamics)
            else:
                current_energy = self._cpu_sweep(model, dynamics)
            
            # Track best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_configuration = model.get_spins().clone()
            
            # Record statistics
            if sweep % self.config.record_interval == 0:
                energy_history.append(current_energy)
                temperature_history.append(current_temp)
                acceptance_rate_history.append(dynamics.get_acceptance_rate())
                
                # Early stopping check
                if self._check_convergence(energy_history):
                    print(f"Converged at sweep {sweep}")
                    break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create result
        result = AnnealingResult(
            best_configuration=best_configuration.cpu(),
            best_energy=best_energy,
            energy_history=energy_history,
            temperature_history=temperature_history,
            acceptance_rate_history=acceptance_rate_history,
            total_time=total_time,
            n_sweeps=sweep + 1,
            algorithm="simulated_annealing",
            device=str(self.device),
            random_seed=self.config.random_seed
        )
        
        return result
    
    def _move_model_to_gpu(self, model: IsingModel) -> IsingModel:
        """Move Ising model to GPU."""
        # Create new model on GPU
        gpu_config = model.config
        gpu_config.device = "cuda"
        gpu_model = IsingModel(gpu_config)
        
        # Copy data to GPU
        gpu_model.spins = model.spins.to(self.device)
        gpu_model.couplings = model.couplings.to(self.device)
        gpu_model.external_fields = model.external_fields.to(self.device)
        
        return gpu_model
    
    def _cpu_sweep(self, model: IsingModel, dynamics: SpinDynamics) -> float:
        """Perform one Monte Carlo sweep on CPU."""
        return dynamics.sweep()
    
    def _gpu_sweep(self, model: IsingModel, dynamics: SpinDynamics) -> float:
        """Perform one Monte Carlo sweep on GPU."""
        # For now, fall back to CPU implementation
        # In full CUDA implementation, would launch custom kernels
        return self._cpu_sweep(model, dynamics)
    
    def _launch_gpu_kernel(self, model: IsingModel, temperature: float) -> None:
        """Launch CUDA kernel for parallel spin updates."""
        # Placeholder for CUDA kernel launch
        # Would implement custom CUDA kernels for:
        # 1. Parallel spin updates
        # 2. Energy computation
        # 3. Random number generation
        pass
    
    def _check_convergence(self, energy_history: List[float]) -> bool:
        """Check if annealing has converged."""
        if len(energy_history) < 50:
            return False
        
        # Check if energy has stabilized
        recent_energies = energy_history[-20:]
        energy_std = np.std(recent_energies)
        energy_mean = np.mean(recent_energies)
        
        # Converged if relative standard deviation is small
        if abs(energy_mean) > 0:
            relative_std = energy_std / abs(energy_mean)
            return relative_std < self.config.energy_tolerance
        else:
            return energy_std < self.config.energy_tolerance
    
    def benchmark(
        self, 
        model_sizes: List[int],
        n_trials: int = 3
    ) -> Dict:
        """
        Benchmark annealer performance on different model sizes.
        
        Args:
            model_sizes: List of model sizes to test
            n_trials: Number of trials per size
            
        Returns:
            Benchmark results dictionary
        """
        results = {}
        
        for size in model_sizes:
            print(f"Benchmarking size {size}...")
            
            size_results = {
                "times": [],
                "energies": [],
                "sweeps_per_second": [],
            }
            
            for trial in range(n_trials):
                # Create test model
                from spin_glass_rl.core.ising_model import IsingModelConfig
                config = IsingModelConfig(n_spins=size, device=str(self.device))
                model = IsingModel(config)
                
                # Add some random couplings
                for _ in range(size):
                    i, j = np.random.randint(0, size, 2)
                    if i != j:
                        strength = np.random.uniform(-1, 1)
                        model.set_coupling(i, j, strength)
                
                # Run annealing
                result = self.anneal(model)
                
                # Record results
                size_results["times"].append(result.total_time)
                size_results["energies"].append(result.best_energy)
                size_results["sweeps_per_second"].append(
                    result.n_sweeps / result.total_time
                )
            
            # Compute statistics
            results[size] = {
                "mean_time": np.mean(size_results["times"]),
                "std_time": np.std(size_results["times"]),
                "mean_energy": np.mean(size_results["energies"]),
                "std_energy": np.std(size_results["energies"]),
                "mean_sps": np.mean(size_results["sweeps_per_second"]),
                "std_sps": np.std(size_results["sweeps_per_second"]),
            }
        
        return results
    
    def plot_benchmark(self, benchmark_results: Dict, save_path: Optional[str] = None) -> None:
        """Plot benchmark results."""
        try:
            import matplotlib.pyplot as plt
            
            sizes = list(benchmark_results.keys())
            times = [benchmark_results[s]["mean_time"] for s in sizes]
            time_stds = [benchmark_results[s]["std_time"] for s in sizes]
            sps = [benchmark_results[s]["mean_sps"] for s in sizes]
            sps_stds = [benchmark_results[s]["std_sps"] for s in sizes]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time vs size
            ax1.errorbar(sizes, times, yerr=time_stds, marker='o', capsize=5)
            ax1.set_xlabel('Problem Size (# spins)')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Annealing Time vs Problem Size')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Sweeps per second vs size
            ax2.errorbar(sizes, sps, yerr=sps_stds, marker='s', capsize=5, color='red')
            ax2.set_xlabel('Problem Size (# spins)')
            ax2.set_ylabel('Sweeps per Second')
            ax2.set_title('Performance vs Problem Size')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_memory_usage(self) -> Dict:
        """Get GPU memory usage statistics."""
        if not self.use_cuda:
            return {"device": "cpu", "memory_allocated": 0, "memory_reserved": 0}
        
        return {
            "device": str(self.device),
            "memory_allocated": torch.cuda.memory_allocated(self.device),
            "memory_reserved": torch.cuda.memory_reserved(self.device),
            "memory_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "memory_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GPUAnnealer(device={self.device}, "
            f"n_sweeps={self.config.n_sweeps}, "
            f"schedule={self.config.schedule_type.value})"
        )