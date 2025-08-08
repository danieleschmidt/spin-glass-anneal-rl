"""Parallel tempering (replica exchange) implementation."""

from typing import Dict, List, Optional, Tuple
import time
import torch
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from spin_glass_rl.core.ising_model import IsingModel
from spin_glass_rl.core.spin_dynamics import SpinDynamics, UpdateRule
from spin_glass_rl.annealing.result import AnnealingResult
from spin_glass_rl.annealing.cuda_kernels import CUDAKernelManager


@dataclass
class ParallelTemperingConfig:
    """Configuration for parallel tempering."""
    n_replicas: int = 8
    n_sweeps: int = 1000
    temp_min: float = 0.1
    temp_max: float = 10.0
    temp_distribution: str = "geometric"  # "geometric", "linear", "exponential"
    
    # Exchange parameters
    exchange_interval: int = 10
    exchange_method: str = "nearest_neighbor"  # "nearest_neighbor", "all_pairs"
    
    # Performance parameters
    n_threads: Optional[int] = None
    
    # Recording parameters
    record_interval: int = 10
    
    # Random seed
    random_seed: Optional[int] = None


class ParallelTempering:
    """
    Parallel tempering (replica exchange) Monte Carlo.
    
    Runs multiple replicas at different temperatures and exchanges
    configurations between adjacent temperature replicas to improve
    sampling efficiency.
    """
    
    def __init__(self, config: ParallelTemperingConfig):
        self.config = config
        
        # Set random seed
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Generate temperature ladder
        self.temperatures = self._generate_temperature_ladder()
        
        # Initialize replicas
        self.replicas: List[IsingModel] = []
        self.dynamics: List[SpinDynamics] = []
        
        # Exchange statistics
        self.exchange_attempts = np.zeros((self.config.n_replicas - 1,))
        self.exchange_accepts = np.zeros((self.config.n_replicas - 1,))
        
        # History tracking
        self.energy_histories: List[List[float]] = [[] for _ in range(self.config.n_replicas)]
        self.temp_histories: List[List[float]] = [[] for _ in range(self.config.n_replicas)]
        
        # Thread pool for parallel execution
        self.n_threads = config.n_threads or min(config.n_replicas, 8)
        
        # Initialize CUDA support if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        if self.use_cuda:
            self.cuda_kernels = CUDAKernelManager(self.device)
        else:
            self.cuda_kernels = None
    
    def run(
        self,
        model: IsingModel,
        update_rule: UpdateRule = UpdateRule.METROPOLIS
    ) -> AnnealingResult:
        """
        Run parallel tempering optimization.
        
        Args:
            model: Base Ising model to optimize
            update_rule: Monte Carlo update rule
            
        Returns:
            AnnealingResult from lowest temperature replica
        """
        start_time = time.time()
        
        # Initialize replicas
        self._initialize_replicas(model, update_rule)
        
        # Track best solution across all replicas
        best_energy = float('inf')
        best_configuration = None
        best_replica_idx = 0
        
        # Main parallel tempering loop
        for sweep in range(self.config.n_sweeps):
            # Parallel Monte Carlo sweeps
            self._parallel_sweeps()
            
            # Exchange between replicas
            if sweep % self.config.exchange_interval == 0 and sweep > 0:
                self._attempt_exchanges()
            
            # Record statistics
            if sweep % self.config.record_interval == 0:
                self._record_statistics()
                
                # Update best solution
                current_best_energy, current_best_replica = self._find_best_solution()
                if current_best_energy < best_energy:
                    best_energy = current_best_energy
                    best_replica_idx = current_best_replica
                    best_configuration = self.replicas[current_best_replica].get_spins().clone()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create result from lowest temperature replica (index 0)
        result = AnnealingResult(
            best_configuration=best_configuration.cpu(),
            best_energy=best_energy,
            energy_history=self.energy_histories[0],  # Lowest temperature trajectory
            temperature_history=self.temp_histories[0],
            acceptance_rate_history=[d.get_acceptance_rate() for d in self.dynamics],
            total_time=total_time,
            n_sweeps=self.config.n_sweeps,
            algorithm="parallel_tempering",
            device=str(model.device),
            random_seed=self.config.random_seed
        )
        
        return result
    
    def _generate_temperature_ladder(self) -> List[float]:
        """Generate temperature ladder for replicas."""
        if self.config.temp_distribution == "geometric":
            # Geometric spacing: T_i = T_max * (T_min/T_max)^(i/(n-1))
            ratio = self.config.temp_min / self.config.temp_max
            temperatures = [
                self.config.temp_max * (ratio ** (i / (self.config.n_replicas - 1)))
                for i in range(self.config.n_replicas)
            ]
        
        elif self.config.temp_distribution == "linear":
            # Linear spacing
            temperatures = np.linspace(
                self.config.temp_max, self.config.temp_min, self.config.n_replicas
            ).tolist()
        
        elif self.config.temp_distribution == "exponential":
            # Exponential spacing
            temperatures = np.logspace(
                np.log10(self.config.temp_max),
                np.log10(self.config.temp_min),
                self.config.n_replicas
            ).tolist()
        
        else:
            raise ValueError(f"Unknown temperature distribution: {self.config.temp_distribution}")
        
        return temperatures
    
    def _initialize_replicas(self, model: IsingModel, update_rule: UpdateRule) -> None:
        """Initialize replica models and dynamics."""
        self.replicas = []
        self.dynamics = []
        
        for i, temp in enumerate(self.temperatures):
            # Create replica (copy of original model)
            replica = model.copy()
            replica.reset_to_random()  # Start with random configuration
            
            # Create dynamics for this replica
            dynamics = SpinDynamics(replica, temp, update_rule)
            
            self.replicas.append(replica)
            self.dynamics.append(dynamics)
    
    def _parallel_sweeps(self) -> None:
        """Perform Monte Carlo sweeps on all replicas in parallel."""
        if self.n_threads == 1:
            # Sequential execution
            for dynamics in self.dynamics:
                dynamics.sweep()
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(dynamics.sweep) for dynamics in self.dynamics]
                # Wait for all sweeps to complete
                for future in futures:
                    future.result()
    
    def _attempt_exchanges(self) -> None:
        """Attempt replica exchanges."""
        if self.config.exchange_method == "nearest_neighbor":
            self._nearest_neighbor_exchange()
        elif self.config.exchange_method == "all_pairs":
            self._all_pairs_exchange()
        else:
            raise ValueError(f"Unknown exchange method: {self.config.exchange_method}")
    
    def _nearest_neighbor_exchange(self) -> None:
        """Attempt exchanges between nearest neighbor replicas."""
        # Randomly choose starting position (even or odd)
        start = np.random.randint(0, 2)
        
        for i in range(start, self.config.n_replicas - 1, 2):
            self._attempt_single_exchange(i, i + 1)
    
    def _all_pairs_exchange(self) -> None:
        """Attempt exchanges between all replica pairs."""
        # Use CUDA optimized exchange if available
        if self.use_cuda and self.cuda_kernels is not None:
            self._cuda_optimized_exchange()
        else:
            # CPU fallback
            for i in range(self.config.n_replicas - 1):
                for j in range(i + 1, self.config.n_replicas):
                    if np.random.rand() < 0.1:  # Limit exchange attempts
                        self._attempt_single_exchange(i, j)
    
    def _attempt_single_exchange(self, i: int, j: int) -> None:
        """Attempt exchange between replicas i and j."""
        # Calculate exchange probability
        beta_i = 1.0 / self.temperatures[i]
        beta_j = 1.0 / self.temperatures[j]
        
        energy_i = self.replicas[i].compute_energy()
        energy_j = self.replicas[j].compute_energy()
        
        # Metropolis exchange criterion
        delta_beta = beta_j - beta_i
        delta_energy = energy_j - energy_i
        exchange_prob = min(1.0, np.exp(delta_beta * delta_energy))
        
        # Attempt exchange
        pair_idx = min(i, j)  # Index for tracking statistics
        self.exchange_attempts[pair_idx] += 1
        
        if np.random.rand() < exchange_prob:
            # Exchange configurations
            temp_spins = self.replicas[i].get_spins().clone()
            self.replicas[i].set_spins(self.replicas[j].get_spins())
            self.replicas[j].set_spins(temp_spins)
            
            self.exchange_accepts[pair_idx] += 1
    
    def _cuda_optimized_exchange(self) -> None:
        """Perform optimized parallel tempering exchanges using CUDA kernels."""
        if not self.use_cuda or self.cuda_kernels is None:
            return
        
        # Prepare tensors for batch exchange
        n_replicas = len(self.replicas)
        n_spins = self.replicas[0].n_spins
        
        # Stack all spin configurations
        spins_arrays = torch.stack([replica.get_spins() for replica in self.replicas])
        
        # Compute energies for all replicas
        energies = torch.tensor([
            replica.compute_energy() for replica in self.replicas
        ], device=self.device)
        
        # Temperature array
        temperatures = torch.tensor(self.temperatures, device=self.device)
        
        # Use CUDA kernel for parallel exchange
        successful_exchanges = self.cuda_kernels.parallel_tempering_exchange_optimized(
            spins_arrays=spins_arrays,
            energies=energies,
            temperatures=temperatures
        )
        
        # Update replicas with new configurations
        for i, replica in enumerate(self.replicas):
            replica.set_spins(spins_arrays[i])
        
        # Update exchange statistics
        if n_replicas > 1:
            self.exchange_accepts[0] += successful_exchanges
    
    def _record_statistics(self) -> None:
        """Record energy and temperature histories."""
        for i, replica in enumerate(self.replicas):
            energy = replica.compute_energy()
            self.energy_histories[i].append(energy)
            self.temp_histories[i].append(self.temperatures[i])
    
    def _find_best_solution(self) -> Tuple[float, int]:
        """Find best solution across all replicas."""
        best_energy = float('inf')
        best_replica = 0
        
        for i, replica in enumerate(self.replicas):
            energy = replica.compute_energy()
            if energy < best_energy:
                best_energy = energy
                best_replica = i
        
        return best_energy, best_replica
    
    def get_exchange_rates(self) -> np.ndarray:
        """Get exchange acceptance rates between replica pairs."""
        rates = np.zeros_like(self.exchange_accepts)
        for i in range(len(rates)):
            if self.exchange_attempts[i] > 0:
                rates[i] = self.exchange_accepts[i] / self.exchange_attempts[i]
        return rates
    
    def plot_replica_trajectories(self, save_path: Optional[str] = None) -> None:
        """Plot energy trajectories for all replicas."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot energy trajectories
            plt.subplot(2, 2, 1)
            for i, history in enumerate(self.energy_histories):
                if history:  # Check if history exists
                    plt.plot(history, label=f'T={self.temperatures[i]:.2f}', alpha=0.7)
            plt.xlabel('Sweep')
            plt.ylabel('Energy')
            plt.title('Energy Trajectories')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Plot temperature ladder
            plt.subplot(2, 2, 2)
            plt.plot(self.temperatures, 'o-')
            plt.xlabel('Replica Index')
            plt.ylabel('Temperature')
            plt.title('Temperature Ladder')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Plot exchange rates
            plt.subplot(2, 2, 3)
            exchange_rates = self.get_exchange_rates()
            if len(exchange_rates) > 0:
                plt.bar(range(len(exchange_rates)), exchange_rates)
                plt.xlabel('Replica Pair')
                plt.ylabel('Exchange Rate')
                plt.title('Exchange Acceptance Rates')
                plt.grid(True, alpha=0.3)
            
            # Plot energy distribution
            plt.subplot(2, 2, 4)
            final_energies = [history[-1] if history else 0 for history in self.energy_histories]
            plt.hist(final_energies, bins=10, alpha=0.7)
            plt.xlabel('Final Energy')
            plt.ylabel('Count')
            plt.title('Final Energy Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def plot_exchange_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot exchange rate matrix."""
        try:
            import matplotlib.pyplot as plt
            
            # Create exchange matrix
            exchange_matrix = np.zeros((self.config.n_replicas, self.config.n_replicas))
            exchange_rates = self.get_exchange_rates()
            
            for i in range(len(exchange_rates)):
                exchange_matrix[i, i+1] = exchange_rates[i]
                exchange_matrix[i+1, i] = exchange_rates[i]
            
            plt.figure(figsize=(8, 6))
            plt.imshow(exchange_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Exchange Rate')
            plt.xlabel('Replica Index')
            plt.ylabel('Replica Index')
            plt.title('Replica Exchange Rate Matrix')
            
            # Add temperature labels
            temp_labels = [f'{t:.2f}' for t in self.temperatures]
            plt.xticks(range(self.config.n_replicas), temp_labels, rotation=45)
            plt.yticks(range(self.config.n_replicas), temp_labels)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        exchange_rates = self.get_exchange_rates()
        final_energies = [history[-1] if history else float('inf') 
                         for history in self.energy_histories]
        
        return {
            "n_replicas": self.config.n_replicas,
            "temperatures": self.temperatures,
            "exchange_rates": exchange_rates.tolist(),
            "mean_exchange_rate": np.mean(exchange_rates),
            "final_energies": final_energies,
            "best_energy": min(final_energies),
            "energy_range": max(final_energies) - min(final_energies),
            "total_exchanges_attempted": self.exchange_attempts.sum(),
            "total_exchanges_accepted": self.exchange_accepts.sum(),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParallelTempering(n_replicas={self.config.n_replicas}, "
            f"temp_range=[{self.config.temp_min:.2f}, {self.config.temp_max:.2f}], "
            f"n_sweeps={self.config.n_sweeps})"
        )