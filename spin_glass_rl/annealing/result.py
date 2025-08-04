"""Annealing result data structures."""

from typing import Dict, List, Optional
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class AnnealingResult:
    """Result of annealing optimization."""
    
    # Best solution found
    best_configuration: torch.Tensor
    best_energy: float
    
    # Optimization trajectory
    energy_history: List[float]
    temperature_history: List[float]
    acceptance_rate_history: List[float]
    
    # Timing and performance
    total_time: float
    n_sweeps: int
    convergence_sweep: Optional[int] = None
    
    # Additional statistics
    final_temperature: float = 0.0
    final_acceptance_rate: float = 0.0
    energy_std: float = 0.0
    
    # Metadata
    algorithm: str = "simulated_annealing"
    device: str = "cpu"
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Compute derived statistics."""
        if self.energy_history:
            self.energy_std = float(np.std(self.energy_history))
            
            # Find convergence point (where energy stabilizes)
            if len(self.energy_history) > 10:
                window_size = min(50, len(self.energy_history) // 4)
                energies = np.array(self.energy_history)
                
                for i in range(window_size, len(energies)):
                    recent_std = np.std(energies[i-window_size:i])
                    if recent_std < 0.01 * abs(self.best_energy):
                        self.convergence_sweep = i - window_size
                        break
        
        if self.temperature_history:
            self.final_temperature = self.temperature_history[-1]
        
        if self.acceptance_rate_history:
            self.final_acceptance_rate = self.acceptance_rate_history[-1]
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "best_energy": self.best_energy,
            "total_time": self.total_time,
            "n_sweeps": self.n_sweeps,
            "convergence_sweep": self.convergence_sweep,
            "final_temperature": self.final_temperature,
            "final_acceptance_rate": self.final_acceptance_rate,
            "energy_std": self.energy_std,
            "algorithm": self.algorithm,
            "device": self.device,
        }
    
    def plot_trajectory(self, save_path: Optional[str] = None) -> None:
        """Plot annealing trajectory."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Energy trajectory
            ax1.plot(self.energy_history)
            ax1.set_xlabel('Sweep')
            ax1.set_ylabel('Energy')
            ax1.set_title('Energy vs Sweep')
            ax1.grid(True)
            
            if self.convergence_sweep:
                ax1.axvline(self.convergence_sweep, color='red', linestyle='--', 
                           label=f'Convergence (sweep {self.convergence_sweep})')
                ax1.legend()
            
            # Temperature schedule
            ax2.plot(self.temperature_history)
            ax2.set_xlabel('Sweep')
            ax2.set_ylabel('Temperature')
            ax2.set_title('Temperature Schedule')
            ax2.set_yscale('log')
            ax2.grid(True)
            
            # Acceptance rate
            ax3.plot(self.acceptance_rate_history)
            ax3.set_xlabel('Sweep')
            ax3.set_ylabel('Acceptance Rate')
            ax3.set_title('Acceptance Rate vs Sweep')
            ax3.grid(True)
            
            # Energy histogram
            ax4.hist(self.energy_history[len(self.energy_history)//2:], bins=50, alpha=0.7)
            ax4.axvline(self.best_energy, color='red', linestyle='--', 
                       label=f'Best Energy: {self.best_energy:.4f}')
            ax4.set_xlabel('Energy')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Energy Distribution (2nd Half)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def save(self, filepath: str) -> None:
        """Save result to file."""
        data = {
            "best_configuration": self.best_configuration.cpu().numpy(),
            "best_energy": self.best_energy,
            "energy_history": self.energy_history,
            "temperature_history": self.temperature_history,
            "acceptance_rate_history": self.acceptance_rate_history,
            "total_time": self.total_time,
            "n_sweeps": self.n_sweeps,
            "convergence_sweep": self.convergence_sweep,
            "final_temperature": self.final_temperature,
            "final_acceptance_rate": self.final_acceptance_rate,
            "energy_std": self.energy_std,
            "algorithm": self.algorithm,
            "device": self.device,
            "random_seed": self.random_seed,
        }
        
        np.savez_compressed(filepath, **data)
    
    @classmethod
    def load(cls, filepath: str) -> "AnnealingResult":
        """Load result from file."""
        data = np.load(filepath, allow_pickle=True)
        
        return cls(
            best_configuration=torch.from_numpy(data["best_configuration"]),
            best_energy=float(data["best_energy"]),
            energy_history=data["energy_history"].tolist(),
            temperature_history=data["temperature_history"].tolist(),
            acceptance_rate_history=data["acceptance_rate_history"].tolist(),
            total_time=float(data["total_time"]),
            n_sweeps=int(data["n_sweeps"]),
            convergence_sweep=int(data["convergence_sweep"]) if data["convergence_sweep"] else None,
            final_temperature=float(data["final_temperature"]),
            final_acceptance_rate=float(data["final_acceptance_rate"]),
            energy_std=float(data["energy_std"]),
            algorithm=str(data["algorithm"]),
            device=str(data["device"]),
            random_seed=int(data["random_seed"]) if data["random_seed"] else None,
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AnnealingResult(best_energy={self.best_energy:.6f}, "
            f"n_sweeps={self.n_sweeps}, "
            f"time={self.total_time:.3f}s, "
            f"converged_at={self.convergence_sweep})"
        )