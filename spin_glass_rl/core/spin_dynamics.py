"""Spin dynamics and Monte Carlo update rules."""

from typing import Callable, Optional, Tuple
import numpy as np
import torch
from enum import Enum

from spin_glass_rl.core.ising_model import IsingModel


class UpdateRule(Enum):
    """Available spin update rules."""
    METROPOLIS = "metropolis"
    GLAUBER = "glauber"
    HEAT_BATH = "heat_bath"
    WOLFF = "wolff"


class SpinDynamics:
    """
    Handles spin dynamics and Monte Carlo updates for Ising models.
    
    Implements various update algorithms including Metropolis, Glauber,
    heat bath, and cluster algorithms like Wolff.
    """
    
    def __init__(
        self,
        model: IsingModel,
        temperature: float = 1.0,
        update_rule: UpdateRule = UpdateRule.METROPOLIS,
        random_seed: Optional[int] = None
    ):
        self.model = model
        self.temperature = temperature
        self.update_rule = update_rule
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Statistics tracking
        self.n_accepted = 0
        self.n_rejected = 0
        self.energy_history = []
        
        # Update function mapping
        self._update_functions = {
            UpdateRule.METROPOLIS: self._metropolis_update,
            UpdateRule.GLAUBER: self._glauber_update,
            UpdateRule.HEAT_BATH: self._heat_bath_update,
            UpdateRule.WOLFF: self._wolff_update,
        }
    
    def set_temperature(self, temperature: float) -> None:
        """Set system temperature."""
        self.temperature = max(temperature, 1e-10)  # Avoid division by zero
    
    def single_spin_update(self, site: Optional[int] = None) -> Tuple[bool, float]:
        """
        Perform single spin update at given site or random site.
        
        Returns:
            (accepted, delta_energy): Whether update was accepted and energy change
        """
        if site is None:
            site = torch.randint(0, self.model.n_spins, (1,)).item()
        
        return self._update_functions[self.update_rule](site)
    
    def sweep(self) -> float:
        """
        Perform one Monte Carlo sweep (N single spin updates).
        
        Returns:
            Current energy after sweep
        """
        n_accepted_sweep = 0
        
        for _ in range(self.model.n_spins):
            accepted, _ = self.single_spin_update()
            if accepted:
                n_accepted_sweep += 1
        
        current_energy = self.model.compute_energy()
        self.energy_history.append(current_energy)
        
        return current_energy
    
    def run_dynamics(self, n_sweeps: int, record_interval: int = 1) -> dict:
        """
        Run Monte Carlo dynamics for specified number of sweeps.
        
        Args:
            n_sweeps: Number of Monte Carlo sweeps
            record_interval: Record energy every N sweeps
            
        Returns:
            Dictionary with dynamics statistics
        """
        initial_energy = self.model.compute_energy()
        
        for sweep in range(n_sweeps):
            self.sweep()
            
            if sweep % record_interval == 0:
                current_energy = self.model.compute_energy()
                self.energy_history.append(current_energy)
        
        final_energy = self.model.compute_energy()
        
        return {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_history": self.energy_history.copy(),
            "acceptance_rate": self.get_acceptance_rate(),
            "n_sweeps": n_sweeps,
            "temperature": self.temperature,
        }
    
    def _metropolis_update(self, site: int) -> Tuple[bool, float]:
        """Metropolis algorithm update."""
        # Calculate energy change if spin is flipped
        local_field = self.model.get_local_field(site)
        delta_energy = 2.0 * self.model.spins[site].item() * local_field
        
        # Accept or reject based on Metropolis criterion
        if delta_energy <= 0:
            # Always accept energy-lowering moves
            self.model.flip_spin(site)
            self.n_accepted += 1
            return True, delta_energy
        else:
            # Accept with probability exp(-ΔE/kT)
            acceptance_prob = torch.exp(-delta_energy / self.temperature)
            if torch.rand(1).item() < acceptance_prob:
                self.model.flip_spin(site)
                self.n_accepted += 1
                return True, delta_energy
            else:
                self.n_rejected += 1
                return False, 0.0
    
    def _glauber_update(self, site: int) -> Tuple[bool, float]:
        """Glauber dynamics update."""
        local_field = self.model.get_local_field(site)
        
        # Probability of spin being +1
        prob_up = 1.0 / (1.0 + torch.exp(-2.0 * local_field / self.temperature))
        
        # Set spin based on probability
        new_spin = 1 if torch.rand(1).item() < prob_up else -1
        
        if new_spin != self.model.spins[site].item():
            delta_energy = self.model.flip_spin(site)
            self.n_accepted += 1
            return True, delta_energy
        else:
            self.n_rejected += 1
            return False, 0.0
    
    def _heat_bath_update(self, site: int) -> Tuple[bool, float]:
        """Heat bath algorithm update."""
        local_field = self.model.get_local_field(site)
        
        # Probability for spin to be +1 in thermal equilibrium
        beta = 1.0 / self.temperature
        prob_up = 1.0 / (1.0 + torch.exp(-2.0 * beta * local_field))
        
        old_spin = self.model.spins[site].item()
        new_spin = 1 if torch.rand(1).item() < prob_up else -1
        
        if new_spin != old_spin:
            self.model.spins[site] = new_spin
            self.model._invalidate_cache()
            delta_energy = 2.0 * old_spin * local_field
            self.n_accepted += 1
            return True, -delta_energy  # Negative because we flipped
        else:
            self.n_rejected += 1
            return False, 0.0
    
    def _wolff_update(self, site: int) -> Tuple[bool, float]:
        """
        Wolff cluster algorithm update.
        
        Note: Simplified implementation for demonstration.
        Full implementation would build and flip entire clusters.
        """
        # For now, fall back to Metropolis for single-site updates
        return self._metropolis_update(site)
    
    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        total_attempts = self.n_accepted + self.n_rejected
        if total_attempts == 0:
            return 0.0
        return self.n_accepted / total_attempts
    
    def reset_statistics(self) -> None:
        """Reset dynamics statistics."""
        self.n_accepted = 0
        self.n_rejected = 0
        self.energy_history = []
    
    def get_autocorrelation_time(self, observable: str = "energy") -> float:
        """
        Estimate autocorrelation time for given observable.
        
        Args:
            observable: Observable to analyze ("energy", "magnetization")
            
        Returns:
            Estimated autocorrelation time in sweep units
        """
        if observable == "energy":
            data = np.array(self.energy_history)
        elif observable == "magnetization":
            # Would need to track magnetization history
            raise NotImplementedError("Magnetization autocorrelation not implemented")
        else:
            raise ValueError(f"Unknown observable: {observable}")
        
        if len(data) < 10:
            return float("inf")
        
        # Simple exponential fit to autocorrelation function
        # This is a simplified implementation
        autocorr = np.correlate(data - data.mean(), data - data.mean(), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # Find where autocorrelation drops to 1/e
        try:
            tau_index = np.where(autocorr < 1.0 / np.e)[0][0]
            return float(tau_index)
        except IndexError:
            return float(len(autocorr))
    
    def thermal_equilibrium_check(self, window_size: int = 100) -> bool:
        """
        Check if system has reached thermal equilibrium.
        
        Args:
            window_size: Size of window for equilibrium check
            
        Returns:
            True if system appears to be in equilibrium
        """
        if len(self.energy_history) < 2 * window_size:
            return False
        
        recent_energies = np.array(self.energy_history[-window_size:])
        older_energies = np.array(self.energy_history[-2*window_size:-window_size])
        
        # Use Welch's t-test to compare means
        from scipy import stats
        _, p_value = stats.ttest_ind(recent_energies, older_energies)
        
        # If p-value > 0.05, means are not significantly different
        return p_value > 0.05
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpinDynamics(temperature={self.temperature:.4f}, "
            f"update_rule={self.update_rule.value}, "
            f"acceptance_rate={self.get_acceptance_rate():.4f})"
        )