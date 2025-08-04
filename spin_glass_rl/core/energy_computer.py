"""Efficient energy computation for Ising models."""

from typing import List, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum

from spin_glass_rl.core.ising_model import IsingModel


class ComputeMode(Enum):
    """Energy computation modes."""
    FULL = "full"
    INCREMENTAL = "incremental"
    VECTORIZED = "vectorized"


@dataclass
class EnergyStats:
    """Energy statistics and decomposition."""
    total_energy: float
    interaction_energy: float
    field_energy: float
    per_spin_energy: Optional[torch.Tensor] = None
    energy_distribution: Optional[torch.Tensor] = None


class EnergyComputer:
    """
    Efficient energy computation for Ising models with various optimization strategies.
    
    Provides methods for full energy calculation, incremental updates,
    and vectorized batch computations.
    """
    
    def __init__(self, model: IsingModel, mode: ComputeMode = ComputeMode.FULL):
        self.model = model
        self.mode = mode
        
        # Cache for incremental updates
        self._cached_energy = None
        self._cache_valid = False
        self._spin_contributions = None
        
        # Precomputed matrices for vectorized operations
        self._interaction_matrix = None
        self._field_vector = None
        self._matrix_valid = False
    
    def compute_total_energy(self, spins: Optional[torch.Tensor] = None) -> float:
        """
        Compute total energy of the system.
        
        Args:
            spins: Optional spin configuration. If None, uses model's current spins.
            
        Returns:
            Total energy of the system
        """
        if spins is None:
            spins = self.model.spins
        
        if self.mode == ComputeMode.FULL:
            return self._compute_full_energy(spins)
        elif self.mode == ComputeMode.INCREMENTAL:
            return self._compute_incremental_energy(spins)
        elif self.mode == ComputeMode.VECTORIZED:
            return self._compute_vectorized_energy(spins)
    
    def compute_energy_change(self, flip_site: int) -> float:
        """
        Compute energy change if spin at flip_site is flipped.
        
        Args:
            flip_site: Index of spin to flip
            
        Returns:
            Energy change ΔE = E_new - E_old
        """
        # Local field at the flip site
        local_field = self._compute_local_field(flip_site)
        
        # Energy change: ΔE = 2 * s_i * h_local
        delta_energy = 2.0 * self.model.spins[flip_site].item() * local_field
        
        return delta_energy
    
    def compute_energy_stats(self, spins: Optional[torch.Tensor] = None) -> EnergyStats:
        """
        Compute detailed energy statistics and decomposition.
        
        Args:
            spins: Optional spin configuration
            
        Returns:
            EnergyStats object with detailed breakdown
        """
        if spins is None:
            spins = self.model.spins
        
        # Interaction energy: -1/2 * Σ J_ij * s_i * s_j
        interaction_energy = self._compute_interaction_energy(spins)
        
        # Field energy: -Σ h_i * s_i
        field_energy = self._compute_field_energy(spins)
        
        total_energy = interaction_energy + field_energy
        
        # Per-spin energy contributions
        per_spin_energy = self._compute_per_spin_energy(spins)
        
        return EnergyStats(
            total_energy=total_energy,
            interaction_energy=interaction_energy,
            field_energy=field_energy,
            per_spin_energy=per_spin_energy
        )
    
    def compute_energy_gradient(self, spins: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy gradient with respect to spin variables.
        
        Args:
            spins: Optional spin configuration
            
        Returns:
            Gradient vector ∇E with respect to each spin
        """
        if spins is None:
            spins = self.model.spins
        
        # Gradient: ∂E/∂s_i = -h_i - Σ J_ij * s_j
        gradient = torch.zeros_like(spins, dtype=torch.float)
        
        for i in range(self.model.n_spins):
            local_field = self._compute_local_field(i, spins)
            gradient[i] = -local_field
        
        return gradient
    
    def compute_batch_energies(self, spin_configs: torch.Tensor) -> torch.Tensor:
        """
        Compute energies for batch of spin configurations.
        
        Args:
            spin_configs: Batch of spin configurations, shape (batch_size, n_spins)
            
        Returns:
            Energy for each configuration, shape (batch_size,)
        """
        batch_size = spin_configs.shape[0]
        energies = torch.zeros(batch_size, device=self.model.device)
        
        for b in range(batch_size):
            energies[b] = self.compute_total_energy(spin_configs[b])
        
        return energies
    
    def _compute_full_energy(self, spins: torch.Tensor) -> float:
        """Compute full energy from scratch."""
        interaction_energy = self._compute_interaction_energy(spins)
        field_energy = self._compute_field_energy(spins)
        return interaction_energy + field_energy
    
    def _compute_incremental_energy(self, spins: torch.Tensor) -> float:
        """Compute energy using incremental updates."""
        if not self._cache_valid or self._cached_energy is None:
            # First computation or cache invalid
            self._cached_energy = self._compute_full_energy(spins)
            self._cache_valid = True
        
        return self._cached_energy
    
    def _compute_vectorized_energy(self, spins: torch.Tensor) -> float:
        """Compute energy using vectorized operations."""
        self._ensure_matrices_valid()
        
        # Vectorized computation: E = -1/2 * s^T J s - h^T s
        if self.model.config.use_sparse:
            interaction_term = -0.5 * torch.sparse.mm(
                spins.unsqueeze(0).float(), 
                torch.sparse.mm(self._interaction_matrix, spins.unsqueeze(1).float())
            ).item()
        else:
            interaction_term = -0.5 * torch.dot(
                spins.float(), torch.mv(self._interaction_matrix, spins.float())
            ).item()
        
        field_term = -torch.dot(self._field_vector, spins.float()).item()
        
        return interaction_term + field_term
    
    def _compute_interaction_energy(self, spins: torch.Tensor) -> float:
        """Compute interaction energy component."""
        if self.model.config.use_sparse:
            # Sparse matrix multiplication
            interaction_energy = -0.5 * torch.sparse.mm(
                spins.unsqueeze(0).float(),
                torch.sparse.mm(self.model.couplings, spins.unsqueeze(1).float())
            ).item()
        else:
            # Dense matrix multiplication
            interaction_energy = -0.5 * torch.dot(
                spins.float(), torch.mv(self.model.couplings, spins.float())
            ).item()
        
        return interaction_energy
    
    def _compute_field_energy(self, spins: torch.Tensor) -> float:
        """Compute external field energy component."""
        return -torch.dot(self.model.external_fields, spins.float()).item()
    
    def _compute_local_field(self, site: int, spins: Optional[torch.Tensor] = None) -> float:
        """Compute local magnetic field at given site."""
        if spins is None:
            spins = self.model.spins
        
        # Coupling contribution: Σ J_ij * s_j
        if self.model.config.use_sparse:
            coupling_field = torch.sparse.mm(
                self.model.couplings[site:site+1, :], spins.unsqueeze(1).float()
            ).item()
        else:
            coupling_field = torch.dot(self.model.couplings[site], spins.float()).item()
        
        # External field contribution
        external_field = self.model.external_fields[site].item()
        
        return coupling_field + external_field
    
    def _compute_per_spin_energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute energy contribution from each spin."""
        per_spin_energy = torch.zeros(self.model.n_spins, device=self.model.device)
        
        for i in range(self.model.n_spins):
            # Local interaction energy
            local_field = self._compute_local_field(i, spins)
            interaction_contribution = -0.5 * spins[i].float() * local_field
            
            # External field contribution
            field_contribution = -self.model.external_fields[i] * spins[i].float()
            
            per_spin_energy[i] = interaction_contribution + field_contribution
        
        return per_spin_energy
    
    def _ensure_matrices_valid(self) -> None:
        """Ensure precomputed matrices are valid."""
        if not self._matrix_valid:
            self._interaction_matrix = self.model.couplings.clone()
            self._field_vector = self.model.external_fields.clone()
            self._matrix_valid = True
    
    def invalidate_cache(self) -> None:
        """Invalidate all caches."""
        self._cache_valid = False
        self._matrix_valid = False
        self._cached_energy = None
        self._spin_contributions = None
    
    def update_incremental_cache(self, flip_site: int, delta_energy: float) -> None:
        """Update incremental energy cache after spin flip."""
        if self._cache_valid and self._cached_energy is not None:
            self._cached_energy += delta_energy
    
    def set_mode(self, mode: ComputeMode) -> None:
        """Set computation mode."""
        self.mode = mode
        if mode != ComputeMode.INCREMENTAL:
            self._cache_valid = False
    
    def benchmark_modes(self, n_trials: int = 100) -> dict:
        """
        Benchmark different computation modes.
        
        Args:
            n_trials: Number of benchmark trials
            
        Returns:
            Dictionary with timing results for each mode
        """
        import time
        
        results = {}
        
        for mode in ComputeMode:
            self.set_mode(mode)
            
            times = []
            for _ in range(n_trials):
                start_time = time.time()
                _ = self.compute_total_energy()
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[mode.value] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
            }
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnergyComputer(mode={self.mode.value}, "
            f"n_spins={self.model.n_spins}, "
            f"cached={self._cache_valid})"
        )