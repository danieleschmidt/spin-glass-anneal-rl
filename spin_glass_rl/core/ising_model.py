"""Ising model implementation for spin-glass optimization."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass

# Import robust error handling
try:
    from spin_glass_rl.utils.robust_error_handling import (
        InputValidator, ModelConfigurationError, robust_operation
    )
    from spin_glass_rl.utils.comprehensive_monitoring import global_performance_monitor
    ROBUST_FEATURES_AVAILABLE = True
except ImportError:
    ROBUST_FEATURES_AVAILABLE = False
    
    # Fallback implementations
    class InputValidator:
        @staticmethod
        def validate_ising_config(config): pass
        @staticmethod
        def validate_tensor(tensor, name=""): pass
        @staticmethod
        def validate_spins(spins): pass
        @staticmethod
        def validate_couplings(couplings, n_spins): pass
    
    class ModelConfigurationError(Exception): pass
    
    def robust_operation(**kwargs):
        def decorator(func): return func
        return decorator


@dataclass
class IsingModelConfig:
    """Configuration for Ising model parameters."""
    n_spins: int
    coupling_strength: float = 1.0
    external_field_strength: float = 0.5
    use_sparse: bool = True
    device: str = "cpu"


class IsingModel:
    """
    Ising model for spin-glass optimization problems.
    
    Represents a system of interacting binary spins with Hamiltonian:
    H = -Σ J_ij * s_i * s_j - Σ h_i * s_i
    
    Where s_i ∈ {-1, +1}, J_ij are coupling constants, h_i are external fields.
    """
    
    def __init__(self, config: IsingModelConfig):
        # Validate configuration
        if ROBUST_FEATURES_AVAILABLE:
            InputValidator.validate_ising_config(config)
        
        self.config = config
        self.n_spins = config.n_spins
        self.device = torch.device(config.device)
        
        # Initialize spin configuration
        # Initialize spin configuration  
        self.spins = (torch.randint(0, 2, (self.n_spins,), device=self.device) * 2 - 1).float()
        
        # Initialize coupling matrix
        if config.use_sparse:
            self.couplings = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                (self.n_spins, self.n_spins),
                device=self.device
            )
        else:
            self.couplings = torch.zeros(
                (self.n_spins, self.n_spins), device=self.device
            )
        
        # Initialize external fields
        self.external_fields = torch.zeros(self.n_spins, device=self.device)
        
        # Energy cache
        self._energy_cache = None
        self._cache_valid = False
    
    def set_coupling(self, i: int, j: int, strength: float) -> None:
        """Set coupling strength between spins i and j."""
        if self.config.use_sparse:
            # Convert to dense, modify, convert back to sparse
            dense = self.couplings.to_dense()
            dense[i, j] = strength
            dense[j, i] = strength  # Symmetric coupling
            self.couplings = dense.to_sparse()
        else:
            self.couplings[i, j] = strength
            self.couplings[j, i] = strength
        
        self._invalidate_cache()
    
    def set_couplings_from_matrix(self, coupling_matrix: torch.Tensor) -> None:
        """Set all couplings from a matrix."""
        if self.config.use_sparse:
            self.couplings = coupling_matrix.to_sparse()
        else:
            self.couplings = coupling_matrix.clone()
        
        self._invalidate_cache()
    
    def set_external_field(self, i: int, strength: float) -> None:
        """Set external field on spin i."""
        self.external_fields[i] = strength
        self._invalidate_cache()
    
    def set_external_fields(self, fields: torch.Tensor) -> None:
        """Set all external fields."""
        self.external_fields = fields.clone().to(self.device)
        self._invalidate_cache()
    
    def flip_spin(self, i: int) -> float:
        """
        Flip spin i and return energy change.
        
        ΔE = 2 * s_i * (Σ J_ij * s_j + h_i)
        """
        if self.config.use_sparse:
            # Sparse matrix multiplication
            local_field = torch.sparse.mm(
                self.couplings[i:i+1, :], self.spins.unsqueeze(1)
            ).item()
        else:
            local_field = torch.dot(self.couplings[i], self.spins).item()
        
        local_field += self.external_fields[i].item()
        
        delta_energy = 2.0 * self.spins[i].item() * local_field
        
        # Flip the spin
        self.spins[i] *= -1
        self._invalidate_cache()
        
        return delta_energy
    
    @robust_operation(component="IsingModel", operation="compute_energy")
    def compute_energy(self) -> float:
        """Compute total energy of current configuration."""
        if self._cache_valid and self._energy_cache is not None:
            return self._energy_cache
        
        # Interaction energy: -1/2 * Σ J_ij * s_i * s_j
        if self.config.use_sparse:
            interaction_energy = -0.5 * torch.sparse.mm(
                self.spins.unsqueeze(0), torch.sparse.mm(self.couplings, self.spins.unsqueeze(1))
            ).item()
        else:
            interaction_energy = -0.5 * torch.dot(
                self.spins, torch.mv(self.couplings, self.spins)
            ).item()
        
        # External field energy: -Σ h_i * s_i
        field_energy = -torch.dot(self.external_fields, self.spins).item()
        
        total_energy = interaction_energy + field_energy
        
        # Cache the result
        self._energy_cache = total_energy
        self._cache_valid = True
        
        return total_energy
    
    def get_local_field(self, i: int) -> float:
        """Get local magnetic field at spin i."""
        if self.config.use_sparse:
            # Convert row to dense for computation
            dense_row = self.couplings.to_dense()[i, :]
            coupling_field = torch.dot(dense_row, self.spins).item()
        else:
            coupling_field = torch.dot(self.couplings[i], self.spins).item()
        
        return coupling_field + self.external_fields[i].item()
    
    def get_magnetization(self) -> float:
        """Get total magnetization."""
        return self.spins.sum().item() / self.n_spins
    
    def set_spins(self, spins: torch.Tensor) -> None:
        """Set spin configuration."""
        self.spins = spins.clone().to(self.device)
        self._invalidate_cache()
    
    def get_spins(self) -> torch.Tensor:
        """Get current spin configuration."""
        return self.spins.clone()
    
    def reset_to_random(self) -> None:
        """Reset to random spin configuration."""
        self.spins = (torch.randint(0, 2, (self.n_spins,), device=self.device) * 2 - 1).float()
        self._invalidate_cache()
    
    def copy(self) -> "IsingModel":
        """Create a deep copy of the model."""
        new_model = IsingModel(self.config)
        new_model.spins = self.spins.clone()
        new_model.couplings = self.couplings.clone()
        new_model.external_fields = self.external_fields.clone()
        return new_model
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary for serialization."""
        return {
            "config": {
                "n_spins": self.config.n_spins,
                "coupling_strength": self.config.coupling_strength,
                "external_field_strength": self.config.external_field_strength,
                "use_sparse": self.config.use_sparse,
                "device": self.config.device,
            },
            "spins": self.spins.cpu().numpy(),
            "couplings": self.couplings.to_dense().cpu().numpy() if self.config.use_sparse 
                        else self.couplings.cpu().numpy(),
            "external_fields": self.external_fields.cpu().numpy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "IsingModel":
        """Create model from dictionary."""
        config = IsingModelConfig(**data["config"])
        model = cls(config)
        
        model.spins = torch.from_numpy(data["spins"]).to(model.device)
        
        couplings = torch.from_numpy(data["couplings"]).to(model.device)
        if config.use_sparse:
            model.couplings = couplings.to_sparse()
        else:
            model.couplings = couplings
        
        model.external_fields = torch.from_numpy(data["external_fields"]).to(model.device)
        
        return model
    
    def _invalidate_cache(self) -> None:
        """Invalidate energy cache."""
        self._cache_valid = False
    
    def __repr__(self) -> str:
        """String representation."""
        energy = self.compute_energy()
        magnetization = self.get_magnetization()
        return (
            f"IsingModel(n_spins={self.n_spins}, "
            f"energy={energy:.4f}, "
            f"magnetization={magnetization:.4f})"
        )