"""Coupling matrix management for Ising models."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from dataclasses import dataclass


@dataclass
class CouplingPattern:
    """Predefined coupling patterns for common problem types."""
    name: str
    description: str
    generator_func: Optional[callable] = None


class CouplingMatrix:
    """
    Manages coupling matrices for Ising models with various connectivity patterns.
    
    Supports both dense and sparse representations, common graph topologies,
    and problem-specific coupling patterns.
    """
    
    # Predefined coupling patterns
    PATTERNS = {
        "fully_connected": CouplingPattern(
            "fully_connected",
            "All spins coupled to all other spins"
        ),
        "nearest_neighbor": CouplingPattern(
            "nearest_neighbor", 
            "Spins coupled only to immediate neighbors"
        ),
        "random_graph": CouplingPattern(
            "random_graph",
            "Random sparse connectivity"
        ),
        "small_world": CouplingPattern(
            "small_world",
            "Small-world network topology"
        ),
        "scale_free": CouplingPattern(
            "scale_free", 
            "Scale-free network topology"
        ),
    }
    
    def __init__(
        self,
        n_spins: int,
        use_sparse: bool = True,
        device: str = "cpu"
    ):
        self.n_spins = n_spins
        self.use_sparse = use_sparse
        self.device = torch.device(device)
        
        # Initialize empty coupling matrix
        if use_sparse:
            self.matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                (n_spins, n_spins),
                device=self.device
            )
        else:
            self.matrix = torch.zeros((n_spins, n_spins), device=self.device, dtype=torch.float32)
        
        # Track connectivity statistics
        self.n_couplings = 0
        self.density = 0.0
    
    def set_coupling(self, i: int, j: int, strength: float) -> None:
        """Set coupling strength between spins i and j."""
        if i >= self.n_spins or j >= self.n_spins:
            raise ValueError(f"Spin indices must be < {self.n_spins}")
        
        if self.use_sparse:
            # Convert to dense temporarily for modification
            dense = self.matrix.to_dense()
            old_value = dense[i, j].item()
            dense[i, j] = strength
            dense[j, i] = strength  # Ensure symmetry
            self.matrix = dense.to_sparse()
            
            # Update coupling count
            if old_value == 0 and strength != 0:
                self.n_couplings += 1
            elif old_value != 0 and strength == 0:
                self.n_couplings -= 1
        else:
            old_value = self.matrix[i, j].item()
            self.matrix[i, j] = strength
            self.matrix[j, i] = strength
            
            if old_value == 0 and strength != 0:
                self.n_couplings += 1
            elif old_value != 0 and strength == 0:
                self.n_couplings -= 1
        
        self._update_density()
    
    def set_couplings_batch(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """
        Set multiple couplings at once for efficiency.
        
        Args:
            indices: Tensor of shape (2, n_couplings) with i,j indices
            values: Tensor of shape (n_couplings,) with coupling strengths
        """
        if self.use_sparse:
            # Create symmetric indices
            sym_indices = torch.cat([indices, indices.flip(0)], dim=1)
            sym_values = torch.cat([values, values], dim=0)
            
            self.matrix = torch.sparse_coo_tensor(
                sym_indices, sym_values, (self.n_spins, self.n_spins), device=self.device
            ).coalesce()
        else:
            self.matrix[indices[0], indices[1]] = values
            self.matrix[indices[1], indices[0]] = values
        
        self.n_couplings = len(values)
        self._update_density()
    
    def generate_pattern(
        self,
        pattern: str,
        strength_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs
    ) -> None:
        """
        Generate coupling matrix with predefined pattern.
        
        Args:
            pattern: Pattern name from PATTERNS dict
            strength_range: Range of coupling strengths
            **kwargs: Pattern-specific parameters
        """
        if pattern not in self.PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern}. Available: {list(self.PATTERNS.keys())}")
        
        if pattern == "fully_connected":
            self._generate_fully_connected(strength_range)
        elif pattern == "nearest_neighbor":
            self._generate_nearest_neighbor(strength_range, **kwargs)
        elif pattern == "random_graph":
            self._generate_random_graph(strength_range, **kwargs)
        elif pattern == "small_world":
            self._generate_small_world(strength_range, **kwargs)
        elif pattern == "scale_free":
            self._generate_scale_free(strength_range, **kwargs)
    
    def _generate_fully_connected(self, strength_range: Tuple[float, float]) -> None:
        """Generate fully connected coupling matrix."""
        min_strength, max_strength = strength_range
        
        if self.use_sparse:
            # Generate all possible pairs
            indices = torch.combinations(torch.arange(self.n_spins), 2).t()
            values = torch.rand(indices.shape[1]) * (max_strength - min_strength) + min_strength
            self.set_couplings_batch(indices, values)
        else:
            # Fill upper triangle
            for i in range(self.n_spins):
                for j in range(i + 1, self.n_spins):
                    strength = torch.rand(1).item() * (max_strength - min_strength) + min_strength
                    self.set_coupling(i, j, strength)
    
    def _generate_nearest_neighbor(
        self, 
        strength_range: Tuple[float, float],
        topology: str = "chain"
    ) -> None:
        """Generate nearest neighbor coupling pattern."""
        min_strength, max_strength = strength_range
        
        if topology == "chain":
            # Linear chain topology
            indices = []
            values = []
            for i in range(self.n_spins - 1):
                indices.append([i, i + 1])
                values.append(torch.rand(1).item() * (max_strength - min_strength) + min_strength)
        
        elif topology == "ring":
            # Ring topology (chain with periodic boundary)
            indices = []
            values = []
            for i in range(self.n_spins):
                j = (i + 1) % self.n_spins
                indices.append([i, j])
                values.append(torch.rand(1).item() * (max_strength - min_strength) + min_strength)
        
        elif topology == "grid":
            # 2D grid topology
            grid_size = int(np.sqrt(self.n_spins))
            if grid_size * grid_size != self.n_spins:
                raise ValueError("Grid topology requires perfect square number of spins")
            
            indices = []
            values = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    i = row * grid_size + col
                    
                    # Right neighbor
                    if col < grid_size - 1:
                        j = row * grid_size + (col + 1)
                        indices.append([i, j])
                        values.append(torch.rand(1).item() * (max_strength - min_strength) + min_strength)
                    
                    # Bottom neighbor
                    if row < grid_size - 1:
                        j = (row + 1) * grid_size + col
                        indices.append([i, j])
                        values.append(torch.rand(1).item() * (max_strength - min_strength) + min_strength)
        
        if indices:
            indices_tensor = torch.tensor(indices, device=self.device).t()
            values_tensor = torch.tensor(values, device=self.device)
            self.set_couplings_batch(indices_tensor, values_tensor)
    
    def _generate_random_graph(
        self,
        strength_range: Tuple[float, float],
        edge_probability: float = 0.1
    ) -> None:
        """Generate random graph coupling pattern."""
        min_strength, max_strength = strength_range
        
        # Generate random edges
        indices = []
        values = []
        
        for i in range(self.n_spins):
            for j in range(i + 1, self.n_spins):
                if torch.rand(1).item() < edge_probability:
                    indices.append([i, j])
                    values.append(torch.rand(1).item() * (max_strength - min_strength) + min_strength)
        
        if indices:
            indices_tensor = torch.tensor(indices, device=self.device).t()
            values_tensor = torch.tensor(values, device=self.device)
            self.set_couplings_batch(indices_tensor, values_tensor)
    
    def _generate_small_world(
        self,
        strength_range: Tuple[float, float],
        k: int = 4,
        p: float = 0.1
    ) -> None:
        """Generate small-world network using Watts-Strogatz model."""
        # Use NetworkX for small-world generation
        G = nx.watts_strogatz_graph(self.n_spins, k, p)
        
        min_strength, max_strength = strength_range
        indices = []
        values = []
        
        for i, j in G.edges():
            indices.append([i, j])
            values.append(torch.uniform(min_strength, max_strength, (1,)).item())
        
        if indices:
            indices_tensor = torch.tensor(indices, device=self.device).t()
            values_tensor = torch.tensor(values, device=self.device)
            self.set_couplings_batch(indices_tensor, values_tensor)
    
    def _generate_scale_free(
        self,
        strength_range: Tuple[float, float],
        m: int = 2
    ) -> None:
        """Generate scale-free network using Barabási-Albert model."""
        G = nx.barabasi_albert_graph(self.n_spins, m)
        
        min_strength, max_strength = strength_range
        indices = []
        values = []
        
        for i, j in G.edges():
            indices.append([i, j])
            values.append(torch.uniform(min_strength, max_strength, (1,)).item())
        
        if indices:
            indices_tensor = torch.tensor(indices, device=self.device).t()
            values_tensor = torch.tensor(values, device=self.device)
            self.set_couplings_batch(indices_tensor, values_tensor)
    
    def get_coupling(self, i: int, j: int) -> float:
        """Get coupling strength between spins i and j."""
        if self.use_sparse:
            return self.matrix.to_dense()[i, j].item()
        else:
            return self.matrix[i, j].item()
    
    def get_neighbors(self, i: int) -> List[int]:
        """Get list of spins coupled to spin i."""
        if self.use_sparse:
            row = self.matrix.to_dense()[i]
        else:
            row = self.matrix[i]
        
        nonzero_indices = torch.nonzero(row).squeeze()
        if nonzero_indices.dim() == 0:
            return [nonzero_indices.item()] if nonzero_indices.numel() > 0 else []
        else:
            return nonzero_indices.tolist()
    
    def get_degree(self, i: int) -> int:
        """Get degree (number of neighbors) of spin i."""
        return len(self.get_neighbors(i))
    
    def get_density(self) -> float:
        """Get coupling density (fraction of possible edges present)."""
        return self.density
    
    def get_connectivity_stats(self) -> Dict:
        """Get connectivity statistics."""
        degrees = [self.get_degree(i) for i in range(self.n_spins)]
        
        return {
            "n_couplings": self.n_couplings,
            "density": self.density,
            "mean_degree": np.mean(degrees),
            "max_degree": np.max(degrees),
            "min_degree": np.min(degrees),
            "std_degree": np.std(degrees),
        }
    
    def to_networkx(self) -> nx.Graph:
        """Convert coupling matrix to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_spins))
        
        if self.use_sparse:
            indices = self.matrix._indices()
            values = self.matrix._values()
            
            for k in range(len(values)):
                i, j = indices[0, k].item(), indices[1, k].item()
                if i < j:  # Avoid duplicate edges
                    G.add_edge(i, j, weight=values[k].item())
        else:
            for i in range(self.n_spins):
                for j in range(i + 1, self.n_spins):
                    weight = self.matrix[i, j].item()
                    if weight != 0:
                        G.add_edge(i, j, weight=weight)
        
        return G
    
    def _update_density(self) -> None:
        """Update coupling density."""
        max_couplings = self.n_spins * (self.n_spins - 1) // 2
        self.density = self.n_couplings / max_couplings if max_couplings > 0 else 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CouplingMatrix(n_spins={self.n_spins}, "
            f"n_couplings={self.n_couplings}, "
            f"density={self.density:.4f}, "
            f"sparse={self.use_sparse})"
        )