"""Minimal CPU-only Ising model implementation for basic functionality."""

import math
import random
from typing import List, Tuple, Optional


class MinimalIsingModel:
    """
    Lightweight CPU-only Ising model for basic functionality.
    
    This provides core functionality without heavy dependencies,
    suitable for testing and basic operations.
    """
    
    def __init__(self, n_spins: int):
        """Initialize with n_spins."""
        if n_spins <= 0:
            raise ValueError(f"Number of spins must be positive, got {n_spins}")
        
        self.n_spins = n_spins
        self.spins = [1 if random.random() > 0.5 else -1 for _ in range(n_spins)]
        self.couplings = [[0.0 for _ in range(n_spins)] for _ in range(n_spins)]
        self.external_fields = [0.0] * n_spins
    
    def set_coupling(self, i: int, j: int, strength: float) -> None:
        """Set coupling between spins i and j."""
        if 0 <= i < self.n_spins and 0 <= j < self.n_spins:
            self.couplings[i][j] = strength
            self.couplings[j][i] = strength  # Symmetric
    
    def set_external_field(self, i: int, strength: float) -> None:
        """Set external field on spin i."""
        if 0 <= i < self.n_spins:
            self.external_fields[i] = strength
    
    def compute_energy(self) -> float:
        """Compute total energy of current configuration."""
        energy = 0.0
        
        # Interaction energy: -1/2 * sum_ij J_ij * s_i * s_j
        for i in range(self.n_spins):
            for j in range(i + 1, self.n_spins):  # Avoid double counting
                energy -= self.couplings[i][j] * self.spins[i] * self.spins[j]
        
        # External field energy: -sum_i h_i * s_i
        for i in range(self.n_spins):
            energy -= self.external_fields[i] * self.spins[i]
        
        return energy
    
    def compute_local_field(self, i: int) -> float:
        """Compute local field at spin i."""
        field = self.external_fields[i]
        for j in range(self.n_spins):
            if i != j:
                field += self.couplings[i][j] * self.spins[j]
        return field
    
    def flip_spin(self, i: int) -> float:
        """Flip spin i and return energy change."""
        if not (0 <= i < self.n_spins):
            return 0.0
        
        local_field = self.compute_local_field(i)
        delta_energy = 2.0 * self.spins[i] * local_field
        
        # Flip the spin
        self.spins[i] *= -1
        
        return delta_energy
    
    def get_magnetization(self) -> float:
        """Get total magnetization."""
        return sum(self.spins) / self.n_spins
    
    def copy(self) -> 'MinimalIsingModel':
        """Create a copy of the model."""
        new_model = MinimalIsingModel(self.n_spins)
        new_model.spins = self.spins[:]
        new_model.external_fields = self.external_fields[:]
        for i in range(self.n_spins):
            for j in range(self.n_spins):
                new_model.couplings[i][j] = self.couplings[i][j]
        return new_model


class MinimalAnnealer:
    """Minimal CPU-only simulated annealing implementation."""
    
    def __init__(self, initial_temp: float = 10.0, final_temp: float = 0.01):
        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError(f"Temperatures must be positive: initial={initial_temp}, final={final_temp}")
        if final_temp > initial_temp:
            raise ValueError(f"Final temperature must be <= initial: {final_temp} > {initial_temp}")
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
    
    def metropolis_accept(self, delta_energy: float, temperature: float) -> bool:
        """Metropolis acceptance criterion."""
        if delta_energy <= 0:
            return True
        if temperature <= 0:
            return False
        probability = math.exp(-delta_energy / temperature)
        return random.random() < probability
    
    def anneal(self, model: MinimalIsingModel, n_sweeps: int = 1000) -> Tuple[float, List[float]]:
        """
        Run simulated annealing.
        
        Returns:
            (best_energy, energy_history)
        """
        best_energy = model.compute_energy()
        best_spins = model.spins[:]
        energy_history = [best_energy]
        
        accepted_moves = 0
        total_moves = 0
        
        for sweep in range(n_sweeps):
            # Update temperature (exponential schedule)
            if n_sweeps > 1:
                alpha = sweep / (n_sweeps - 1)
                temperature = self.initial_temp * (self.final_temp / self.initial_temp) ** alpha
            else:
                temperature = self.initial_temp
            
            # Perform one sweep
            sweep_energy_changes = []
            for _ in range(model.n_spins):
                # Choose random spin
                i = random.randint(0, model.n_spins - 1)
                
                # Compute energy change if we flip this spin
                delta_energy = model.flip_spin(i)
                total_moves += 1
                
                # Accept or reject
                if self.metropolis_accept(delta_energy, temperature):
                    accepted_moves += 1
                    sweep_energy_changes.append(delta_energy)
                else:
                    # Reject: flip back
                    model.spins[i] *= -1
                
                # Update best solution
                current_energy = model.compute_energy()
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_spins = model.spins[:]
            
            # Record energy
            if sweep % max(1, n_sweeps // 50) == 0:  # Record ~50 points
                current_energy = model.compute_energy()
                energy_history.append(current_energy)
        
        # Restore best configuration
        model.spins = best_spins
        
        return best_energy, energy_history


def create_test_problem(n_spins: int = 10) -> MinimalIsingModel:
    """Create a test Ising model problem."""
    model = MinimalIsingModel(n_spins)
    
    # Add random couplings
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if random.random() < 0.3:  # 30% coupling density
                strength = random.uniform(-1, 1)
                model.set_coupling(i, j, strength)
    
    # Add random external fields
    for i in range(n_spins):
        if random.random() < 0.2:  # 20% field density
            field = random.uniform(-0.5, 0.5)
            model.set_external_field(i, field)
    
    return model


def demo_basic_functionality():
    """Demonstrate basic spin-glass annealing functionality."""
    print("🧲 Minimal Spin-Glass Annealing Demo")
    print("=" * 40)
    
    # Create test problem
    print("Creating test problem...")
    model = create_test_problem(n_spins=8)
    initial_energy = model.compute_energy()
    initial_magnetization = model.get_magnetization()
    
    print(f"Initial energy: {initial_energy:.4f}")
    print(f"Initial magnetization: {initial_magnetization:.4f}")
    print(f"Initial spins: {model.spins}")
    
    # Run annealing
    print("\nRunning simulated annealing...")
    annealer = MinimalAnnealer(initial_temp=5.0, final_temp=0.01)
    best_energy, energy_history = annealer.anneal(model, n_sweeps=500)
    
    final_magnetization = model.get_magnetization()
    
    print(f"Final energy: {best_energy:.4f}")
    print(f"Final magnetization: {final_magnetization:.4f}")
    print(f"Final spins: {model.spins}")
    print(f"Energy improvement: {initial_energy - best_energy:.4f}")
    print(f"Energy history length: {len(energy_history)}")
    
    print("\n✅ Basic functionality working!")


if __name__ == "__main__":
    demo_basic_functionality()