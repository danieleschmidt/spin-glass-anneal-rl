"""Core spin-glass model components."""

# Try to import full implementations, fall back to minimal
try:
    from spin_glass_rl.core.ising_model import IsingModel
    from spin_glass_rl.core.spin_dynamics import SpinDynamics  
    from spin_glass_rl.core.coupling_matrix import CouplingMatrix
    from spin_glass_rl.core.energy_computer import EnergyComputer
    from spin_glass_rl.core.constraints import ConstraintEncoder
    CORE_FULL_AVAILABLE = True
except ImportError:
    # Import minimal implementations when full features unavailable
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel as IsingModel
    CORE_FULL_AVAILABLE = False

# Always export minimal version
from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer

if CORE_FULL_AVAILABLE:
    __all__ = [
        "IsingModel", "SpinDynamics", "CouplingMatrix", "EnergyComputer", 
        "ConstraintEncoder", "MinimalIsingModel", "MinimalAnnealer"
    ]
else:
    __all__ = ["IsingModel", "MinimalIsingModel", "MinimalAnnealer"]