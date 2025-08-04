"""Spin-Glass-Anneal-RL: GPU-accelerated optimization via physics-inspired RL."""

from spin_glass_rl._version import __version__
from spin_glass_rl.core import IsingModel, SpinDynamics, CouplingMatrix
from spin_glass_rl.annealing import GPUAnnealer, TemperatureScheduler
from spin_glass_rl.problems import SchedulingProblem, RoutingProblem

__all__ = [
    "__version__",
    "IsingModel",
    "SpinDynamics", 
    "CouplingMatrix",
    "GPUAnnealer",
    "TemperatureScheduler",
    "SchedulingProblem",
    "RoutingProblem",
]