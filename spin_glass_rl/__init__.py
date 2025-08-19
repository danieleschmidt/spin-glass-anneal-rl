"""
Spin-Glass Annealing RL: GPU-accelerated optimization framework.

A comprehensive framework for solving complex optimization problems using
physics-inspired spin-glass models and reinforcement learning.
"""

# Version
try:
    from spin_glass_rl._version import __version__
except ImportError:
    __version__ = "1.0.0"

# Try to import main components, fall back to minimal versions
try:
    # Full featured imports (require torch, numpy, etc.)
    from spin_glass_rl.core import IsingModel, SpinDynamics, CouplingMatrix
    from spin_glass_rl.annealing import GPUAnnealer, TemperatureScheduler
    from spin_glass_rl.problems import (
        SchedulingProblem, RoutingProblem, ResourceAllocationProblem
    )
    FULL_FEATURES_AVAILABLE = True
except ImportError:
    # Minimal fallback imports (pure Python)
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer
    FULL_FEATURES_AVAILABLE = False

# Always available minimal implementations
from spin_glass_rl.core.minimal_ising import (
    MinimalIsingModel,
    MinimalAnnealer,
    create_test_problem,
    demo_basic_functionality
)

# Export commonly used classes
if FULL_FEATURES_AVAILABLE:
    __all__ = [
        "__version__",
        "IsingModel",
        "SpinDynamics", 
        "CouplingMatrix",
        "GPUAnnealer",
        "TemperatureScheduler",
        "SchedulingProblem",
        "RoutingProblem",
        "MinimalIsingModel",
        "MinimalAnnealer",
        "create_test_problem",
        "demo_basic_functionality",
        "FULL_FEATURES_AVAILABLE"
    ]
else:
    # Alias minimal implementations to main names for compatibility
    IsingModel = MinimalIsingModel
    Annealer = MinimalAnnealer
    
    __all__ = [
        "__version__",
        "IsingModel",
        "Annealer",
        "MinimalIsingModel",
        "MinimalAnnealer", 
        "create_test_problem",
        "demo_basic_functionality",
        "FULL_FEATURES_AVAILABLE"
    ]

# Feature detection
def get_available_features():
    """Return dict of available features."""
    features = {
        "minimal_cpu": True,
        "full_torch": FULL_FEATURES_AVAILABLE,
        "gpu_acceleration": False,
        "quantum_hardware": False,
        "distributed_computing": False
    }
    
    if FULL_FEATURES_AVAILABLE:
        try:
            import torch
            features["full_torch"] = True
            features["gpu_acceleration"] = torch.cuda.is_available()
        except ImportError:
            pass
    
    return features