"""
Production configuration for Spin-Glass-Anneal-RL framework.
"""

from spin_glass_rl.core import IsingModelConfig
from spin_glass_rl.annealing import GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.optimization.adaptive_optimization import AdaptiveConfig, OptimizationStrategy
from spin_glass_rl.optimization.high_performance_computing import ComputeConfig

# Production-ready configurations

SMALL_PROBLEM_CONFIG = IsingModelConfig(
    n_spins=100,
    coupling_strength=1.0,
    external_field_strength=0.5,
    use_sparse=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

MEDIUM_PROBLEM_CONFIG = IsingModelConfig(
    n_spins=500,
    coupling_strength=1.0,
    external_field_strength=0.5,
    use_sparse=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

LARGE_PROBLEM_CONFIG = IsingModelConfig(
    n_spins=2000,
    coupling_strength=1.0,
    external_field_strength=0.5,
    use_sparse=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

PRODUCTION_ANNEALER_CONFIG = GPUAnnealerConfig(
    n_sweeps=5000,
    initial_temp=10.0,
    final_temp=0.001,
    schedule_type=ScheduleType.ADAPTIVE,
    enable_adaptive_optimization=True,
    enable_caching=True,
    enable_performance_profiling=True,
    random_seed=None  # Use random seed for production
)

ADAPTIVE_CONFIG = AdaptiveConfig(
    strategy=OptimizationStrategy.ADAPTIVE_SIMULATED_ANNEALING,
    adaptation_interval=100,
    auto_adjust_temperature=True,
    target_acceptance_rate=0.4,
    enable_early_stopping=True,
    convergence_threshold=1e-6
)

COMPUTE_CONFIG = ComputeConfig(
    enable_multiprocessing=True,
    enable_gpu_acceleration=True,
    batch_size=1000,
    memory_limit_gb=8.0,
    enable_vectorization=True
)
