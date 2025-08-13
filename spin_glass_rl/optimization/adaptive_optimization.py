"""Adaptive optimization strategies for large-scale problems."""

import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    ADAPTIVE_SIMULATED_ANNEALING = "adaptive_sa"
    PARALLEL_TEMPERING = "parallel_tempering"
    POPULATION_ANNEALING = "population_annealing"
    QUANTUM_INSPIRED = "quantum_inspired"
    HYBRID_CLASSICAL_QUANTUM = "hybrid_cq"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_SIMULATED_ANNEALING
    adaptation_interval: int = 100  # sweeps
    performance_window: int = 50
    auto_adjust_temperature: bool = True
    auto_adjust_schedule: bool = True
    auto_tune_parameters: bool = True
    target_acceptance_rate: float = 0.4
    acceptance_tolerance: float = 0.1
    enable_early_stopping: bool = True
    convergence_threshold: float = 1e-6
    max_stagnation_sweeps: int = 500
    enable_restart: bool = True
    restart_temperature_factor: float = 2.0


class AdaptiveOptimizer(ABC):
    """Base class for adaptive optimization strategies."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.adaptation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "energy": [],
            "acceptance_rate": [],
            "temperature": [],
            "convergence_rate": []
        }
        
    @abstractmethod
    def adapt_parameters(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt optimization parameters based on current state."""
        pass
    
    @abstractmethod
    def should_restart(self, current_state: Dict[str, Any]) -> bool:
        """Determine if optimization should restart."""
        pass
    
    def track_performance(self, metrics: Dict[str, float]) -> None:
        """Track performance metrics."""
        for key, value in metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)


class AdaptiveSimulatedAnnealing(AdaptiveOptimizer):
    """Adaptive simulated annealing with dynamic parameter adjustment."""
    
    def __init__(self, config: AdaptiveConfig):
        super().__init__(config)
        self.temperature_history: List[float] = []
        self.energy_history: List[float] = []
        self.last_adaptation_sweep = 0
        
    def adapt_parameters(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt SA parameters based on current performance."""
        adaptations = {}
        current_sweep = current_state.get("sweep", 0)
        
        # Only adapt at specified intervals
        if current_sweep - self.last_adaptation_sweep < self.config.adaptation_interval:
            return adaptations
        
        self.last_adaptation_sweep = current_sweep
        
        # Get recent performance
        recent_energies = self.performance_metrics["energy"][-self.config.performance_window:]
        recent_acceptance = self.performance_metrics["acceptance_rate"][-self.config.performance_window:]
        
        if len(recent_acceptance) < 10:
            return adaptations
        
        # Adapt temperature based on acceptance rate
        if self.config.auto_adjust_temperature:
            avg_acceptance = np.mean(recent_acceptance[-10:])
            target_rate = self.config.target_acceptance_rate
            tolerance = self.config.acceptance_tolerance
            
            current_temp = current_state.get("temperature", 1.0)
            
            if avg_acceptance < target_rate - tolerance:
                # Acceptance too low, increase temperature
                new_temp = current_temp * 1.1
                adaptations["temperature"] = new_temp
                
            elif avg_acceptance > target_rate + tolerance:
                # Acceptance too high, decrease temperature
                new_temp = current_temp * 0.95
                adaptations["temperature"] = max(new_temp, 0.001)  # Minimum temperature
        
        # Adapt cooling schedule based on energy improvement
        if self.config.auto_adjust_schedule and len(recent_energies) >= 20:
            recent_improvement = recent_energies[0] - recent_energies[-1]
            
            if recent_improvement < 0.001:  # Poor improvement
                # Slow down cooling
                adaptations["cooling_factor"] = current_state.get("cooling_factor", 0.99) * 1.01
            elif recent_improvement > 0.1:  # Good improvement
                # Speed up cooling slightly
                adaptations["cooling_factor"] = current_state.get("cooling_factor", 0.99) * 0.99
        
        # Record adaptation
        if adaptations:
            self.adaptation_history.append({
                "sweep": current_sweep,
                "adaptations": adaptations.copy(),
                "metrics": {
                    "avg_acceptance": np.mean(recent_acceptance[-10:]) if recent_acceptance else 0,
                    "energy_improvement": recent_energies[0] - recent_energies[-1] if len(recent_energies) >= 2 else 0
                }
            })
        
        return adaptations
    
    def should_restart(self, current_state: Dict[str, Any]) -> bool:
        """Check if optimization should restart."""
        if not self.config.enable_restart:
            return False
        
        recent_energies = self.performance_metrics["energy"][-self.config.max_stagnation_sweeps:]
        
        if len(recent_energies) < self.config.max_stagnation_sweeps:
            return False
        
        # Check for stagnation
        energy_std = np.std(recent_energies)
        energy_mean = np.mean(recent_energies)
        
        relative_std = energy_std / abs(energy_mean) if energy_mean != 0 else float('inf')
        
        return relative_std < self.config.convergence_threshold


class ParallelTemperingOptimizer(AdaptiveOptimizer):
    """Adaptive parallel tempering with dynamic replica management."""
    
    def __init__(self, config: AdaptiveConfig, n_replicas: int = 8):
        super().__init__(config)
        self.n_replicas = n_replicas
        self.replica_temps: List[float] = []
        self.exchange_rates: List[float] = []
        self.optimal_temp_range = (0.1, 10.0)
        
    def adapt_parameters(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt PT parameters."""
        adaptations = {}
        
        if "exchange_rates" in current_state:
            self.exchange_rates = current_state["exchange_rates"]
            
            # Adjust temperature spacing based on exchange rates
            if len(self.exchange_rates) > 0:
                avg_exchange_rate = np.mean(self.exchange_rates)
                
                if avg_exchange_rate < 0.2:  # Too few exchanges
                    # Reduce temperature spacing
                    adaptations["temp_spacing_factor"] = 0.9
                elif avg_exchange_rate > 0.8:  # Too many exchanges
                    # Increase temperature spacing
                    adaptations["temp_spacing_factor"] = 1.1
        
        return adaptations
    
    def should_restart(self, current_state: Dict[str, Any]) -> bool:
        """PT rarely needs restart."""
        return False


class PopulationOptimizer(AdaptiveOptimizer):
    """Population-based optimization with adaptive population management."""
    
    def __init__(self, config: AdaptiveConfig, population_size: int = 50):
        super().__init__(config)
        self.population_size = population_size
        self.diversity_threshold = 0.1
        
    def adapt_parameters(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt population parameters."""
        adaptations = {}
        
        # Monitor population diversity
        if "population_diversity" in current_state:
            diversity = current_state["population_diversity"]
            
            if diversity < self.diversity_threshold:
                # Low diversity, increase mutation rate or inject random individuals
                adaptations["mutation_rate"] = min(
                    current_state.get("mutation_rate", 0.1) * 1.2, 0.5
                )
                adaptations["inject_random"] = True
        
        return adaptations
    
    def should_restart(self, current_state: Dict[str, Any]) -> bool:
        """Check for population convergence."""
        if "population_diversity" in current_state:
            return current_state["population_diversity"] < 0.01
        return False


class AutoScalingManager:
    """Automatic scaling for optimization workloads."""
    
    def __init__(self):
        self.resource_monitors: Dict[str, Callable] = {}
        self.scaling_policies: Dict[str, Dict] = {}
        self.current_resources: Dict[str, Any] = {}
        
    def register_resource_monitor(self, name: str, monitor_func: Callable) -> None:
        """Register a resource monitoring function."""
        self.resource_monitors[name] = monitor_func
    
    def set_scaling_policy(
        self,
        resource_name: str,
        scale_up_threshold: float,
        scale_down_threshold: float,
        max_scale: int = 10,
        min_scale: int = 1
    ) -> None:
        """Set scaling policy for a resource."""
        self.scaling_policies[resource_name] = {
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "max_scale": max_scale,
            "min_scale": min_scale,
            "current_scale": 1
        }
    
    def evaluate_scaling(self) -> Dict[str, int]:
        """Evaluate if scaling is needed."""
        scaling_actions = {}
        
        for resource_name, monitor_func in self.resource_monitors.items():
            if resource_name not in self.scaling_policies:
                continue
            
            try:
                current_usage = monitor_func()
                policy = self.scaling_policies[resource_name]
                current_scale = policy["current_scale"]
                
                if current_usage > policy["scale_up_threshold"]:
                    # Scale up
                    new_scale = min(current_scale + 1, policy["max_scale"])
                    if new_scale > current_scale:
                        scaling_actions[resource_name] = new_scale
                        policy["current_scale"] = new_scale
                
                elif current_usage < policy["scale_down_threshold"]:
                    # Scale down
                    new_scale = max(current_scale - 1, policy["min_scale"])
                    if new_scale < current_scale:
                        scaling_actions[resource_name] = new_scale
                        policy["current_scale"] = new_scale
                        
            except Exception as e:
                print(f"Warning: Resource monitoring failed for {resource_name}: {e}")
        
        return scaling_actions


class CacheManager:
    """Intelligent caching for optimization computations."""
    
    def __init__(self, max_cache_size: int = 10000):
        self.max_cache_size = max_cache_size
        self.energy_cache: Dict[str, float] = {}
        self.computation_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def _compute_key(self, spins: torch.Tensor) -> str:
        """Compute cache key for spin configuration."""
        # Use hash of spin configuration
        return str(hash(tuple(spins.cpu().numpy().tolist())))
    
    def get_energy(self, spins: torch.Tensor) -> Optional[float]:
        """Get cached energy for spin configuration."""
        key = self._compute_key(spins)
        
        if key in self.energy_cache:
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.energy_cache[key]
        
        self.miss_count += 1
        return None
    
    def store_energy(self, spins: torch.Tensor, energy: float) -> None:
        """Store energy in cache."""
        key = self._compute_key(spins)
        
        # Manage cache size
        if len(self.energy_cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.energy_cache[key] = energy
        self.access_times[key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Evict least recently used cache entries."""
        if not self.access_times:
            return
        
        # Remove 10% of oldest entries
        n_to_remove = max(1, len(self.access_times) // 10)
        oldest_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])[:n_to_remove]
        
        for key in oldest_keys:
            self.energy_cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.energy_cache),
            "max_size": self.max_cache_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.get_hit_rate(),
            "memory_usage_estimate": len(self.energy_cache) * 8 / (1024 * 1024)  # Rough estimate in MB
        }


class PerformanceProfiler:
    """Profile optimization performance and suggest improvements."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Monitor memory before
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()
                
                result = func(*args, **kwargs)
                
                # Record timing
                duration = time.time() - start_time
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = []
                self.operation_times[operation_name].append(duration)
                
                # Monitor memory after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
                    self.memory_usage.append(memory_used)
                
                return result
            return wrapper
        return decorator
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations."""
        analysis = {
            "operation_timings": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Analyze operation timings
        for op_name, times in self.operation_times.items():
            if times:
                analysis["operation_timings"][op_name] = {
                    "avg_time": np.mean(times),
                    "total_time": np.sum(times),
                    "call_count": len(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times)
                }
        
        # Identify bottlenecks
        if analysis["operation_timings"]:
            sorted_ops = sorted(
                analysis["operation_timings"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )
            
            # Top 3 time consumers are potential bottlenecks
            for op_name, stats in sorted_ops[:3]:
                if stats["total_time"] > 0.1:  # > 100ms total
                    analysis["bottlenecks"].append({
                        "operation": op_name,
                        "total_time": stats["total_time"],
                        "percentage": stats["total_time"] / sum(s["total_time"] for _, s in sorted_ops) * 100
                    })
        
        # Generate recommendations
        if analysis["bottlenecks"]:
            analysis["recommendations"].extend([
                "Consider optimizing the identified bottleneck operations",
                "Use GPU acceleration for computation-intensive operations",
                "Implement caching for repeated calculations"
            ])
        
        if np.mean(self.memory_usage) > 100:  # > 100MB average
            analysis["recommendations"].append("Consider reducing memory usage or using sparse representations")
        
        return analysis


# Global instances
global_cache_manager = CacheManager()
global_performance_profiler = PerformanceProfiler()
global_autoscaling_manager = AutoScalingManager()