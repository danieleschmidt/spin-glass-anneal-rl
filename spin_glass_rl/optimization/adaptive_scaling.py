"""Adaptive scaling and auto-tuning system for optimization parameters."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import torch

from spin_glass_rl.utils.robust_logging import get_logger
from spin_glass_rl.utils.monitoring import PerformanceMonitor

logger = get_logger("adaptive_scaling")


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float
    energy: float
    acceptance_rate: float
    sweep_time_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    convergence_rate: float
    problem_size: int
    temperature: float


@dataclass
class ScalingPolicy:
    """Configuration for adaptive scaling behavior."""
    # Performance thresholds
    min_acceptance_rate: float = 0.1
    max_acceptance_rate: float = 0.9
    target_sweep_time_ms: float = 100.0
    max_memory_usage_mb: float = 8000.0
    target_gpu_utilization: float = 0.8
    
    # Scaling parameters
    temperature_adjustment_factor: float = 1.1
    batch_size_adjustment_factor: float = 1.2
    memory_scaling_threshold: float = 0.9
    
    # Convergence criteria
    convergence_patience: int = 100
    min_improvement_threshold: float = 1e-6
    
    # Auto-tuning parameters
    enable_temperature_tuning: bool = True
    enable_batch_size_tuning: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True


class AdaptiveScaler:
    """
    Adaptive scaling system that automatically adjusts optimization parameters
    based on real-time performance metrics and resource utilization.
    """
    
    def __init__(self, 
                 policy: Optional[ScalingPolicy] = None,
                 history_size: int = 1000,
                 adjustment_interval: int = 50):
        
        self.policy = policy or ScalingPolicy()
        self.history_size = history_size
        self.adjustment_interval = adjustment_interval
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_size)
        self.adjustments_history: List[Dict[str, Any]] = []
        
        # Current parameters
        self.current_params = {
            'temperature': 1.0,
            'batch_size': 1,
            'n_parallel_chains': 1,
            'memory_limit_mb': 4000,
            'use_mixed_precision': False
        }
        
        # Performance tracking
        self.performance_tracker = {
            'best_energy': float('inf'),
            'no_improvement_count': 0,
            'total_adjustments': 0,
            'successful_adjustments': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("Adaptive scaler initialized", policy=policy.__dict__ if policy else "default")
    
    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record performance metrics for scaling decisions."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Update performance tracking
            if metrics.energy < self.performance_tracker['best_energy']:
                improvement = self.performance_tracker['best_energy'] - metrics.energy
                self.performance_tracker['best_energy'] = metrics.energy
                self.performance_tracker['no_improvement_count'] = 0
                
                logger.debug(f"New best energy: {metrics.energy:.6f} (improvement: {improvement:.6f})")
            else:
                self.performance_tracker['no_improvement_count'] += 1
    
    def should_adjust_parameters(self) -> bool:
        """Check if parameters should be adjusted."""
        if len(self.metrics_history) < self.adjustment_interval:
            return False
        
        # Check if enough time has passed since last adjustment
        if self.adjustments_history:
            last_adjustment_time = self.adjustments_history[-1]['timestamp']
            if time.time() - last_adjustment_time < 10.0:  # Min 10 seconds between adjustments
                return False
        
        return True
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        if len(self.metrics_history) < 10:
            return {'insufficient_data': True}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
        
        # Calculate trends
        energies = [m.energy for m in recent_metrics]
        acceptance_rates = [m.acceptance_rate for m in recent_metrics]
        sweep_times = [m.sweep_time_ms for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        gpu_utilization = [m.gpu_utilization for m in recent_metrics]
        
        analysis = {
            'energy_trend': np.polyfit(range(len(energies)), energies, 1)[0],  # Slope
            'mean_acceptance_rate': np.mean(acceptance_rates),
            'mean_sweep_time_ms': np.mean(sweep_times),
            'max_memory_usage_mb': np.max(memory_usage),
            'mean_gpu_utilization': np.mean(gpu_utilization),
            'energy_variance': np.var(energies),
            'convergence_stalled': self.performance_tracker['no_improvement_count'] > self.policy.convergence_patience
        }
        
        return analysis
    
    def generate_parameter_adjustments(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter adjustment recommendations."""
        adjustments = {}
        
        if analysis.get('insufficient_data'):
            return adjustments
        
        # Temperature adjustments
        if self.policy.enable_temperature_tuning:
            acceptance_rate = analysis['mean_acceptance_rate']
            
            if acceptance_rate < self.policy.min_acceptance_rate:
                # Too few acceptances, increase temperature
                new_temp = self.current_params['temperature'] * self.policy.temperature_adjustment_factor
                adjustments['temperature'] = new_temp
                adjustments['temperature_reason'] = f"Low acceptance rate: {acceptance_rate:.3f}"
                
            elif acceptance_rate > self.policy.max_acceptance_rate:
                # Too many acceptances, decrease temperature
                new_temp = self.current_params['temperature'] / self.policy.temperature_adjustment_factor
                adjustments['temperature'] = max(new_temp, 0.001)  # Don't go below minimum
                adjustments['temperature_reason'] = f"High acceptance rate: {acceptance_rate:.3f}"
        
        # Batch size adjustments
        if self.policy.enable_batch_size_tuning:
            sweep_time = analysis['mean_sweep_time_ms']
            
            if sweep_time < self.policy.target_sweep_time_ms * 0.5:
                # Sweeps too fast, can increase batch size
                new_batch = int(self.current_params['batch_size'] * self.policy.batch_size_adjustment_factor)
                adjustments['batch_size'] = min(new_batch, 64)  # Reasonable upper limit
                adjustments['batch_size_reason'] = f"Fast sweeps: {sweep_time:.1f}ms"
                
            elif sweep_time > self.policy.target_sweep_time_ms * 2.0:
                # Sweeps too slow, decrease batch size
                new_batch = max(1, int(self.current_params['batch_size'] / self.policy.batch_size_adjustment_factor))
                adjustments['batch_size'] = new_batch
                adjustments['batch_size_reason'] = f"Slow sweeps: {sweep_time:.1f}ms"
        
        # Memory optimization
        if self.policy.enable_memory_optimization:
            max_memory = analysis['max_memory_usage_mb']
            
            if max_memory > self.policy.max_memory_usage_mb * self.policy.memory_scaling_threshold:
                # Memory usage too high
                if not self.current_params['use_mixed_precision']:
                    adjustments['use_mixed_precision'] = True
                    adjustments['mixed_precision_reason'] = f"High memory: {max_memory:.0f}MB"
                else:
                    # Already using mixed precision, reduce parallel chains
                    new_chains = max(1, self.current_params['n_parallel_chains'] - 1)
                    adjustments['n_parallel_chains'] = new_chains
                    adjustments['parallel_chains_reason'] = f"High memory: {max_memory:.0f}MB"
        
        # GPU utilization optimization
        if self.policy.enable_gpu_optimization and torch.cuda.is_available():
            gpu_util = analysis['mean_gpu_utilization']
            
            if gpu_util < self.policy.target_gpu_utilization * 0.7:
                # Low GPU utilization, can increase parallel work
                new_chains = min(8, self.current_params['n_parallel_chains'] + 1)
                adjustments['n_parallel_chains'] = new_chains
                adjustments['gpu_utilization_reason'] = f"Low GPU util: {gpu_util:.3f}"
        
        # Convergence-based adjustments
        if analysis['convergence_stalled']:
            # Try different strategies to escape local minima
            if np.random.random() < 0.5:
                # Increase temperature for exploration
                new_temp = self.current_params['temperature'] * 1.5
                adjustments['temperature'] = new_temp
                adjustments['convergence_reason'] = "Stalled convergence - increase exploration"
            else:
                # Restart with different initialization
                adjustments['restart_optimization'] = True
                adjustments['convergence_reason'] = "Stalled convergence - restart"
        
        return adjustments
    
    def apply_adjustments(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter adjustments and return results."""
        if not adjustments:
            return {'applied': False, 'reason': 'No adjustments needed'}
        
        applied_changes = {}
        
        with self._lock:
            for param, new_value in adjustments.items():
                if param.endswith('_reason') or param == 'restart_optimization':
                    continue
                    
                if param in self.current_params:
                    old_value = self.current_params[param]
                    self.current_params[param] = new_value
                    applied_changes[param] = {'old': old_value, 'new': new_value}
            
            # Record the adjustment
            adjustment_record = {
                'timestamp': time.time(),
                'adjustments': adjustments,
                'applied_changes': applied_changes,
                'performance_state': self.performance_tracker.copy()
            }
            
            self.adjustments_history.append(adjustment_record)
            self.performance_tracker['total_adjustments'] += 1
        
        logger.info("Applied parameter adjustments", adjustments=applied_changes)
        return {'applied': True, 'changes': applied_changes}
    
    def auto_tune(self) -> Optional[Dict[str, Any]]:
        """Perform automatic parameter tuning based on current metrics."""
        if not self.should_adjust_parameters():
            return None
        
        try:
            # Analyze recent performance
            analysis = self.analyze_performance_trends()
            
            # Generate adjustments
            adjustments = self.generate_parameter_adjustments(analysis)
            
            if not adjustments:
                logger.debug("No parameter adjustments needed")
                return None
            
            # Apply adjustments
            result = self.apply_adjustments(adjustments)
            
            if result['applied']:
                self.performance_tracker['successful_adjustments'] += 1
            
            return result
            
        except Exception as e:
            logger.error("Error during auto-tuning", exception=e)
            return None
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        with self._lock:
            return self.current_params.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = {
                'current_params': self.current_params.copy(),
                'performance_tracker': self.performance_tracker.copy(),
                'total_adjustments': len(self.adjustments_history),
                'metrics_count': len(self.metrics_history)
            }
            
            if self.metrics_history:
                recent = self.metrics_history[-1]
                stats['latest_metrics'] = {
                    'energy': recent.energy,
                    'acceptance_rate': recent.acceptance_rate,
                    'sweep_time_ms': recent.sweep_time_ms,
                    'memory_usage_mb': recent.memory_usage_mb
                }
            
            return stats
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking counters."""
        with self._lock:
            self.performance_tracker = {
                'best_energy': float('inf'),
                'no_improvement_count': 0,
                'total_adjustments': 0,
                'successful_adjustments': 0
            }
            logger.info("Performance tracking reset")
    
    def set_policy(self, policy: ScalingPolicy) -> None:
        """Update scaling policy."""
        with self._lock:
            self.policy = policy
            logger.info("Scaling policy updated", new_policy=policy.__dict__)
    
    def export_tuning_history(self, filepath: str) -> None:
        """Export tuning history for analysis."""
        import json
        
        export_data = {
            'policy': self.policy.__dict__,
            'current_params': self.current_params,
            'performance_tracker': self.performance_tracker,
            'adjustments_history': self.adjustments_history,
            'metrics_summary': {
                'total_metrics': len(self.metrics_history),
                'avg_energy': np.mean([m.energy for m in self.metrics_history]) if self.metrics_history else 0,
                'avg_acceptance_rate': np.mean([m.acceptance_rate for m in self.metrics_history]) if self.metrics_history else 0,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Tuning history exported to {filepath}")


class MultiObjectiveScaler(AdaptiveScaler):
    """Extended scaler for multi-objective optimization."""
    
    def __init__(self, 
                 objectives: List[str],
                 objective_weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.objectives = objectives
        self.objective_weights = objective_weights or {obj: 1.0 for obj in objectives}
        
        # Multi-objective specific tracking
        self.pareto_front = []
        self.objective_history = defaultdict(list)
        
        logger.info("Multi-objective scaler initialized", objectives=objectives, weights=self.objective_weights)
    
    def record_multi_objective_metrics(self, objective_values: Dict[str, float], metrics: ScalingMetrics) -> None:
        """Record metrics for multi-objective optimization."""
        self.record_metrics(metrics)
        
        # Update objective history
        for obj, value in objective_values.items():
            self.objective_history[obj].append(value)
        
        # Update Pareto front
        self._update_pareto_front(objective_values)
    
    def _update_pareto_front(self, objective_values: Dict[str, float]) -> None:
        """Update Pareto front with new solution."""
        solution = objective_values.copy()
        solution['timestamp'] = time.time()
        
        # Check if solution is dominated by existing solutions
        is_dominated = False
        new_pareto_front = []
        
        for pareto_solution in self.pareto_front:
            if self._dominates(pareto_solution, solution):
                is_dominated = True
                new_pareto_front.append(pareto_solution)
            elif not self._dominates(solution, pareto_solution):
                new_pareto_front.append(pareto_solution)
        
        if not is_dominated:
            new_pareto_front.append(solution)
        
        self.pareto_front = new_pareto_front
        
        logger.debug(f"Pareto front updated, size: {len(self.pareto_front)}")
    
    def _dominates(self, solution_a: Dict[str, float], solution_b: Dict[str, float]) -> bool:
        """Check if solution_a dominates solution_b."""
        all_better_or_equal = True
        at_least_one_better = False
        
        for obj in self.objectives:
            a_value = solution_a.get(obj, float('inf'))
            b_value = solution_b.get(obj, float('inf'))
            
            # Assuming minimization for all objectives
            if a_value > b_value:
                all_better_or_equal = False
                break
            elif a_value < b_value:
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def get_pareto_front(self) -> List[Dict[str, float]]:
        """Get current Pareto front."""
        return self.pareto_front.copy()


# Context manager for automatic scaling
class AutoScalingContext:
    """Context manager for automatic parameter scaling during optimization."""
    
    def __init__(self, 
                 scaler: AdaptiveScaler,
                 monitor: Optional[PerformanceMonitor] = None,
                 tuning_interval: int = 100):
        self.scaler = scaler
        self.monitor = monitor
        self.tuning_interval = tuning_interval
        self.step_count = 0
        
    def __enter__(self):
        logger.info("Auto-scaling context started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.info("Auto-scaling context completed successfully")
        else:
            logger.error("Auto-scaling context exited with error", exception=exc_val)
    
    def step(self, energy: float, acceptance_rate: float, temperature: float) -> Optional[Dict[str, Any]]:
        """Record step and potentially trigger parameter adjustment."""
        self.step_count += 1
        
        # Create metrics from available data
        metrics = ScalingMetrics(
            timestamp=time.time(),
            energy=energy,
            acceptance_rate=acceptance_rate,
            sweep_time_ms=1.0,  # Default if not measured
            memory_usage_mb=0.0,  # Default if not measured
            gpu_utilization=0.0,  # Default if not measured
            convergence_rate=0.0,
            problem_size=1000,  # Default
            temperature=temperature
        )
        
        # Add monitor data if available
        if self.monitor:
            current_metrics = self.monitor.get_current_metrics()
            if 'system' in current_metrics:
                sys_metrics = current_metrics['system']
                metrics.memory_usage_mb = sys_metrics.get('memory_available_gb', 0) * 1024
                metrics.gpu_utilization = sys_metrics.get('gpu_utilization', 0) / 100
        
        # Record metrics
        self.scaler.record_metrics(metrics)
        
        # Check if tuning should be performed
        if self.step_count % self.tuning_interval == 0:
            return self.scaler.auto_tune()
        
        return None
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return self.scaler.get_current_parameters()