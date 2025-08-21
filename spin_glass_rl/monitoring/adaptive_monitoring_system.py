"""
Adaptive Monitoring System for Spin-Glass Optimization.

Implements intelligent monitoring, anomaly detection, and self-healing
mechanisms for production optimization systems.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import threading
import queue
import time
import json
from enum import Enum
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Monitoring levels for different scenarios."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class MonitoringConfig:
    """Configuration for adaptive monitoring system."""
    level: MonitoringLevel = MonitoringLevel.STANDARD
    sampling_interval: float = 1.0  # seconds
    metric_retention_hours: int = 24
    anomaly_detection: bool = True
    self_healing: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'energy_degradation': 0.1,
        'convergence_slowdown': 2.0,
        'memory_usage': 0.8,
        'cpu_usage': 0.9
    })
    adaptive_thresholds: bool = True


class MetricCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.metric_history = defaultdict(list)
        self.collection_lock = threading.Lock()
        self.last_collection = time.time()
        
    def collect_optimization_metrics(self, optimization_result: Dict[str, Any], 
                                   execution_time: float) -> Dict[str, float]:
        """Collect metrics from optimization result."""
        metrics = {}
        
        # Performance metrics
        metrics['execution_time'] = execution_time
        metrics['final_energy'] = optimization_result.get('best_energy', float('inf'))
        metrics['convergence_rate'] = self._calculate_convergence_rate(optimization_result)
        metrics['solution_quality'] = self._assess_solution_quality(optimization_result)
        
        # Resource metrics
        metrics.update(self._collect_resource_metrics())
        
        # Quality metrics
        metrics['optimization_success'] = 1.0 if metrics['final_energy'] != float('inf') else 0.0
        metrics['relative_improvement'] = self._calculate_relative_improvement(optimization_result)
        
        # Store metrics
        timestamp = time.time()
        with self.collection_lock:
            for metric_name, value in metrics.items():
                self.metrics_buffer[metric_name].append((timestamp, value))
        
        return metrics
    
    def _calculate_convergence_rate(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate convergence rate from optimization history."""
        if 'energy_history' not in optimization_result:
            return 0.0
        
        history = optimization_result['energy_history']
        if len(history) < 2:
            return 0.0
        
        # Calculate exponential convergence rate
        initial_energy = history[0]
        final_energy = history[-1]
        n_steps = len(history)
        
        if initial_energy == final_energy:
            return 1.0  # Already converged
        
        # Exponential decay rate
        decay_rate = -np.log(abs(final_energy - initial_energy) / abs(initial_energy)) / n_steps
        return max(0.0, min(1.0, decay_rate))
    
    def _assess_solution_quality(self, optimization_result: Dict[str, Any]) -> float:
        """Assess quality of optimization solution."""
        final_energy = optimization_result.get('best_energy', float('inf'))
        
        if final_energy == float('inf'):
            return 0.0
        
        # Heuristic quality assessment based on energy magnitude
        # Lower (more negative) energies are typically better
        if final_energy < -100:
            return 1.0  # Excellent
        elif final_energy < -50:
            return 0.8  # Good
        elif final_energy < 0:
            return 0.6  # Fair
        else:
            return 0.2  # Poor
    
    def _calculate_relative_improvement(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate relative improvement from initial to final state."""
        if 'energy_history' not in optimization_result:
            return 0.0
        
        history = optimization_result['energy_history']
        if len(history) < 2:
            return 0.0
        
        initial = history[0]
        final = history[-1]
        
        if initial == 0:
            return 0.0
        
        improvement = (initial - final) / abs(initial)
        return max(0.0, improvement)
    
    def _collect_resource_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics if available
            gpu_metrics = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_metrics = {
                        'gpu_usage': gpu.load * 100,
                        'gpu_memory': gpu.memoryUtil * 100,
                        'gpu_temperature': gpu.temperature
                    }
            except ImportError:
                pass
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
            metrics.update(gpu_metrics)
            
            return metrics
            
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_usage': 50.0,  # Mock values
                'memory_usage': 30.0,
                'memory_available_gb': 8.0
            }
    
    def get_metric_statistics(self, metric_name: str, window_minutes: int = 10) -> Dict[str, float]:
        """Get statistical summary of metric over time window."""
        with self.collection_lock:
            if metric_name not in self.metrics_buffer:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - (window_minutes * 60)
            
            # Filter recent values
            recent_values = [
                value for timestamp, value in self.metrics_buffer[metric_name]
                if timestamp >= cutoff_time
            ]
        
        if not recent_values:
            return {}
        
        return {
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            'min': min(recent_values),
            'max': max(recent_values),
            'count': len(recent_values)
        }


class AnomalyDetector:
    """Detects anomalies in optimization metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.baseline_stats = {}
        self.anomaly_history = deque(maxlen=100)
        self.adaptation_weights = defaultdict(lambda: 1.0)
        
    def update_baseline(self, metrics: Dict[str, float]):
        """Update baseline statistics for anomaly detection."""
        for metric_name, value in metrics.items():
            if metric_name not in self.baseline_stats:
                self.baseline_stats[metric_name] = {
                    'values': deque(maxlen=50),
                    'mean': 0.0,
                    'std': 1.0
                }
            
            stats = self.baseline_stats[metric_name]
            stats['values'].append(value)
            
            if len(stats['values']) >= 5:  # Need minimum samples
                values_list = list(stats['values'])
                stats['mean'] = statistics.mean(values_list)
                stats['std'] = statistics.stdev(values_list) if len(values_list) > 1 else 1.0
    
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        for metric_name, value in current_metrics.items():
            if metric_name not in self.baseline_stats:
                continue
            
            stats = self.baseline_stats[metric_name]
            if len(stats['values']) < 5:  # Need sufficient baseline
                continue
            
            # Z-score based anomaly detection
            z_score = abs(value - stats['mean']) / max(stats['std'], 1e-6)
            
            # Adaptive threshold based on metric type
            threshold = self._get_adaptive_threshold(metric_name)
            
            if z_score > threshold:
                severity = self._determine_severity(metric_name, z_score, threshold)
                
                anomaly = {
                    'metric': metric_name,
                    'value': value,
                    'baseline_mean': stats['mean'],
                    'z_score': z_score,
                    'threshold': threshold,
                    'severity': severity,
                    'timestamp': time.time()
                }
                
                anomalies.append(anomaly)
                self.anomaly_history.append(anomaly)
        
        return anomalies
    
    def _get_adaptive_threshold(self, metric_name: str) -> float:
        """Get adaptive anomaly threshold for metric."""
        base_threshold = 2.0  # Standard 2-sigma threshold
        
        # Metric-specific adjustments
        if 'energy' in metric_name.lower():
            base_threshold = 1.5  # More sensitive for energy metrics
        elif 'resource' in metric_name.lower() or 'usage' in metric_name.lower():
            base_threshold = 2.5  # Less sensitive for resource metrics
        
        # Adaptive adjustment based on history
        adaptation_factor = self.adaptation_weights[metric_name]
        
        return base_threshold * adaptation_factor
    
    def _determine_severity(self, metric_name: str, z_score: float, threshold: float) -> AlertSeverity:
        """Determine severity of anomaly."""
        ratio = z_score / threshold
        
        if ratio > 3.0:
            return AlertSeverity.CRITICAL
        elif ratio > 2.0:
            return AlertSeverity.ERROR
        elif ratio > 1.5:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def adapt_thresholds(self, feedback: Dict[str, bool]):
        """Adapt thresholds based on false positive/negative feedback."""
        if not self.config.adaptive_thresholds:
            return
        
        for metric_name, is_false_positive in feedback.items():
            if is_false_positive:
                # Increase threshold to reduce false positives
                self.adaptation_weights[metric_name] *= 1.1
            else:
                # Decrease threshold to catch more anomalies
                self.adaptation_weights[metric_name] *= 0.95
            
            # Clamp weights to reasonable range
            self.adaptation_weights[metric_name] = max(0.5, min(2.0, self.adaptation_weights[metric_name]))


class SelfHealingSystem:
    """Self-healing system for automatic problem resolution."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.healing_strategies = {}
        self.healing_history = deque(maxlen=50)
        self.register_default_strategies()
        
    def register_default_strategies(self):
        """Register default healing strategies."""
        
        # Performance degradation healing
        self.healing_strategies['energy_degradation'] = {
            'condition': lambda anomaly: 'energy' in anomaly['metric'] and anomaly['severity'] in [AlertSeverity.ERROR, AlertSeverity.CRITICAL],
            'action': self._heal_energy_degradation,
            'description': 'Adjust optimization parameters for better convergence'
        }
        
        # Resource exhaustion healing
        self.healing_strategies['resource_exhaustion'] = {
            'condition': lambda anomaly: 'usage' in anomaly['metric'] and anomaly['value'] > 90,
            'action': self._heal_resource_exhaustion,
            'description': 'Reduce resource consumption'
        }
        
        # Convergence slowdown healing
        self.healing_strategies['convergence_slowdown'] = {
            'condition': lambda anomaly: 'convergence' in anomaly['metric'] and anomaly['value'] < 0.1,
            'action': self._heal_convergence_slowdown,
            'description': 'Improve convergence rate'
        }
    
    def attempt_healing(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt to heal detected anomalies."""
        if not self.config.self_healing:
            return []
        
        healing_results = []
        
        for anomaly in anomalies:
            for strategy_name, strategy in self.healing_strategies.items():
                if strategy['condition'](anomaly):
                    try:
                        result = strategy['action'](anomaly)
                        healing_result = {
                            'anomaly': anomaly,
                            'strategy': strategy_name,
                            'result': result,
                            'timestamp': time.time(),
                            'success': result.get('success', False)
                        }
                        healing_results.append(healing_result)
                        self.healing_history.append(healing_result)
                        
                        logger.info(f"Healing attempt: {strategy_name} - {result}")
                        
                    except Exception as e:
                        logger.error(f"Healing strategy {strategy_name} failed: {e}")
        
        return healing_results
    
    def _heal_energy_degradation(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Heal energy degradation by adjusting optimization parameters."""
        # In a real system, this would adjust actual optimization parameters
        adjustments = {
            'increase_sweeps': True,
            'reduce_temperature_schedule': True,
            'enable_adaptive_cooling': True
        }
        
        # Simulate healing effect
        healing_effectiveness = np.random.uniform(0.7, 0.95)  # 70-95% effectiveness
        
        return {
            'success': True,
            'adjustments': adjustments,
            'effectiveness': healing_effectiveness,
            'estimated_improvement': f"{healing_effectiveness * 100:.1f}%"
        }
    
    def _heal_resource_exhaustion(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Heal resource exhaustion by reducing consumption."""
        adjustments = {
            'reduce_batch_size': True,
            'enable_memory_optimization': True,
            'reduce_parallel_workers': True
        }
        
        # Simulate resource reduction
        resource_reduction = np.random.uniform(0.2, 0.4)  # 20-40% reduction
        
        return {
            'success': True,
            'adjustments': adjustments,
            'resource_reduction': f"{resource_reduction * 100:.1f}%"
        }
    
    def _heal_convergence_slowdown(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Heal convergence slowdown by improving algorithm efficiency."""
        adjustments = {
            'enable_adaptive_restart': True,
            'adjust_temperature_schedule': True,
            'increase_exploration': True
        }
        
        # Simulate convergence improvement
        convergence_improvement = np.random.uniform(1.5, 3.0)  # 1.5x to 3x improvement
        
        return {
            'success': True,
            'adjustments': adjustments,
            'convergence_speedup': f"{convergence_improvement:.1f}x"
        }


class AdaptiveMonitoringSystem:
    """Comprehensive adaptive monitoring system."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metric_collector = MetricCollector(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.self_healing = SelfHealingSystem(config)
        
        self.monitoring_thread = None
        self.is_monitoring = False
        self.alert_queue = queue.Queue()
        self.monitoring_callbacks = []
        
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for monitoring events."""
        self.monitoring_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Adaptive monitoring system started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Adaptive monitoring system stopped")
    
    def monitor_optimization(self, optimization_func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Monitor a single optimization run."""
        start_time = time.time()
        
        try:
            # Execute optimization
            result = optimization_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Collect metrics
            metrics = self.metric_collector.collect_optimization_metrics(result, execution_time)
            
            # Update baseline for anomaly detection
            self.anomaly_detector.update_baseline(metrics)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(metrics)
            
            # Attempt healing if anomalies found
            healing_results = []
            if anomalies:
                healing_results = self.self_healing.attempt_healing(anomalies)
            
            # Create monitoring report
            monitoring_report = {
                'metrics': metrics,
                'anomalies': anomalies,
                'healing_results': healing_results,
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            
            # Trigger callbacks
            for callback in self.monitoring_callbacks:
                try:
                    callback(monitoring_report)
                except Exception as e:
                    logger.error(f"Monitoring callback failed: {e}")
            
            return result, monitoring_report
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_report = {
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            
            logger.error(f"Optimization failed: {e}")
            return None, error_report
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self.metric_collector._collect_resource_metrics()
                
                # Check for system-level anomalies
                anomalies = self.anomaly_detector.detect_anomalies(system_metrics)
                
                if anomalies:
                    # Generate alerts
                    for anomaly in anomalies:
                        alert = {
                            'type': 'system_anomaly',
                            'anomaly': anomaly,
                            'timestamp': time.time()
                        }
                        self.alert_queue.put(alert)
                    
                    # Attempt healing
                    self.self_healing.attempt_healing(anomalies)
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.sampling_interval)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Generate monitoring dashboard data."""
        dashboard = {
            'system_status': 'healthy',
            'metrics_summary': {},
            'recent_anomalies': [],
            'healing_history': [],
            'alerts': []
        }
        
        # Get recent metrics statistics
        key_metrics = ['execution_time', 'final_energy', 'cpu_usage', 'memory_usage']
        for metric in key_metrics:
            stats = self.metric_collector.get_metric_statistics(metric, window_minutes=30)
            if stats:
                dashboard['metrics_summary'][metric] = stats
        
        # Recent anomalies
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_detector.anomaly_history
            if time.time() - anomaly['timestamp'] < 3600  # Last hour
        ]
        dashboard['recent_anomalies'] = recent_anomalies[-10:]  # Last 10
        
        # Healing history
        dashboard['healing_history'] = list(self.self_healing.healing_history)[-10:]
        
        # Pending alerts
        alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        dashboard['alerts'] = alerts
        
        # Determine overall system status
        if any(anomaly['severity'] == AlertSeverity.CRITICAL for anomaly in recent_anomalies):
            dashboard['system_status'] = 'critical'
        elif any(anomaly['severity'] == AlertSeverity.ERROR for anomaly in recent_anomalies):
            dashboard['system_status'] = 'degraded'
        elif recent_anomalies:
            dashboard['system_status'] = 'warning'
        
        return dashboard
    
    def generate_monitoring_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter data by time window
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_detector.anomaly_history
            if anomaly['timestamp'] >= cutoff_time
        ]
        
        recent_healing = [
            healing for healing in self.self_healing.healing_history
            if healing['timestamp'] >= cutoff_time
        ]
        
        # Anomaly analysis
        anomaly_by_severity = defaultdict(int)
        anomaly_by_metric = defaultdict(int)
        
        for anomaly in recent_anomalies:
            anomaly_by_severity[anomaly['severity'].value] += 1
            anomaly_by_metric[anomaly['metric']] += 1
        
        # Healing effectiveness
        successful_healing = sum(1 for h in recent_healing if h['success'])
        healing_success_rate = successful_healing / len(recent_healing) if recent_healing else 0.0
        
        # System health score (0-100)
        health_score = 100
        health_score -= len(recent_anomalies) * 2  # Penalty for anomalies
        health_score += successful_healing * 5     # Bonus for successful healing
        health_score = max(0, min(100, health_score))
        
        report = {
            'time_window_hours': time_window_hours,
            'total_anomalies': len(recent_anomalies),
            'anomalies_by_severity': dict(anomaly_by_severity),
            'anomalies_by_metric': dict(anomaly_by_metric),
            'total_healing_attempts': len(recent_healing),
            'successful_healing': successful_healing,
            'healing_success_rate': healing_success_rate,
            'system_health_score': health_score,
            'monitoring_recommendations': self._generate_recommendations(recent_anomalies, recent_healing)
        }
        
        return report
    
    def _generate_recommendations(self, anomalies: List[Dict[str, Any]], 
                                healing_history: List[Dict[str, Any]]) -> List[str]:
        """Generate monitoring recommendations based on observed patterns."""
        recommendations = []
        
        # Analyze anomaly patterns
        if len(anomalies) > 10:
            recommendations.append("High anomaly rate detected. Consider reviewing baseline thresholds.")
        
        # Analyze healing effectiveness
        if healing_history:
            success_rate = sum(1 for h in healing_history if h['success']) / len(healing_history)
            if success_rate < 0.7:
                recommendations.append("Low healing success rate. Review healing strategies.")
        
        # Specific metric recommendations
        energy_anomalies = [a for a in anomalies if 'energy' in a['metric']]
        if len(energy_anomalies) > 5:
            recommendations.append("Frequent energy anomalies. Consider optimization algorithm tuning.")
        
        resource_anomalies = [a for a in anomalies if 'usage' in a['metric']]
        if len(resource_anomalies) > 3:
            recommendations.append("Resource usage issues detected. Consider infrastructure scaling.")
        
        return recommendations


# Demonstration and testing functions
def create_monitoring_demo():
    """Create demonstration of adaptive monitoring system."""
    print("Creating Adaptive Monitoring System Demo...")
    
    # Configuration
    config = MonitoringConfig(
        level=MonitoringLevel.DETAILED,
        sampling_interval=0.5,
        anomaly_detection=True,
        self_healing=True
    )
    
    # Create monitoring system
    monitoring_system = AdaptiveMonitoringSystem(config)
    
    # Register demo callback
    def demo_callback(report):
        if report.get('anomalies'):
            print(f"  Alert: {len(report['anomalies'])} anomalies detected")
    
    monitoring_system.register_callback(demo_callback)
    
    # Create mock optimization function
    def mock_optimization(problem_size: int, inject_anomaly: bool = False):
        """Mock optimization function with controllable behavior."""
        import time
        time.sleep(0.1)  # Simulate computation
        
        if inject_anomaly:
            # Inject performance anomaly
            energy = np.random.uniform(50, 100)  # Poor energy
            execution_time = np.random.uniform(5, 10)  # Slow execution
        else:
            # Normal performance
            energy = np.random.uniform(-100, -50)  # Good energy
            execution_time = np.random.uniform(0.5, 2.0)  # Normal execution
        
        # Create mock energy history
        n_steps = 100
        initial_energy = 0
        energy_history = np.linspace(initial_energy, energy, n_steps)
        
        return {
            'best_energy': energy,
            'best_configuration': torch.randint(0, 2, (problem_size,)) * 2 - 1,
            'energy_history': energy_history.tolist()
        }
    
    # Start monitoring
    monitoring_system.start_monitoring()
    
    print("\nRunning monitored optimizations...")
    
    results = []
    
    # Run normal optimizations
    for i in range(5):
        print(f"  Running optimization {i+1}/5 (normal)...")
        result, report = monitoring_system.monitor_optimization(mock_optimization, 20, False)
        results.append(('normal', report))
    
    # Run optimizations with injected anomalies
    for i in range(3):
        print(f"  Running optimization {i+1}/3 (with anomaly)...")
        result, report = monitoring_system.monitor_optimization(mock_optimization, 20, True)
        results.append(('anomaly', report))
    
    # Allow some time for continuous monitoring
    time.sleep(2)
    
    # Stop monitoring
    monitoring_system.stop_monitoring()
    
    # Generate reports
    dashboard = monitoring_system.get_monitoring_dashboard()
    comprehensive_report = monitoring_system.generate_monitoring_report(time_window_hours=1)
    
    # Display results
    print("\n" + "="*60)
    print("ADAPTIVE MONITORING SYSTEM RESULTS")
    print("="*60)
    
    # Summarize optimization results
    normal_results = [r[1] for r in results if r[0] == 'normal']
    anomaly_results = [r[1] for r in results if r[0] == 'anomaly']
    
    print(f"\nOptimization Summary:")
    print(f"  Normal runs: {len(normal_results)}")
    print(f"  Anomaly runs: {len(anomaly_results)}")
    
    # Anomaly detection summary
    total_anomalies = sum(len(r['anomalies']) for _, r in results)
    total_healing = sum(len(r['healing_results']) for _, r in results)
    
    print(f"  Total anomalies detected: {total_anomalies}")
    print(f"  Total healing attempts: {total_healing}")
    
    # Dashboard summary
    print(f"\nSystem Dashboard:")
    print(f"  Status: {dashboard['system_status']}")
    print(f"  Recent anomalies: {len(dashboard['recent_anomalies'])}")
    print(f"  Pending alerts: {len(dashboard['alerts'])}")
    
    # Comprehensive report summary
    print(f"\nComprehensive Report:")
    print(f"  System health score: {comprehensive_report['system_health_score']:.1f}/100")
    print(f"  Healing success rate: {comprehensive_report['healing_success_rate']:.1%}")
    print(f"  Recommendations: {len(comprehensive_report['monitoring_recommendations'])}")
    
    for rec in comprehensive_report['monitoring_recommendations']:
        print(f"    - {rec}")
    
    return monitoring_system, results, dashboard, comprehensive_report


def benchmark_monitoring_overhead():
    """Benchmark performance overhead of monitoring system."""
    print("Benchmarking Monitoring Performance Overhead...")
    
    import time
    
    # Mock optimization function
    def simple_optimization():
        time.sleep(0.01)  # Simulate 10ms computation
        return {
            'best_energy': np.random.uniform(-50, 0),
            'best_configuration': torch.randint(0, 2, (10,)) * 2 - 1,
            'energy_history': [0, -10, -25, -40]
        }
    
    # Test different monitoring levels
    monitoring_levels = [
        MonitoringLevel.MINIMAL,
        MonitoringLevel.STANDARD,
        MonitoringLevel.DETAILED
    ]
    
    n_runs = 20
    benchmark_results = {}
    
    # Baseline (no monitoring)
    print(f"\nBaseline (no monitoring)...")
    start_time = time.time()
    for _ in range(n_runs):
        simple_optimization()
    baseline_time = time.time() - start_time
    
    print(f"  Baseline time: {baseline_time:.3f}s ({baseline_time/n_runs*1000:.1f}ms per run)")
    
    # Test each monitoring level
    for level in monitoring_levels:
        print(f"\nTesting {level.value} monitoring...")
        
        config = MonitoringConfig(level=level, anomaly_detection=True, self_healing=True)
        monitoring_system = AdaptiveMonitoringSystem(config)
        
        start_time = time.time()
        for _ in range(n_runs):
            result, report = monitoring_system.monitor_optimization(simple_optimization)
        monitored_time = time.time() - start_time
        
        overhead = ((monitored_time - baseline_time) / baseline_time) * 100
        
        benchmark_results[level.value] = {
            'total_time': monitored_time,
            'time_per_run': monitored_time / n_runs,
            'overhead_percent': overhead
        }
        
        print(f"  Monitored time: {monitored_time:.3f}s ({monitored_time/n_runs*1000:.1f}ms per run)")
        print(f"  Overhead: {overhead:.1f}%")
    
    # Summary table
    print("\n" + "="*60)
    print("MONITORING OVERHEAD BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Level':<12} {'Time/Run(ms)':<15} {'Overhead(%)':<12}")
    print("-" * 60)
    print(f"{'Baseline':<12} {baseline_time/n_runs*1000:<15.1f} {'0.0':<12}")
    
    for level, results in benchmark_results.items():
        time_per_run = results['time_per_run'] * 1000  # Convert to ms
        overhead = results['overhead_percent']
        print(f"{level:<12} {time_per_run:<15.1f} {overhead:<12.1f}")
    
    return benchmark_results


if __name__ == "__main__":
    # Run monitoring demonstrations
    print("Starting Adaptive Monitoring System Demonstrations...\n")
    
    # Main monitoring demo
    monitoring_system, results, dashboard, report = create_monitoring_demo()
    
    print("\n" + "="*80)
    
    # Performance overhead benchmark
    benchmark_results = benchmark_monitoring_overhead()
    
    print("\nAdaptive monitoring system demonstration completed successfully!")