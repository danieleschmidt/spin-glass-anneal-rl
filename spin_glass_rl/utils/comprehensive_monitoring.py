"""Comprehensive monitoring and alerting system for production deployment."""

import time
import threading
import psutil
import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path
import warnings

from spin_glass_rl.utils.robust_logging import get_logger
from spin_glass_rl.utils.health_checks import HealthStatus

logger = get_logger("monitoring")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class Alert:
    """System alert."""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolve_timestamp: Optional[float] = None


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float
    gpu_memory: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


class SystemMonitor:
    """Comprehensive system monitoring with alerting."""
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 history_size: int = 1000):
        
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Default alert thresholds
        self.thresholds = alert_thresholds or {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_usage': 95.0,
            'gpu_usage': 90.0,
            'gpu_memory': 95.0,
            'temperature': 80.0,  # Celsius
            'network_errors': 10.0,  # errors per minute
            'disk_errors': 5.0   # errors per minute
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Component-specific monitors
        self.component_monitors: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.baseline_learning_rate = 0.1
        
        logger.info("System monitor initialized")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                snapshot = self._collect_metrics()
                self.metrics_history.append(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                # Update performance baselines
                self._update_baselines(snapshot)
                
                # Sleep until next collection
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> MetricSnapshot:
        """Collect comprehensive system metrics."""
        # Basic system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network and disk I/O
        net_io = psutil.net_io_counters()._asdict()
        disk_io = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        
        # GPU metrics (if available)
        gpu_usage = 0.0
        gpu_memory = 0.0
        temperature = None
        
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
                
                # Memory usage
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory = (allocated / total) * 100
                
                # Temperature (if supported)
                try:
                    import nvidia_ml_py as nvml
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    temp_info = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    temperature = float(temp_info)
                except (ImportError, Exception):
                    pass
                    
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        # Component-specific metrics
        component_metrics = {}
        for component, monitor_func in self.component_monitors.items():
            try:
                component_metrics[component] = monitor_func()
            except Exception as e:
                logger.warning(f"Component monitor '{component}' failed: {e}")
        
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_io=net_io,
            disk_io=disk_io,
            temperature=temperature
        )
        
        return snapshot
    
    def _check_alerts(self, snapshot: MetricSnapshot):
        """Check metrics against thresholds and generate alerts."""
        alerts_generated = []
        
        # Check CPU usage
        if snapshot.cpu_usage > self.thresholds['cpu_usage']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.WARNING if snapshot.cpu_usage < 95 else AlertLevel.CRITICAL,
                component="cpu",
                message=f"High CPU usage: {snapshot.cpu_usage:.1f}%",
                details={"usage": snapshot.cpu_usage, "threshold": self.thresholds['cpu_usage']}
            )
            alerts_generated.append(alert)
        
        # Check memory usage
        if snapshot.memory_usage > self.thresholds['memory_usage']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.WARNING if snapshot.memory_usage < 95 else AlertLevel.CRITICAL,
                component="memory",
                message=f"High memory usage: {snapshot.memory_usage:.1f}%",
                details={"usage": snapshot.memory_usage, "threshold": self.thresholds['memory_usage']}
            )
            alerts_generated.append(alert)
        
        # Check disk usage
        if snapshot.disk_usage > self.thresholds['disk_usage']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.CRITICAL,
                component="disk",
                message=f"High disk usage: {snapshot.disk_usage:.1f}%",
                details={"usage": snapshot.disk_usage, "threshold": self.thresholds['disk_usage']}
            )
            alerts_generated.append(alert)
        
        # Check GPU usage
        if snapshot.gpu_usage > self.thresholds['gpu_usage']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.WARNING,
                component="gpu",
                message=f"High GPU usage: {snapshot.gpu_usage:.1f}%",
                details={"usage": snapshot.gpu_usage, "threshold": self.thresholds['gpu_usage']}
            )
            alerts_generated.append(alert)
        
        # Check GPU memory
        if snapshot.gpu_memory > self.thresholds['gpu_memory']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.WARNING if snapshot.gpu_memory < 98 else AlertLevel.CRITICAL,
                component="gpu_memory",
                message=f"High GPU memory usage: {snapshot.gpu_memory:.1f}%",
                details={"usage": snapshot.gpu_memory, "threshold": self.thresholds['gpu_memory']}
            )
            alerts_generated.append(alert)
        
        # Check temperature
        if snapshot.temperature and snapshot.temperature > self.thresholds['temperature']:
            alert = Alert(
                timestamp=snapshot.timestamp,
                level=AlertLevel.WARNING if snapshot.temperature < 90 else AlertLevel.CRITICAL,
                component="temperature",
                message=f"High temperature: {snapshot.temperature:.1f}°C",
                details={"temperature": snapshot.temperature, "threshold": self.thresholds['temperature']}
            )
            alerts_generated.append(alert)
        
        # Process generated alerts
        for alert in alerts_generated:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Alert):
        """Process and handle alert."""
        # Add to alert history
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.FATAL: logger.critical
        }[alert.level]
        
        log_level(f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _update_baselines(self, snapshot: MetricSnapshot):
        """Update performance baselines using exponential moving average."""
        metrics = {
            'cpu_usage': snapshot.cpu_usage,
            'memory_usage': snapshot.memory_usage,
            'gpu_usage': snapshot.gpu_usage,
            'gpu_memory': snapshot.gpu_memory
        }
        
        for metric, value in metrics.items():
            if metric in self.performance_baselines:
                # Exponential moving average
                old_baseline = self.performance_baselines[metric]
                new_baseline = (1 - self.baseline_learning_rate) * old_baseline + \
                              self.baseline_learning_rate * value
                self.performance_baselines[metric] = new_baseline
            else:
                self.performance_baselines[metric] = value
    
    def register_component_monitor(self, name: str, monitor_func: Callable[[], Dict[str, Any]]):
        """Register component-specific monitoring function."""
        self.component_monitors[name] = monitor_func
        logger.info(f"Registered component monitor: {name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get most recent metrics snapshot."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[MetricSnapshot]:
        """Get metrics history for specified time period."""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self.alerts if not a.resolved]
    
    def resolve_alert(self, alert_index: int):
        """Mark alert as resolved."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            self.alerts[alert_index].resolve_timestamp = time.time()
            logger.info(f"Alert resolved: {self.alerts[alert_index].message}")
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = self.get_metrics_history(5)  # Last 5 minutes
        if not recent_metrics:
            return 0.0
        
        # Calculate average utilization
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_gpu = np.mean([m.gpu_usage for m in recent_metrics])
        
        # Health scoring (inverted - lower utilization = better health)
        cpu_health = max(0, 100 - avg_cpu)
        memory_health = max(0, 100 - avg_memory)
        gpu_health = max(0, 100 - avg_gpu) if avg_gpu > 0 else 100
        
        # Weight components
        overall_health = (cpu_health * 0.4 + memory_health * 0.4 + gpu_health * 0.2)
        
        # Penalty for active alerts
        active_alerts = self.get_active_alerts()
        alert_penalty = len(active_alerts) * 5  # 5 points per active alert
        
        return max(0.0, min(100.0, overall_health - alert_penalty))
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        health_score = self.get_system_health_score()
        
        report = {
            "timestamp": time.time(),
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical",
            "active_alerts_count": len(active_alerts),
            "total_alerts_24h": len([a for a in self.alerts if a.timestamp > time.time() - 86400]),
        }
        
        if current_metrics:
            report.update({
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "disk_usage": current_metrics.disk_usage,
                "gpu_usage": current_metrics.gpu_usage,
                "gpu_memory": current_metrics.gpu_memory,
                "temperature": current_metrics.temperature
            })
        
        if active_alerts:
            report["active_alerts"] = [
                {
                    "level": alert.level.value,
                    "component": alert.component,
                    "message": alert.message,
                    "age_minutes": (time.time() - alert.timestamp) / 60
                }
                for alert in active_alerts
            ]
        
        return report
    
    def export_metrics(self, filepath: Path, format: str = "json"):
        """Export metrics history to file."""
        if format == "json":
            data = []
            for snapshot in self.metrics_history:
                data.append({
                    "timestamp": snapshot.timestamp,
                    "cpu_usage": snapshot.cpu_usage,
                    "memory_usage": snapshot.memory_usage,
                    "disk_usage": snapshot.disk_usage,
                    "gpu_usage": snapshot.gpu_usage,
                    "gpu_memory": snapshot.gpu_memory,
                    "temperature": snapshot.temperature
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to: {filepath}")


class OptimizationMonitor:
    """Monitor optimization performance and convergence."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.optimization_history: deque = deque(maxlen=history_size)
        self.convergence_detector = ConvergenceDetector()
    
    def record_optimization_step(self, 
                                energy: float,
                                temperature: float,
                                acceptance_rate: float,
                                step_time_ms: float,
                                **kwargs):
        """Record optimization step metrics."""
        step_data = {
            'timestamp': time.time(),
            'energy': energy,
            'temperature': temperature,
            'acceptance_rate': acceptance_rate,
            'step_time_ms': step_time_ms,
            **kwargs
        }
        
        self.optimization_history.append(step_data)
        
        # Check convergence
        energies = [step['energy'] for step in list(self.optimization_history)[-100:]]
        if len(energies) >= 10:
            is_converged = self.convergence_detector.check_convergence(energies)
            step_data['converged'] = is_converged
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization performance summary."""
        if not self.optimization_history:
            return {"status": "No optimization data"}
        
        recent_steps = list(self.optimization_history)[-100:]
        
        energies = [step['energy'] for step in recent_steps]
        step_times = [step['step_time_ms'] for step in recent_steps]
        acceptance_rates = [step['acceptance_rate'] for step in recent_steps]
        
        return {
            "total_steps": len(self.optimization_history),
            "recent_steps": len(recent_steps),
            "best_energy": min(energies),
            "current_energy": energies[-1],
            "energy_improvement": energies[0] - energies[-1] if len(energies) > 1 else 0.0,
            "mean_step_time_ms": np.mean(step_times),
            "mean_acceptance_rate": np.mean(acceptance_rates),
            "convergence_status": self.convergence_detector.get_status(),
            "estimated_time_to_convergence": self.convergence_detector.estimate_time_to_convergence()
        }


class ConvergenceDetector:
    """Detect optimization convergence."""
    
    def __init__(self, 
                 window_size: int = 50,
                 tolerance: float = 1e-6,
                 patience: int = 20):
        
        self.window_size = window_size
        self.tolerance = tolerance
        self.patience = patience
        
        self.energy_history: deque = deque(maxlen=window_size)
        self.no_improvement_count = 0
        self.best_energy = float('inf')
        self.convergence_time: Optional[float] = None
    
    def check_convergence(self, energies: List[float]) -> bool:
        """Check if optimization has converged."""
        if len(energies) < self.window_size:
            return False
        
        current_best = min(energies)
        
        if current_best < self.best_energy - self.tolerance:
            self.best_energy = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check for convergence
        if self.no_improvement_count >= self.patience:
            if self.convergence_time is None:
                self.convergence_time = time.time()
            return True
        
        return False
    
    def get_status(self) -> str:
        """Get convergence status."""
        if self.convergence_time:
            return "converged"
        elif self.no_improvement_count > self.patience // 2:
            return "converging"
        else:
            return "optimizing"
    
    def estimate_time_to_convergence(self) -> Optional[float]:
        """Estimate time until convergence (if not already converged)."""
        if self.convergence_time:
            return 0.0
        
        if self.no_improvement_count > 0:
            remaining_patience = self.patience - self.no_improvement_count
            # Simple linear estimate (could be improved)
            return remaining_patience * 1.0  # Assume 1 second per step
        
        return None


def demo_monitoring():
    """Demonstrate monitoring system."""
    print("Comprehensive Monitoring System Demo")
    print("=" * 40)
    
    # Create system monitor
    monitor = SystemMonitor(monitoring_interval=2.0)
    
    # Add alert callback
    def alert_handler(alert):
        print(f"🚨 ALERT: {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("Monitoring started. Collecting metrics for 10 seconds...")
    time.sleep(10)
    
    # Get health report
    report = monitor.generate_health_report()
    print(f"\nHealth Report:")
    print(f"  Health Score: {report['health_score']:.1f}/100")
    print(f"  Status: {report['status']}")
    print(f"  CPU Usage: {report['cpu_usage']:.1f}%")
    print(f"  Memory Usage: {report['memory_usage']:.1f}%")
    print(f"  Active Alerts: {report['active_alerts_count']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("Monitoring stopped.")


if __name__ == "__main__":
    demo_monitoring()