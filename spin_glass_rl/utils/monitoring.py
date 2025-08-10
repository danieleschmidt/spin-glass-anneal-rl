"""Advanced monitoring and health checking utilities."""

import time
import torch
import psutil
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""
    energy: float
    temperature: float
    acceptance_rate: float
    sweep_time_ms: float
    convergence_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, 
                 max_history: int = 1000,
                 collection_interval: float = 1.0,
                 enable_gpu_monitoring: bool = True):
        self.max_history = max_history
        self.collection_interval = collection_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=max_history)
        self.optimization_metrics: deque = deque(maxlen=max_history)
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Alerts and thresholds
        self.alert_callbacks: List[Callable] = []
        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "gpu_memory_percent": 95.0,
            "gpu_temperature": 85.0,
        }
        
        # Performance statistics
        self.stats = defaultdict(list)
        
        logger.info(f"Performance monitor initialized (GPU monitoring: {self.enable_gpu_monitoring})")
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                self._check_alerts(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # GPU metrics
        gpu_memory_used_gb = 0.0
        gpu_memory_total_gb = 0.0
        gpu_utilization = 0.0
        
        if self.enable_gpu_monitoring:
            try:
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # GPU utilization (approximation)
                gpu_utilization = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
                
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_utilization=gpu_utilization
        )
    
    def record_optimization_metrics(self, 
                                   energy: float, 
                                   temperature: float, 
                                   acceptance_rate: float, 
                                   sweep_time_ms: float) -> None:
        """Record optimization step metrics."""
        metrics = OptimizationMetrics(
            energy=energy,
            temperature=temperature,
            acceptance_rate=acceptance_rate,
            sweep_time_ms=sweep_time_ms
        )
        
        self.optimization_metrics.append(metrics)
        
        # Calculate convergence rate
        if len(self.optimization_metrics) > 1:
            prev_energy = self.optimization_metrics[-2].energy
            convergence_rate = abs(energy - prev_energy) / max(abs(prev_energy), 1e-10)
            metrics.convergence_rate = convergence_rate
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if self.enable_gpu_monitoring:
            gpu_memory_percent = (metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb) * 100
            if gpu_memory_percent > self.thresholds["gpu_memory_percent"]:
                alerts.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
        
        # Fire alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and optimization metrics."""
        current_system = self._collect_system_metrics()
        
        result = {
            "system": {
                "cpu_percent": current_system.cpu_percent,
                "memory_percent": current_system.memory_percent,
                "memory_available_gb": current_system.memory_available_gb,
                "gpu_memory_used_gb": current_system.gpu_memory_used_gb,
                "gpu_memory_total_gb": current_system.gpu_memory_total_gb,
                "gpu_utilization": current_system.gpu_utilization,
            },
            "optimization": {}
        }
        
        if self.optimization_metrics:
            latest_opt = self.optimization_metrics[-1]
            result["optimization"] = {
                "energy": latest_opt.energy,
                "temperature": latest_opt.temperature,
                "acceptance_rate": latest_opt.acceptance_rate,
                "sweep_time_ms": latest_opt.sweep_time_ms,
                "convergence_rate": latest_opt.convergence_rate,
            }
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        # System statistics
        if self.system_metrics:
            cpu_values = [m.cpu_percent for m in self.system_metrics]
            memory_values = [m.memory_percent for m in self.system_metrics]
            
            stats["system"] = {
                "cpu_mean": sum(cpu_values) / len(cpu_values),
                "cpu_max": max(cpu_values),
                "memory_mean": sum(memory_values) / len(memory_values),
                "memory_max": max(memory_values),
            }
            
            if self.enable_gpu_monitoring:
                gpu_util_values = [m.gpu_utilization for m in self.system_metrics]
                stats["system"]["gpu_utilization_mean"] = sum(gpu_util_values) / len(gpu_util_values)
                stats["system"]["gpu_utilization_max"] = max(gpu_util_values)
        
        # Optimization statistics
        if self.optimization_metrics:
            energy_values = [m.energy for m in self.optimization_metrics]
            sweep_times = [m.sweep_time_ms for m in self.optimization_metrics]
            acceptance_rates = [m.acceptance_rate for m in self.optimization_metrics]
            
            stats["optimization"] = {
                "energy_min": min(energy_values),
                "energy_max": max(energy_values),
                "energy_final": energy_values[-1],
                "energy_improvement": energy_values[0] - energy_values[-1] if len(energy_values) > 1 else 0,
                "sweep_time_mean_ms": sum(sweep_times) / len(sweep_times),
                "sweep_time_max_ms": max(sweep_times),
                "acceptance_rate_mean": sum(acceptance_rates) / len(acceptance_rates),
                "total_sweeps": len(self.optimization_metrics),
            }
        
        return stats
    
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics to file."""
        try:
            data = {
                "system_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "memory_available_gb": m.memory_available_gb,
                        "gpu_memory_used_gb": m.gpu_memory_used_gb,
                        "gpu_memory_total_gb": m.gpu_memory_total_gb,
                        "gpu_utilization": m.gpu_utilization,
                    }
                    for m in self.system_metrics
                ],
                "optimization_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "energy": m.energy,
                        "temperature": m.temperature,
                        "acceptance_rate": m.acceptance_rate,
                        "sweep_time_ms": m.sweep_time_ms,
                        "convergence_rate": m.convergence_rate,
                    }
                    for m in self.optimization_metrics
                ],
                "statistics": self.get_statistics()
            }
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise


class HealthChecker:
    """System health checking utilities."""
    
    @staticmethod
    def check_system_requirements() -> Dict[str, Any]:
        """Check if system meets minimum requirements."""
        checks = {}
        
        # Python version
        import sys
        python_version = sys.version_info
        checks["python_version"] = {
            "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "supported": python_version >= (3, 8),
            "recommendation": "Python 3.8+ required"
        }
        
        # PyTorch
        try:
            import torch
            checks["pytorch"] = {
                "version": torch.__version__,
                "available": True,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            checks["pytorch"] = {"available": False, "recommendation": "Install PyTorch"}
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        checks["memory"] = {
            "total_gb": memory_gb,
            "available_gb": memory.available / (1024**3),
            "sufficient": memory_gb >= 4.0,  # Minimum 4GB
            "recommendation": "8GB+ RAM recommended for large problems"
        }
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        checks["disk_space"] = {
            "free_gb": disk_free_gb,
            "sufficient": disk_free_gb >= 1.0,  # Minimum 1GB free
            "recommendation": "2GB+ free space recommended"
        }
        
        # CPU cores
        cpu_count = psutil.cpu_count()
        checks["cpu"] = {
            "cores": cpu_count,
            "sufficient": cpu_count >= 2,
            "recommendation": "4+ cores recommended for parallel operations"
        }
        
        return checks
    
    @staticmethod
    def check_gpu_health() -> Dict[str, Any]:
        """Check GPU health and capabilities."""
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        
        try:
            device_count = torch.cuda.device_count()
            devices = []
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_total = props.total_memory / (1024**3)
                
                devices.append({
                    "index": i,
                    "name": props.name,
                    "memory_total_gb": memory_total,
                    "memory_allocated_gb": memory_allocated,
                    "memory_free_gb": memory_total - memory_allocated,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessors": props.multi_processor_count,
                    "healthy": memory_allocated / memory_total < 0.95  # Not critically full
                })
            
            return {
                "available": True,
                "device_count": device_count,
                "devices": devices,
                "driver_version": torch.version.cuda
            }
            
        except Exception as e:
            return {"available": True, "error": str(e), "healthy": False}
    
    @staticmethod
    def diagnose_performance_issues(monitor: PerformanceMonitor) -> List[str]:
        """Diagnose potential performance issues."""
        issues = []
        
        try:
            stats = monitor.get_statistics()
            
            # System resource issues
            if "system" in stats:
                sys_stats = stats["system"]
                
                if sys_stats.get("cpu_mean", 0) > 85:
                    issues.append("High average CPU usage may slow optimization")
                
                if sys_stats.get("memory_max", 0) > 90:
                    issues.append("High memory usage detected, consider reducing problem size")
                
                if sys_stats.get("gpu_utilization_mean", 0) < 30 and torch.cuda.is_available():
                    issues.append("Low GPU utilization, check if GPU acceleration is enabled")
            
            # Optimization performance issues
            if "optimization" in stats:
                opt_stats = stats["optimization"]
                
                if opt_stats.get("sweep_time_mean_ms", 0) > 1000:
                    issues.append("Slow sweep times detected, consider smaller problems or GPU acceleration")
                
                if opt_stats.get("acceptance_rate_mean", 0) < 0.1:
                    issues.append("Very low acceptance rate, try higher initial temperature")
                
                if opt_stats.get("acceptance_rate_mean", 0) > 0.9:
                    issues.append("Very high acceptance rate, try lower initial temperature")
                
                if opt_stats.get("energy_improvement", 0) < 1e-6:
                    issues.append("Little energy improvement, may need more sweeps or better initialization")
        
        except Exception as e:
            issues.append(f"Error diagnosing performance: {e}")
        
        return issues


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_global_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_global_monitor()
    monitor.start_monitoring()


def stop_global_monitoring() -> None:
    """Stop global performance monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


def get_system_health_report() -> Dict[str, Any]:
    """Get comprehensive system health report."""
    health_checker = HealthChecker()
    
    report = {
        "timestamp": time.time(),
        "system_requirements": health_checker.check_system_requirements(),
        "gpu_health": health_checker.check_gpu_health(),
    }
    
    # Add current performance metrics if monitoring is active
    global _global_monitor
    if _global_monitor:
        report["current_metrics"] = _global_monitor.get_current_metrics()
        report["performance_statistics"] = _global_monitor.get_statistics()
        report["performance_issues"] = health_checker.diagnose_performance_issues(_global_monitor)
    
    return report