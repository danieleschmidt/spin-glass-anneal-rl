"""Comprehensive monitoring and metrics collection."""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import torch
import numpy as np
from enum import Enum
import json
import logging


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    GPU = "gpu"
    OPTIMIZATION = "optimization"
    SYSTEM = "system"


@dataclass
class MetricSnapshot:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_type: MetricType = MetricType.PERFORMANCE


@dataclass
class SystemResources:
    """System resource usage snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Monitor system and optimization performance."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        
        # Setup logging
        self.logger = logging.getLogger("SpinGlassRL.Monitor")
        self.logger.setLevel(logging.INFO)
        
        # Performance counters
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitor_interval = interval
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                resources = self._collect_system_resources()
                self._record_system_metrics(resources)
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_resources(self) -> SystemResources:
        """Collect current system resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent
        )
        
        return resources
    
    def _record_system_metrics(self, resources: SystemResources) -> None:
        """Record system metrics."""
        self.record_metric("system.cpu_percent", resources.cpu_percent, MetricType.SYSTEM)
        self.record_metric("system.memory_percent", resources.memory_percent, MetricType.MEMORY)
        self.record_metric("system.memory_available_gb", resources.memory_available_gb, MetricType.MEMORY)
        self.record_metric("system.disk_usage_percent", resources.disk_usage_percent, MetricType.SYSTEM)
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.PERFORMANCE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric value."""
        snapshot = MetricSnapshot(
            name=name,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            metric_type=metric_type
        )
        
        self.metrics[name].append(snapshot)
    
    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation execution time."""
        self.operation_counts[operation] += 1
        self.operation_times[operation].append(duration)
        self.record_metric(f"operation.{operation}.duration", duration, MetricType.PERFORMANCE)
    
    def record_error(self, component: str, error_type: str) -> None:
        """Record an error occurrence."""
        error_key = f"{component}.{error_type}"
        self.error_counts[error_key] += 1
        self.record_metric(f"errors.{error_key}", 1, MetricType.SYSTEM)
    
    def get_metric_summary(self, name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        if name not in self.metrics:
            return {}
        
        recent_values = list(self.metrics[name])[-window_size:]
        if not recent_values:
            return {}
        
        values = [snapshot.value for snapshot in recent_values]
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "p95": np.percentile(values, 95),
            "latest": values[-1] if values else 0.0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "monitoring_duration": 0.0,
            "system_metrics": {},
            "operation_stats": {},
            "error_summary": dict(self.error_counts),
            "alerts": []
        }
        
        # System metrics summary
        system_metrics = [name for name in self.metrics.keys() if name.startswith("system.")]
        for metric in system_metrics:
            report["system_metrics"][metric] = self.get_metric_summary(metric)
        
        # Operation statistics
        for operation, times in self.operation_times.items():
            if times:
                report["operation_stats"][operation] = {
                    "total_calls": self.operation_counts[operation],
                    "avg_time": np.mean(times),
                    "total_time": np.sum(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times)
                }
        
        return report


# Global monitoring instances
global_performance_monitor = PerformanceMonitor()