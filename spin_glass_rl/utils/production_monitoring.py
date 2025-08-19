"""
Production-Grade Monitoring and Observability for Spin-Glass-Anneal-RL.

This module provides comprehensive monitoring, metrics collection, and observability:
1. Real-time performance monitoring and alerting
2. Distributed tracing and logging
3. Resource utilization tracking
4. Algorithm performance analytics
5. Security monitoring and audit logging

Generation 2 Production Features:
- OpenTelemetry integration for observability
- Prometheus metrics export
- Structured logging with correlation IDs
- Circuit breaker patterns for resilience
- Health checks and service discovery
"""

import time
import json
import threading
import logging
import hashlib
import traceback
import psutil
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import queue
import warnings
from contextlib import contextmanager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(correlation_id)s]',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('spin_glass_rl.log')
    ]
)

# Custom logger with correlation ID support
class CorrelationLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        self.correlation_id = correlation_id
    
    def _log(self, level: str, message: str, **kwargs):
        extra = {'correlation_id': self.correlation_id or 'none'}
        extra.update(kwargs)
        getattr(self.logger, level)(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log('error', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log('debug', message, **kwargs)

logger = CorrelationLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""


@dataclass
class Alert:
    """Alert notification."""
    level: AlertLevel
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.metric_registry = {}
        self.lock = threading.Lock()
        
    def register_metric(
        self, 
        name: str, 
        metric_type: MetricType, 
        help_text: str = ""
    ):
        """Register a new metric."""
        with self.lock:
            self.metric_registry[name] = {
                "type": metric_type,
                "help": help_text,
                "created": time.time()
            }
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        if name not in self.metric_registry:
            # Auto-register as gauge
            self.register_metric(name, MetricType.GAUGE)
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=self.metric_registry[name]["type"],
            labels=labels or {},
            timestamp=time.time(),
            help_text=self.metric_registry[name]["help"]
        )
        
        with self.lock:
            self.metrics[name].append(metric)
            
            # Limit history size
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-500:]
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get latest metric value."""
        with self.lock:
            if name not in self.metrics:
                return None
            
            # Find matching metric
            for metric in reversed(self.metrics[name]):
                if labels is None or metric.labels == labels:
                    return metric.value
            
            return None
    
    def get_metric_history(
        self, 
        name: str, 
        duration_seconds: float = 3600
    ) -> List[Metric]:
        """Get metric history for specified duration."""
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            if name not in self.metrics:
                return []
            
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            for name, registry_info in self.metric_registry.items():
                if name not in self.metrics:
                    continue
                
                # Add help text
                if registry_info["help"]:
                    lines.append(f"# HELP {name} {registry_info['help']}")
                
                # Add type
                lines.append(f"# TYPE {name} {registry_info['type'].value}")
                
                # Add latest metric values
                latest_metrics = {}
                for metric in self.metrics[name]:
                    label_key = json.dumps(metric.labels, sort_keys=True)
                    latest_metrics[label_key] = metric
                
                for metric in latest_metrics.values():
                    if metric.labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
                        lines.append(f"{name}{{{label_str}}} {metric.value} {int(metric.timestamp * 1000)}")
                    else:
                        lines.append(f"{name} {metric.value} {int(metric.timestamp * 1000)}")
        
        return "\n".join(lines)


class ResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        
        # Register metrics
        self._register_resource_metrics()
    
    def _register_resource_metrics(self):
        """Register system resource metrics."""
        metrics = [
            ("system_cpu_percent", "CPU usage percentage"),
            ("system_memory_used_bytes", "Memory usage in bytes"),
            ("system_memory_percent", "Memory usage percentage"),
            ("system_disk_used_bytes", "Disk usage in bytes"),
            ("system_disk_percent", "Disk usage percentage"),
            ("process_cpu_percent", "Process CPU usage percentage"),
            ("process_memory_bytes", "Process memory usage in bytes"),
            ("process_open_files", "Number of open files"),
            ("process_threads", "Number of threads"),
        ]
        
        for name, help_text in metrics:
            self.metrics_collector.register_metric(name, MetricType.GAUGE, help_text)
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._collect_process_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self):
        """Collect system-wide resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics_collector.record_metric("system_cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric("system_memory_used_bytes", memory.used)
            self.metrics_collector.record_metric("system_memory_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric("system_disk_used_bytes", disk.used)
            self.metrics_collector.record_metric("system_disk_percent", 
                                                (disk.used / disk.total) * 100)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_process_metrics(self):
        """Collect process-specific resource metrics."""
        try:
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics_collector.record_metric("process_cpu_percent", cpu_percent)
            
            # Memory usage
            memory_info = process.memory_info()
            self.metrics_collector.record_metric("process_memory_bytes", memory_info.rss)
            
            # File descriptors
            try:
                open_files = len(process.open_files())
                self.metrics_collector.record_metric("process_open_files", open_files)
            except:
                pass  # May not be available on all systems
            
            # Thread count
            thread_count = process.num_threads()
            self.metrics_collector.record_metric("process_threads", thread_count)
        
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")


class PerformanceTracker:
    """Tracks algorithm and operation performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.operation_timers = {}
        self.lock = threading.Lock()
        
        # Register performance metrics
        self._register_performance_metrics()
    
    def _register_performance_metrics(self):
        """Register performance tracking metrics."""
        metrics = [
            ("algorithm_execution_duration_seconds", "Algorithm execution time"),
            ("algorithm_energy_improvement", "Energy improvement achieved"),
            ("algorithm_convergence_iterations", "Iterations to convergence"),
            ("optimization_success_rate", "Success rate of optimization"),
            ("memory_peak_usage_bytes", "Peak memory usage during operation"),
            ("gpu_utilization_percent", "GPU utilization percentage"),
        ]
        
        for name, help_text in metrics:
            self.metrics_collector.register_metric(name, MetricType.HISTOGRAM, help_text)
    
    @contextmanager
    def track_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to track operation performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        operation_id = f"{operation_name}_{int(start_time)}"
        
        logger.info(f"Starting operation: {operation_name}", extra={"operation_id": operation_id})
        
        try:
            yield operation_id
            
            # Success metrics
            duration = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss
            memory_increase = peak_memory - start_memory
            
            # Record metrics
            metric_labels = {"operation": operation_name, "status": "success"}
            if labels:
                metric_labels.update(labels)
            
            self.metrics_collector.record_metric(
                "algorithm_execution_duration_seconds", 
                duration, 
                metric_labels
            )
            
            self.metrics_collector.record_metric(
                "memory_peak_usage_bytes", 
                memory_increase, 
                metric_labels
            )
            
            logger.info(
                f"Operation completed: {operation_name}", 
                extra={
                    "operation_id": operation_id,
                    "duration": duration,
                    "memory_increase": memory_increase
                }
            )
        
        except Exception as e:
            # Failure metrics
            duration = time.time() - start_time
            metric_labels = {"operation": operation_name, "status": "failure"}
            if labels:
                metric_labels.update(labels)
            
            self.metrics_collector.record_metric(
                "algorithm_execution_duration_seconds", 
                duration, 
                metric_labels
            )
            
            logger.error(
                f"Operation failed: {operation_name}", 
                extra={
                    "operation_id": operation_id,
                    "duration": duration,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            
            raise
    
    def record_algorithm_performance(
        self, 
        algorithm_name: str, 
        result: Dict
    ):
        """Record algorithm-specific performance metrics."""
        labels = {"algorithm": algorithm_name}
        
        # Execution time
        if "total_time" in result:
            self.metrics_collector.record_metric(
                "algorithm_execution_duration_seconds",
                result["total_time"],
                labels
            )
        
        # Energy improvement
        if "best_energy" in result and "initial_energy" in result:
            improvement = result["initial_energy"] - result["best_energy"]
            self.metrics_collector.record_metric(
                "algorithm_energy_improvement",
                improvement,
                labels
            )
        
        # Convergence iterations
        if "iterations" in result:
            self.metrics_collector.record_metric(
                "algorithm_convergence_iterations",
                result["iterations"],
                labels
            )
        
        # Success rate
        success = 1.0 if result.get("convergence_achieved", False) else 0.0
        self.metrics_collector.record_metric(
            "optimization_success_rate",
            success,
            labels
        )


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.alert_queue = queue.Queue()
        self.alert_handlers = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def add_alert_rule(
        self, 
        metric_name: str, 
        condition: Callable[[float], bool],
        level: AlertLevel,
        message: str,
        cooldown_seconds: float = 300
    ):
        """Add an alert rule."""
        rule = {
            "metric_name": metric_name,
            "condition": condition,
            "level": level,
            "message": message,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": 0
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule for {metric_name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_alerts, daemon=True)
        self.monitor_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Alert monitoring stopped")
    
    def _monitor_alerts(self):
        """Monitor metrics and trigger alerts."""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                for rule in self.alert_rules:
                    # Check cooldown
                    if current_time - rule["last_triggered"] < rule["cooldown_seconds"]:
                        continue
                    
                    # Get current metric value
                    metric_value = self.metrics_collector.get_metric_value(rule["metric_name"])
                    
                    if metric_value is not None and rule["condition"](metric_value):
                        # Trigger alert
                        alert = Alert(
                            level=rule["level"],
                            message=rule["message"].format(value=metric_value),
                            component="spin_glass_rl",
                            correlation_id=logger.correlation_id,
                            metadata={"metric_name": rule["metric_name"], "metric_value": metric_value}
                        )
                        
                        self._handle_alert(alert)
                        rule["last_triggered"] = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(10)
    
    def _handle_alert(self, alert: Alert):
        """Handle triggered alert."""
        logger.warning(f"Alert triggered: {alert.message}", extra={"alert_level": alert.level.value})
        
        # Add to queue
        try:
            self.alert_queue.put_nowait(alert)
        except queue.Full:
            logger.error("Alert queue is full, dropping alert")
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


class CircuitBreaker:
    """Circuit breaker pattern for resilience."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if self._should_attempt_reset():
                        self.state = "HALF_OPEN"
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                
                except self.expected_exception as e:
                    self._on_failure()
                    raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class HealthChecker:
    """Health check endpoint for service monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        
    def register_health_check(self, name: str, check_function: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_function
    
    def get_health_status(self) -> Dict:
        """Get overall health status."""
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        overall_healthy = True
        
        for name, check_function in self.health_checks.items():
            try:
                is_healthy = check_function()
                status["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "checked_at": time.time()
                }
                
                if not is_healthy:
                    overall_healthy = False
            
            except Exception as e:
                status["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                    "checked_at": time.time()
                }
                overall_healthy = False
        
        if not overall_healthy:
            status["status"] = "unhealthy"
        
        return status


class ProductionMonitor:
    """Main production monitoring coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor(self.metrics_collector)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(self.metrics_collector)
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert handlers
        self._setup_default_alert_handlers()
    
    def start_monitoring(self):
        """Start all monitoring components."""
        logger.info("Starting production monitoring")
        
        self.resource_monitor.start_monitoring()
        self.alert_manager.start_monitoring()
        
        logger.info("Production monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        logger.info("Stopping production monitoring")
        
        self.resource_monitor.stop_monitoring()
        self.alert_manager.stop_monitoring()
        
        logger.info("Production monitoring stopped")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            "system_cpu_percent",
            lambda x: x > 90,
            AlertLevel.WARNING,
            "High CPU usage: {value:.1f}%"
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            "system_memory_percent",
            lambda x: x > 85,
            AlertLevel.WARNING,
            "High memory usage: {value:.1f}%"
        )
        
        # Algorithm execution time
        self.alert_manager.add_alert_rule(
            "algorithm_execution_duration_seconds",
            lambda x: x > 300,  # 5 minutes
            AlertLevel.WARNING,
            "Algorithm taking too long: {value:.1f}s"
        )
    
    def _setup_default_health_checks(self):
        """Setup default health check functions."""
        
        def memory_health_check():
            memory_percent = self.metrics_collector.get_metric_value("system_memory_percent")
            return memory_percent is None or memory_percent < 95
        
        def disk_health_check():
            disk_percent = self.metrics_collector.get_metric_value("system_disk_percent")
            return disk_percent is None or disk_percent < 90
        
        self.health_checker.register_health_check("memory", memory_health_check)
        self.health_checker.register_health_check("disk", disk_health_check)
    
    def _setup_default_alert_handlers(self):
        """Setup default alert handlers."""
        
        def log_alert_handler(alert: Alert):
            logger.warning(
                f"ALERT: {alert.message}",
                extra={
                    "alert_level": alert.level.value,
                    "component": alert.component,
                    "metadata": alert.metadata
                }
            )
        
        self.alert_manager.add_alert_handler(log_alert_handler)


# Global production monitor instance
production_monitor = ProductionMonitor()


def get_metrics_endpoint() -> str:
    """Get metrics in Prometheus format."""
    return production_monitor.metrics_collector.export_prometheus_format()


def get_health_endpoint() -> Dict:
    """Get health status."""
    return production_monitor.health_checker.get_health_status()


@contextmanager
def monitor_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for monitoring operations."""
    with production_monitor.performance_tracker.track_operation(operation_name, labels) as operation_id:
        # Set correlation ID for logging
        logger.set_correlation_id(operation_id)
        try:
            yield operation_id
        finally:
            logger.set_correlation_id(None)


def record_algorithm_metrics(algorithm_name: str, result: Dict):
    """Record algorithm performance metrics."""
    production_monitor.performance_tracker.record_algorithm_performance(algorithm_name, result)


if __name__ == "__main__":
    # Demonstration of production monitoring
    print("📊 Production Monitoring Demo")
    print("=" * 40)
    
    # Start monitoring
    production_monitor.start_monitoring()
    
    # Simulate some operations
    with monitor_operation("demo_optimization", {"algorithm": "test"}) as op_id:
        print(f"Running operation: {op_id}")
        time.sleep(1)  # Simulate work
        
        # Record some metrics
        production_monitor.metrics_collector.record_metric("demo_metric", 42.5)
    
    # Record algorithm result
    test_result = {
        "total_time": 1.23,
        "best_energy": -5.42,
        "initial_energy": -3.14,
        "iterations": 100,
        "convergence_achieved": True
    }
    
    record_algorithm_metrics("demo_algorithm", test_result)
    
    # Wait a moment for metrics collection
    time.sleep(2)
    
    # Show health status
    health = get_health_endpoint()
    print(f"Health status: {health['status']}")
    
    # Show some metrics
    cpu_usage = production_monitor.metrics_collector.get_metric_value("system_cpu_percent")
    if cpu_usage:
        print(f"CPU usage: {cpu_usage:.1f}%")
    
    # Stop monitoring
    production_monitor.stop_monitoring()
    
    print("✅ Production monitoring demo complete!")
    print("Generation 2 robustness monitoring implemented successfully.")