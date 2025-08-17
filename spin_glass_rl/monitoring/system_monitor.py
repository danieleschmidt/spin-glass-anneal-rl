"""System monitoring for optimization performance and health."""

import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetric:
    """System metric data point."""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """System alert."""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class PerformanceTracker:
    """Track performance metrics for optimization operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record a performance metric."""
        metric = SystemMetric(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        self.metrics[name].append(metric)
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation execution."""
        self.operation_counts[operation] += 1
        self.record_metric(f"{operation}_duration", duration, "seconds", {"operation": operation})
        
        if not success:
            self.error_counts[operation] += 1
            self.record_metric(f"{operation}_errors", 1, "count", {"operation": operation})
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics:
            return {}
        
        values = [m.value for m in self.metrics[metric_name]]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "sum": sum(values)
        }
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations."""
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())
        uptime = time.time() - self.start_time
        
        operation_details = {}
        for op, count in self.operation_counts.items():
            errors = self.error_counts.get(op, 0)
            error_rate = errors / count if count > 0 else 0
            
            # Get duration stats if available
            duration_stats = self.get_metric_stats(f"{op}_duration")
            
            operation_details[op] = {
                "count": count,
                "errors": errors,
                "error_rate": error_rate,
                "duration_stats": duration_stats
            }
        
        return {
            "uptime_seconds": uptime,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / max(total_operations, 1),
            "operations_per_second": total_operations / max(uptime, 1),
            "operation_details": operation_details
        }


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            # Try to use psutil if available
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available": True
            }
        except ImportError:
            # Fallback: try to read from /proc/self/status
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            rss_kb = int(line.split()[1])
                            return {
                                "rss_mb": rss_kb / 1024,
                                "vms_mb": 0,  # Not available
                                "percent": 0,  # Not available
                                "available": True
                            }
            except Exception:
                pass
            
            return {"available": False}
    
    def _get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage information."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "available": True
            }
        except ImportError:
            # Fallback: try to read from /proc/loadavg
            try:
                with open('/proc/loadavg', 'r') as f:
                    load_avg = float(f.read().split()[0])
                
                return {
                    "load_avg": load_avg,
                    "available": True
                }
            except Exception:
                return {"available": False}
    
    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Get current resource usage snapshot."""
        return {
            "timestamp": time.time(),
            "memory": self._get_memory_usage(),
            "cpu": self._get_cpu_usage(),
            "baseline_memory": self.baseline_memory
        }
    
    def check_resource_alerts(self, snapshot: Dict[str, Any]) -> List[Alert]:
        """Check for resource usage alerts."""
        alerts = []
        timestamp = snapshot["timestamp"]
        
        # Memory alerts
        memory = snapshot["memory"]
        if memory.get("available", False):
            # Alert if memory usage > 1GB
            if memory.get("rss_mb", 0) > 1024:
                alerts.append(Alert(
                    timestamp=timestamp,
                    level=AlertLevel.WARNING,
                    component="memory",
                    message=f"High memory usage: {memory['rss_mb']:.1f} MB",
                    metric_value=memory["rss_mb"],
                    threshold=1024
                ))
            
            # Alert if memory usage > 80%
            if memory.get("percent", 0) > 80:
                alerts.append(Alert(
                    timestamp=timestamp,
                    level=AlertLevel.ERROR,
                    component="memory",
                    message=f"Critical memory usage: {memory['percent']:.1f}%",
                    metric_value=memory["percent"],
                    threshold=80
                ))
        
        # CPU alerts
        cpu = snapshot["cpu"]
        if cpu.get("available", False):
            if cpu.get("cpu_percent", 0) > 90:
                alerts.append(Alert(
                    timestamp=timestamp,
                    level=AlertLevel.WARNING,
                    component="cpu",
                    message=f"High CPU usage: {cpu['cpu_percent']:.1f}%",
                    metric_value=cpu["cpu_percent"],
                    threshold=90
                ))
        
        return alerts


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, alert_history_size: int = 100):
        self.performance_tracker = PerformanceTracker()
        self.resource_monitor = ResourceMonitor()
        self.alerts = deque(maxlen=alert_history_size)
        self.monitoring_enabled = True
        self.last_check_time = time.time()
        
    def start_operation(self, operation_name: str) -> 'OperationContext':
        """Start monitoring an operation."""
        return OperationContext(self, operation_name)
    
    def record_operation_result(self, operation: str, duration: float, success: bool = True):
        """Record operation completion."""
        if self.monitoring_enabled:
            self.performance_tracker.record_operation(operation, duration, success)
    
    def record_custom_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record custom metric."""
        if self.monitoring_enabled:
            self.performance_tracker.record_metric(name, value, unit, tags)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        # Get resource snapshot
        resource_snapshot = self.resource_monitor.get_resource_snapshot()
        
        # Check for alerts
        new_alerts = self.resource_monitor.check_resource_alerts(resource_snapshot)
        
        # Add new alerts to history
        for alert in new_alerts:
            self.alerts.append(alert)
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_operation_summary()
        
        # Determine overall health status
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 300]  # Last 5 minutes
        critical_alerts = [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in recent_alerts if a.level == AlertLevel.ERROR]
        
        if critical_alerts:
            health_status = "critical"
        elif error_alerts:
            health_status = "degraded"
        elif recent_alerts:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "timestamp": time.time(),
            "health_status": health_status,
            "resource_usage": resource_snapshot,
            "performance": performance_summary,
            "recent_alerts": [asdict(a) for a in recent_alerts],
            "total_alerts": len(self.alerts),
            "monitoring_enabled": self.monitoring_enabled
        }
    
    def get_monitoring_report(self) -> str:
        """Generate human-readable monitoring report."""
        health = self.check_system_health()
        
        report = [
            "📊 System Monitoring Report",
            "=" * 40,
            f"Status: {health['health_status'].upper()}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(health['timestamp']))}",
            ""
        ]
        
        # Resource usage
        resources = health["resource_usage"]
        if resources["memory"].get("available", False):
            memory = resources["memory"]
            report.append(f"💾 Memory: {memory.get('rss_mb', 0):.1f} MB ({memory.get('percent', 0):.1f}%)")
        
        if resources["cpu"].get("available", False):
            cpu = resources["cpu"]
            if "cpu_percent" in cpu:
                report.append(f"⚡ CPU: {cpu['cpu_percent']:.1f}%")
            elif "load_avg" in cpu:
                report.append(f"⚡ Load: {cpu['load_avg']:.2f}")
        
        # Performance summary
        perf = health["performance"]
        report.extend([
            "",
            f"🚀 Performance:",
            f"  Uptime: {perf['uptime_seconds']:.1f}s",
            f"  Operations: {perf['total_operations']} ({perf['operations_per_second']:.2f}/s)",
            f"  Error Rate: {perf['overall_error_rate']:.2%}"
        ])
        
        # Recent alerts
        recent_alerts = health["recent_alerts"]
        if recent_alerts:
            report.extend(["", "🚨 Recent Alerts:"])
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                timestamp = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
                report.append(f"  {timestamp} {alert['level'].upper()}: {alert['message']}")
        
        return "\n".join(report)
    
    def enable_monitoring(self):
        """Enable system monitoring."""
        self.monitoring_enabled = True
    
    def disable_monitoring(self):
        """Disable system monitoring."""
        self.monitoring_enabled = False


class OperationContext:
    """Context manager for monitoring operations."""
    
    def __init__(self, monitor: SystemMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            success = exc_type is None and self.success
            self.monitor.record_operation_result(self.operation_name, duration, success)
    
    def mark_failure(self):
        """Mark operation as failed."""
        self.success = False


# Global system monitor instance
global_monitor = SystemMonitor()


def monitored_operation(operation_name: str):
    """Decorator for monitoring operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with global_monitor.start_operation(operation_name) as context:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context.mark_failure()
                    raise e
        return wrapper
    return decorator


# Quick test function
def test_system_monitoring():
    """Test system monitoring framework."""
    print("📊 Testing System Monitoring Framework...")
    
    monitor = SystemMonitor()
    
    # Test operation monitoring
    with monitor.start_operation("test_operation") as op:
        time.sleep(0.1)  # Simulate work
        monitor.record_custom_metric("test_metric", 42.0, "units")
    
    # Test operation with failure
    with monitor.start_operation("failing_operation") as op:
        op.mark_failure()
    
    # Test direct operation recording
    monitor.record_operation_result("quick_operation", 0.05, True)
    monitor.record_operation_result("slow_operation", 2.0, False)
    
    # Get health check
    health = monitor.check_system_health()
    print(f"✅ Health status: {health['health_status']}")
    print(f"✅ Total operations: {health['performance']['total_operations']}")
    
    # Test decorated function
    @monitored_operation("decorated_test")
    def test_function(x):
        time.sleep(0.01)
        return x * 2
    
    result = test_function(21)
    print(f"✅ Decorated function result: {result}")
    
    # Generate report
    report = monitor.get_monitoring_report()
    print("\n" + report)
    
    print("\n📊 System Monitoring Framework test completed!")


if __name__ == "__main__":
    test_system_monitoring()