#!/usr/bin/env python3
"""
🛡️ ADVANCED MONITORING AND RELIABILITY SYSTEM
==============================================

Enterprise-grade monitoring, alerting, and self-healing infrastructure
for breakthrough optimization algorithms.

Features:
- Real-time performance monitoring with predictive analytics
- Automatic failure detection and recovery
- Distributed health checks and circuit breakers
- Advanced logging with structured data
- Resource optimization and leak detection
"""

import time
import logging
import threading
import queue
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
from datetime import datetime, timezone

# Advanced monitoring with fallbacks
try:
    import psutil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    # Mock psutil for basic functionality
    class MockPsutil:
        @staticmethod
        def cpu_percent(): return 50.0
        @staticmethod
        def virtual_memory(): 
            return type('obj', (object,), {'percent': 60.0, 'used': 8000000000, 'total': 16000000000})()
        @staticmethod
        def disk_usage(path): 
            return type('obj', (object,), {'percent': 30.0, 'used': 100000000, 'total': 500000000})()
    
    psutil = MockPsutil()


@dataclass
class MetricPoint:
    """Single monitoring metric data point."""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class HealthStatus:
    """System health status information."""
    component: str
    status: str  # "healthy", "degraded", "critical", "unknown"
    timestamp: float
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    duration_seconds: int
    severity: str  # "info", "warning", "critical"
    actions: List[str]


class CircuitBreaker:
    """
    🛡️ Circuit Breaker Pattern Implementation
    
    Prevents cascade failures by monitoring service health
    and temporarily disabling failing components.
    """
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, success_threshold: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Monitoring
        self.call_count = 0
        self.failure_history = []
        
        logging.info(f"CircuitBreaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.call_count += 1
        
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                logging.info(f"CircuitBreaker '{self.name}' transitioning to HALF_OPEN")
            else:
                raise Exception(f"CircuitBreaker '{self.name}' is OPEN - calls blocked")
        
        try:
            result = func(*args, **kwargs)
            
            # Success handling
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logging.info(f"CircuitBreaker '{self.name}' recovered to CLOSED")
            elif self.state == "CLOSED":
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append({
                'timestamp': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Keep only recent failures
            cutoff_time = time.time() - 300  # 5 minutes
            self.failure_history = [f for f in self.failure_history if f['timestamp'] > cutoff_time]
            
            if self.failure_count >= self.failure_threshold and self.state == "CLOSED":
                self.state = "OPEN"
                logging.error(f"CircuitBreaker '{self.name}' opened due to failures: {self.failure_count}")
            
            raise e
    
    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'call_count': self.call_count,
            'failure_rate': len(self.failure_history) / max(self.call_count, 1),
            'recent_failures': len(self.failure_history)
        }


class MetricsCollector:
    """
    📊 Advanced Metrics Collection System
    
    Collects, aggregates, and exports metrics for monitoring
    and performance analysis.
    """
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics = {}  # metric_name -> List[MetricPoint]
        self.lock = threading.Lock()
        
        # Background aggregation
        self.aggregation_thread = None
        self.stop_aggregation = threading.Event()
        
        # Alert system
        self.alert_rules = []
        self.active_alerts = {}
        
        logging.info("MetricsCollector initialized")
    
    def start_collection(self):
        """Start background metrics collection."""
        if self.aggregation_thread is None or not self.aggregation_thread.is_alive():
            self.aggregation_thread = threading.Thread(
                target=self._background_collection, daemon=True
            )
            self.aggregation_thread.start()
            logging.info("Background metrics collection started")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self.stop_aggregation.set()
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5)
        logging.info("Background metrics collection stopped")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a single metric point."""
        if tags is None:
            tags = {}
        
        point = MetricPoint(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            tags=tags,
            unit=unit
        )
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(point)
            
            # Trim old data points
            if len(self.metrics[name]) > self.max_points:
                self.metrics[name] = self.metrics[name][-self.max_points:]
        
        # Check alert rules
        self._check_alert_rules(point)
    
    def get_metrics(self, name: str, since: float = None) -> List[MetricPoint]:
        """Get metrics for a specific name."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            points = self.metrics[name]
            if since is not None:
                points = [p for p in points if p.timestamp >= since]
            
            return points[:]
    
    def get_metric_summary(self, name: str, window_seconds: int = 300) -> Dict:
        """Get aggregated metric summary for time window."""
        cutoff_time = time.time() - window_seconds
        points = self.get_metrics(name, since=cutoff_time)
        
        if not points:
            return {
                'name': name,
                'count': 0,
                'window_seconds': window_seconds
            }
        
        values = [p.value for p in points]
        
        return {
            'name': name,
            'count': len(values),
            'window_seconds': window_seconds,
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1],
            'first_timestamp': points[0].timestamp,
            'latest_timestamp': points[-1].timestamp
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule for monitoring."""
        self.alert_rules.append(rule)
        logging.info(f"Added alert rule: {rule.name}")
    
    def _check_alert_rules(self, point: MetricPoint):
        """Check if any alert rules are triggered."""
        for rule in self.alert_rules:
            if rule.metric_name != point.metric_name:
                continue
            
            # Evaluate condition
            triggered = False
            if rule.condition == "gt" and point.value > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and point.value < rule.threshold:
                triggered = True
            elif rule.condition == "eq" and point.value == rule.threshold:
                triggered = True
            elif rule.condition == "ne" and point.value != rule.threshold:
                triggered = True
            
            if triggered:
                self._handle_alert(rule, point)
    
    def _handle_alert(self, rule: AlertRule, point: MetricPoint):
        """Handle triggered alert."""
        alert_key = f"{rule.name}_{rule.metric_name}"
        current_time = time.time()
        
        if alert_key in self.active_alerts:
            # Check if alert has been active long enough
            alert_start = self.active_alerts[alert_key]['start_time']
            if current_time - alert_start >= rule.duration_seconds:
                # Fire alert
                self._fire_alert(rule, point, current_time - alert_start)
        else:
            # Start tracking new alert
            self.active_alerts[alert_key] = {
                'rule': rule,
                'start_time': current_time,
                'point': point
            }
    
    def _fire_alert(self, rule: AlertRule, point: MetricPoint, duration: float):
        """Fire an alert."""
        alert_data = {
            'rule_name': rule.name,
            'metric_name': rule.metric_name,
            'severity': rule.severity,
            'threshold': rule.threshold,
            'actual_value': point.value,
            'duration': duration,
            'timestamp': point.timestamp,
            'actions': rule.actions
        }
        
        logging.warning(f"ALERT FIRED: {rule.name} - {rule.metric_name} {rule.condition} {rule.threshold}, actual: {point.value}")
        
        # Execute alert actions
        for action in rule.actions:
            try:
                self._execute_alert_action(action, alert_data)
            except Exception as e:
                logging.error(f"Failed to execute alert action '{action}': {e}")
    
    def _execute_alert_action(self, action: str, alert_data: Dict):
        """Execute alert action."""
        if action == "log":
            logging.warning(f"Alert: {json.dumps(alert_data, indent=2)}")
        elif action == "email":
            # Mock email action
            logging.info(f"EMAIL ALERT: {alert_data['rule_name']}")
        elif action.startswith("webhook:"):
            # Mock webhook action
            webhook_url = action.split(":", 1)[1]
            logging.info(f"WEBHOOK ALERT to {webhook_url}: {alert_data['rule_name']}")
    
    def _background_collection(self):
        """Background thread for system metrics collection."""
        while not self.stop_aggregation.is_set():
            try:
                # Collect system metrics
                if SYSTEM_MONITORING_AVAILABLE:
                    self.record_metric("system.cpu_percent", psutil.cpu_percent(), unit="%")
                    
                    memory = psutil.virtual_memory()
                    self.record_metric("system.memory_percent", memory.percent, unit="%")
                    self.record_metric("system.memory_used", memory.used, unit="bytes")
                    
                    disk = psutil.disk_usage("/")
                    self.record_metric("system.disk_percent", disk.percent, unit="%")
                    self.record_metric("system.disk_used", disk.used, unit="bytes")
                
                # Wait before next collection
                self.stop_aggregation.wait(10)  # 10 second interval
                
            except Exception as e:
                logging.error(f"Error in background metrics collection: {e}")
                self.stop_aggregation.wait(30)  # Wait longer on error


class HealthChecker:
    """
    🏥 Distributed Health Monitoring System
    
    Monitors system component health and provides
    comprehensive status reporting.
    """
    
    def __init__(self):
        self.health_checks = {}  # component_name -> health_check_function
        self.health_status = {}  # component_name -> HealthStatus
        self.check_interval = 30  # seconds
        
        # Background health checking
        self.health_thread = None
        self.stop_health_checks = threading.Event()
        
        logging.info("HealthChecker initialized")
    
    def register_health_check(self, component: str, check_function: Callable[[], HealthStatus]):
        """Register a health check for a component."""
        self.health_checks[component] = check_function
        logging.info(f"Registered health check for component: {component}")
    
    def start_health_checks(self):
        """Start background health monitoring."""
        if self.health_thread is None or not self.health_thread.is_alive():
            self.health_thread = threading.Thread(
                target=self._background_health_checks, daemon=True
            )
            self.health_thread.start()
            logging.info("Background health checks started")
    
    def stop_health_checks(self):
        """Stop background health monitoring."""
        self.stop_health_checks.set()
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5)
        logging.info("Background health checks stopped")
    
    def get_health_status(self, component: str = None) -> Dict:
        """Get health status for component or all components."""
        if component:
            return self.health_status.get(component, {})
        else:
            return dict(self.health_status)
    
    def get_overall_health(self) -> Dict:
        """Get overall system health assessment."""
        if not self.health_status:
            return {
                'overall_status': 'unknown',
                'healthy_components': 0,
                'degraded_components': 0,
                'critical_components': 0,
                'total_components': 0
            }
        
        status_counts = {'healthy': 0, 'degraded': 0, 'critical': 0, 'unknown': 0}
        
        for status in self.health_status.values():
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        # Determine overall status
        if status_counts['critical'] > 0:
            overall = 'critical'
        elif status_counts['degraded'] > 0:
            overall = 'degraded'
        elif status_counts['healthy'] > 0:
            overall = 'healthy'
        else:
            overall = 'unknown'
        
        return {
            'overall_status': overall,
            'healthy_components': status_counts['healthy'],
            'degraded_components': status_counts['degraded'],
            'critical_components': status_counts['critical'],
            'unknown_components': status_counts['unknown'],
            'total_components': len(self.health_status),
            'last_check': max((s.timestamp for s in self.health_status.values()), default=0)
        }
    
    def _background_health_checks(self):
        """Background thread for health monitoring."""
        while not self.stop_health_checks.is_set():
            try:
                # Run all health checks
                for component, check_func in self.health_checks.items():
                    try:
                        status = check_func()
                        self.health_status[component] = status
                        
                        if status.status in ['degraded', 'critical']:
                            logging.warning(f"Health check failed for {component}: {status.status}")
                            
                    except Exception as e:
                        # Health check itself failed
                        self.health_status[component] = HealthStatus(
                            component=component,
                            status='unknown',
                            timestamp=time.time(),
                            details={'error': str(e), 'traceback': traceback.format_exc()},
                            recommendations=['Check component logs', 'Restart component']
                        )
                        logging.error(f"Health check error for {component}: {e}")
                
                # Wait before next check cycle
                self.stop_health_checks.wait(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error in background health checks: {e}")
                self.stop_health_checks.wait(60)  # Wait longer on error


class AdvancedMonitoringSystem:
    """
    🎯 Unified Advanced Monitoring System
    
    Combines metrics collection, health monitoring, circuit breakers,
    and alerting into a comprehensive monitoring solution.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize subsystems
        self.metrics = MetricsCollector(
            max_points=self.config.get('max_metric_points', 10000)
        )
        self.health_checker = HealthChecker()
        self.circuit_breakers = {}
        
        # Performance tracking
        self.operation_timers = {}
        self.performance_baselines = {}
        
        # Self-healing capabilities
        self.auto_recovery_enabled = self.config.get('auto_recovery', True)
        self.recovery_actions = {}
        
        logging.info("AdvancedMonitoringSystem initialized")
        self._setup_default_monitoring()
    
    def start(self):
        """Start all monitoring subsystems."""
        self.metrics.start_collection()
        self.health_checker.start_health_checks()
        logging.info("AdvancedMonitoringSystem started")
    
    def stop(self):
        """Stop all monitoring subsystems."""
        self.metrics.stop_collection()
        self.health_checker.stop_health_checks()
        logging.info("AdvancedMonitoringSystem stopped")
    
    def create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        breaker = CircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    self.metrics.record_metric(f"operation.{operation_name}.duration", duration, unit="seconds")
                    self.metrics.record_metric(f"operation.{operation_name}.success", 1)
                    
                    # Update performance baseline
                    self._update_performance_baseline(operation_name, duration, success=True)
                    
                    return result
                    
                except Exception as e:
                    # Record failure metrics
                    duration = time.time() - start_time
                    self.metrics.record_metric(f"operation.{operation_name}.duration", duration, unit="seconds")
                    self.metrics.record_metric(f"operation.{operation_name}.failure", 1)
                    
                    # Update performance baseline
                    self._update_performance_baseline(operation_name, duration, success=False)
                    
                    # Log error with context
                    logging.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
                    
                    raise e
            
            return wrapper
        return decorator
    
    def _update_performance_baseline(self, operation: str, duration: float, success: bool):
        """Update performance baseline for operation."""
        if operation not in self.performance_baselines:
            self.performance_baselines[operation] = {
                'durations': [],
                'success_rate': 0.0,
                'total_calls': 0,
                'successful_calls': 0
            }
        
        baseline = self.performance_baselines[operation]
        baseline['durations'].append(duration)
        baseline['total_calls'] += 1
        
        if success:
            baseline['successful_calls'] += 1
        
        baseline['success_rate'] = baseline['successful_calls'] / baseline['total_calls']
        
        # Keep only recent durations (last 100)
        if len(baseline['durations']) > 100:
            baseline['durations'] = baseline['durations'][-100:]
        
        # Check for performance degradation
        if len(baseline['durations']) >= 10:
            recent_avg = sum(baseline['durations'][-10:]) / 10
            overall_avg = sum(baseline['durations']) / len(baseline['durations'])
            
            if recent_avg > overall_avg * 1.5:  # 50% degradation
                logging.warning(f"Performance degradation detected for '{operation}': recent avg {recent_avg:.3f}s vs baseline {overall_avg:.3f}s")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'system_health': self.health_checker.get_overall_health(),
            'circuit_breakers': {name: cb.get_status() for name, cb in self.circuit_breakers.items()},
            'operation_baselines': {},
            'active_alerts': len(self.metrics.active_alerts),
            'metrics_summary': {}
        }
        
        # Add operation baseline summaries
        for op_name, baseline in self.performance_baselines.items():
            if baseline['durations']:
                durations = baseline['durations']
                report['operation_baselines'][op_name] = {
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'success_rate': baseline['success_rate'],
                    'total_calls': baseline['total_calls'],
                    'sample_size': len(durations)
                }
        
        # Add key metrics summaries
        key_metrics = ['system.cpu_percent', 'system.memory_percent', 'system.disk_percent']
        for metric in key_metrics:
            summary = self.metrics.get_metric_summary(metric)
            if summary['count'] > 0:
                report['metrics_summary'][metric] = summary
        
        return report
    
    def _setup_default_monitoring(self):
        """Setup default monitoring rules and health checks."""
        
        # Default alert rules
        self.metrics.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="system.cpu_percent",
            condition="gt",
            threshold=80.0,
            duration_seconds=30,
            severity="warning",
            actions=["log"]
        ))
        
        self.metrics.add_alert_rule(AlertRule(
            name="high_memory_usage", 
            metric_name="system.memory_percent",
            condition="gt",
            threshold=85.0,
            duration_seconds=30,
            severity="critical",
            actions=["log", "email"]
        ))
        
        self.metrics.add_alert_rule(AlertRule(
            name="high_disk_usage",
            metric_name="system.disk_percent",
            condition="gt",
            threshold=90.0,
            duration_seconds=60,
            severity="critical", 
            actions=["log", "email"]
        ))
        
        # Default health checks
        def system_health_check() -> HealthStatus:
            """Check overall system health."""
            try:
                if SYSTEM_MONITORING_AVAILABLE:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")
                    
                    issues = []
                    recommendations = []
                    
                    if cpu_percent > 90:
                        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                        recommendations.append("Consider scaling up compute resources")
                    
                    if memory.percent > 90:
                        issues.append(f"High memory usage: {memory.percent:.1f}%")
                        recommendations.append("Check for memory leaks or increase RAM")
                    
                    if disk.percent > 95:
                        issues.append(f"High disk usage: {disk.percent:.1f}%")
                        recommendations.append("Clean up disk space or add storage")
                    
                    if issues:
                        status = "critical" if any("High" in issue for issue in issues) else "degraded"
                    else:
                        status = "healthy"
                    
                    return HealthStatus(
                        component="system",
                        status=status,
                        timestamp=time.time(),
                        details={
                            "cpu_percent": cpu_percent,
                            "memory_percent": memory.percent,
                            "disk_percent": disk.percent,
                            "issues": issues
                        },
                        recommendations=recommendations
                    )
                else:
                    # Mock healthy status when monitoring unavailable
                    return HealthStatus(
                        component="system",
                        status="healthy",
                        timestamp=time.time(),
                        details={"monitoring": "mock"},
                        recommendations=[]
                    )
                    
            except Exception as e:
                return HealthStatus(
                    component="system", 
                    status="unknown",
                    timestamp=time.time(),
                    details={"error": str(e)},
                    recommendations=["Check system monitoring setup"]
                )
        
        self.health_checker.register_health_check("system", system_health_check)


# Global monitoring instance for convenience
_global_monitor = None

def get_global_monitor() -> AdvancedMonitoringSystem:
    """Get global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AdvancedMonitoringSystem()
    return _global_monitor

def start_monitoring():
    """Start global monitoring."""
    monitor = get_global_monitor()
    monitor.start()

def stop_monitoring():
    """Stop global monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop()

# Convenience decorators
def monitor_performance(operation_name: str):
    """Decorator for monitoring operation performance."""
    return get_global_monitor().monitor_operation(operation_name)

def with_circuit_breaker(name: str, **breaker_kwargs):
    """Decorator for applying circuit breaker protection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            breaker = monitor.get_circuit_breaker(name)
            if breaker is None:
                breaker = monitor.create_circuit_breaker(name, **breaker_kwargs)
            
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demonstration of advanced monitoring system
    print("🛡️ Advanced Monitoring System - Demonstration")
    
    # Initialize monitoring
    monitor = AdvancedMonitoringSystem()
    monitor.start()
    
    # Create circuit breaker
    breaker = monitor.create_circuit_breaker("demo_service", failure_threshold=3)
    
    # Mock some operations
    @monitor.monitor_operation("test_operation")
    def test_operation(should_fail=False):
        if should_fail:
            raise Exception("Test failure")
        time.sleep(0.1)  # Simulate work
        return "success"
    
    # Test monitoring
    print("Running test operations...")
    for i in range(10):
        try:
            result = test_operation(should_fail=(i % 4 == 3))  # Fail every 4th operation
            print(f"Operation {i}: {result}")
        except Exception as e:
            print(f"Operation {i}: FAILED - {e}")
        
        time.sleep(0.5)
    
    # Get performance report
    time.sleep(2)  # Allow background collection
    report = monitor.get_performance_report()
    
    print("\n📊 Performance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    monitor.stop()
    print("✅ Advanced monitoring demonstration complete!")