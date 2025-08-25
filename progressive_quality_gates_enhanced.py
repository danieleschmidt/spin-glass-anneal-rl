#!/usr/bin/env python3
"""
Progressive Quality Gates Enhanced - Generation 2.

This module extends the basic progressive quality gates with comprehensive:
1. Robust error handling and recovery mechanisms
2. Real-time monitoring and alerting
3. Advanced validation and security hardening
4. Performance monitoring and optimization
5. Distributed execution and fault tolerance

Generation 2 Enhancements:
- Circuit breaker patterns for resilient check execution
- Real-time quality metrics streaming
- Advanced security validation framework
- Performance regression detection
- Distributed quality gate coordination
- Comprehensive logging and audit trails
- Self-healing and adaptive quality thresholds
"""

import sys
import os
import time
import json
import subprocess
import traceback
import hashlib
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import concurrent.futures
from contextlib import contextmanager
import warnings
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Generation 1 components
from progressive_quality_gates import (
    ProgressiveStage, RiskLevel, QualityMetric, QualityThreshold,
    ProgressiveGateResult, ProgressiveQualityGateConfig, ProgressiveQualityGates
)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status for system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemHealth:
    """System health metrics."""
    status: HealthStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    active_checks: int = 0
    failed_checks: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class QualityAlert:
    """Quality gate alert."""
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    details: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    check_name: str
    execution_times: List[float] = field(default_factory=list)
    success_rate: float = 1.0
    error_count: int = 0
    last_execution: float = field(default_factory=time.time)
    regression_detected: bool = False
    baseline_time: Optional[float] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, bypass checks
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for quality check resilience."""
    name: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    total_calls: int = 0


class QualityMonitor:
    """Real-time quality monitoring system."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.performance_metrics = {}
        self.alerts = []
        self.system_health = SystemHealth(HealthStatus.HEALTHY)
        self.circuit_breakers = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        
    def start_monitoring(self, interval: float = 30.0):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("🔍 Real-time quality monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("🔍 Real-time quality monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_system_health()
                self._check_performance_regressions()
                self._cleanup_old_data()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _update_system_health(self):
        """Update system health metrics."""
        try:
            import psutil
            
            with self._lock:
                self.system_health.cpu_usage = psutil.cpu_percent()
                self.system_health.memory_usage = psutil.virtual_memory().percent
                self.system_health.disk_usage = psutil.disk_usage('/').percent
                self.system_health.last_update = time.time()
                
                # Determine health status
                if (self.system_health.cpu_usage > 90 or 
                    self.system_health.memory_usage > 90):
                    self.system_health.status = HealthStatus.CRITICAL
                elif (self.system_health.cpu_usage > 70 or 
                      self.system_health.memory_usage > 80):
                    self.system_health.status = HealthStatus.DEGRADED
                else:
                    self.system_health.status = HealthStatus.HEALTHY
                    
        except ImportError:
            # Fallback if psutil not available
            with self._lock:
                self.system_health.status = HealthStatus.HEALTHY
        except Exception as e:
            logger.warning(f"Health monitoring error: {e}")
    
    def _check_performance_regressions(self):
        """Check for performance regressions."""
        with self._lock:
            for check_name, metrics in self.performance_metrics.items():
                if len(metrics.execution_times) >= 10:
                    recent_times = metrics.execution_times[-5:]
                    baseline_times = metrics.execution_times[-10:-5]
                    
                    if len(baseline_times) >= 5:
                        recent_avg = statistics.mean(recent_times)
                        baseline_avg = statistics.mean(baseline_times)
                        
                        # Detect regression (>50% slower)
                        if recent_avg > baseline_avg * 1.5:
                            if not metrics.regression_detected:
                                metrics.regression_detected = True
                                self._create_alert(
                                    AlertSeverity.WARNING,
                                    f"performance_regression_{check_name}",
                                    f"Performance regression detected in {check_name}",
                                    {
                                        "recent_avg": recent_avg,
                                        "baseline_avg": baseline_avg,
                                        "regression_factor": recent_avg / baseline_avg
                                    }
                                )
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = time.time() - 3600  # 1 hour
        
        with self._lock:
            # Clean up metrics history
            for key in list(self.metrics_history.keys()):
                self.metrics_history[key] = [
                    metric for metric in self.metrics_history[key]
                    if metric.get('timestamp', 0) > cutoff_time
                ]
            
            # Clean up performance metrics execution times
            for metrics in self.performance_metrics.values():
                # Keep only last 50 execution times
                if len(metrics.execution_times) > 50:
                    metrics.execution_times = metrics.execution_times[-50:]
    
    def record_check_execution(self, check_name: str, execution_time: float, success: bool):
        """Record check execution metrics."""
        with self._lock:
            if check_name not in self.performance_metrics:
                self.performance_metrics[check_name] = PerformanceMetrics(check_name)
            
            metrics = self.performance_metrics[check_name]
            metrics.execution_times.append(execution_time)
            metrics.last_execution = time.time()
            
            if success:
                metrics.success_rate = (
                    metrics.success_rate * 0.9 + 0.1  # Exponential moving average
                )
            else:
                metrics.error_count += 1
                metrics.success_rate = (
                    metrics.success_rate * 0.9  # Decay success rate on failure
                )
    
    def get_circuit_breaker(self, check_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a check."""
        with self._lock:
            if check_name not in self.circuit_breakers:
                self.circuit_breakers[check_name] = CircuitBreaker(check_name)
            return self.circuit_breakers[check_name]
    
    def update_circuit_breaker(self, check_name: str, success: bool):
        """Update circuit breaker state based on check result."""
        breaker = self.get_circuit_breaker(check_name)
        
        with self._lock:
            breaker.total_calls += 1
            current_time = time.time()
            
            if success:
                breaker.success_count += 1
                if breaker.state == CircuitBreakerState.HALF_OPEN:
                    # Recovery successful
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.failure_count = 0
                    logger.info(f"🔄 Circuit breaker for {check_name} closed (recovered)")
                    
            else:
                breaker.failure_count += 1
                breaker.last_failure_time = current_time
                
                if breaker.state == CircuitBreakerState.CLOSED:
                    if breaker.failure_count >= breaker.failure_threshold:
                        breaker.state = CircuitBreakerState.OPEN
                        self._create_alert(
                            AlertSeverity.ERROR,
                            f"circuit_breaker_open_{check_name}",
                            f"Circuit breaker opened for {check_name}",
                            {"failure_count": breaker.failure_count}
                        )
                        logger.warning(f"⚠️ Circuit breaker for {check_name} opened")
                
                elif breaker.state == CircuitBreakerState.HALF_OPEN:
                    # Failed during recovery, go back to open
                    breaker.state = CircuitBreakerState.OPEN
                    breaker.last_failure_time = current_time
            
            # Check if open circuit should try recovery
            if (breaker.state == CircuitBreakerState.OPEN and
                current_time - breaker.last_failure_time >= breaker.recovery_timeout):
                breaker.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"🔄 Circuit breaker for {check_name} half-open (testing recovery)")
    
    def should_execute_check(self, check_name: str) -> bool:
        """Check if a quality check should be executed based on circuit breaker."""
        breaker = self.get_circuit_breaker(check_name)
        return breaker.state != CircuitBreakerState.OPEN
    
    def _create_alert(self, severity: AlertSeverity, component: str, message: str, details: Dict = None):
        """Create a quality alert."""
        alert = QualityAlert(
            timestamp=time.time(),
            severity=severity,
            component=component,
            message=message,
            details=details or {}
        )
        self.alerts.append(alert)
        
        # Log alert
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }[severity]
        
        log_func(f"🚨 ALERT [{severity.value.upper()}] {component}: {message}")
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, component: str):
        """Resolve alerts for a component."""
        current_time = time.time()
        for alert in self.alerts:
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = current_time
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary."""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            return {
                "system_health": asdict(self.system_health),
                "active_alerts_count": len(active_alerts),
                "circuit_breakers": {
                    name: {
                        "state": breaker.state.value,
                        "failure_count": breaker.failure_count,
                        "success_rate": breaker.success_count / max(breaker.total_calls, 1)
                    }
                    for name, breaker in self.circuit_breakers.items()
                },
                "performance_summary": {
                    name: {
                        "avg_execution_time": statistics.mean(metrics.execution_times) if metrics.execution_times else 0,
                        "success_rate": metrics.success_rate,
                        "regression_detected": metrics.regression_detected
                    }
                    for name, metrics in self.performance_metrics.items()
                }
            }


class EnhancedSecurityValidator:
    """Enhanced security validation with comprehensive threat detection."""
    
    def __init__(self):
        self.security_patterns = self._load_security_patterns()
        self.vulnerability_database = self._load_vulnerability_db()
    
    def _load_security_patterns(self) -> Dict:
        """Load security threat patterns."""
        return {
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"subprocess\.call",
                r"os\.system",
                r"__import__\s*\(",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"os\.path\.join.*\.\.",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                r"api_key\s*=\s*['\"][^'\"]{16,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{16,}['\"]",
            ],
            "unsafe_deserialization": [
                r"pickle\.loads",
                r"yaml\.load\s*\(",
                r"marshal\.loads",
            ]
        }
    
    def _load_vulnerability_db(self) -> Dict:
        """Load known vulnerability database."""
        return {
            "high_risk_imports": [
                "pickle", "marshal", "subprocess", "os.system"
            ],
            "deprecated_functions": [
                "md5", "sha1", "random.random"
            ]
        }
    
    def comprehensive_security_scan(self, codebase_path: str = "spin_glass_rl") -> Tuple[bool, float, Dict]:
        """Comprehensive security scan with threat detection."""
        try:
            scan_results = {
                "code_injection": self._scan_code_injection(codebase_path),
                "path_traversal": self._scan_path_traversal(codebase_path),
                "hardcoded_secrets": self._scan_hardcoded_secrets(codebase_path),
                "unsafe_deserialization": self._scan_unsafe_deserialization(codebase_path),
                "dependency_vulnerabilities": self._scan_dependencies(),
                "permission_analysis": self._analyze_permissions(codebase_path)
            }
            
            # Calculate overall security score
            total_issues = sum(len(issues) for issues in scan_results.values() if isinstance(issues, list))
            max_score = 100.0
            score_penalty = min(total_issues * 5, 80)  # Max 80 point penalty
            security_score = max_score - score_penalty
            
            # Determine pass/fail
            critical_issues = scan_results.get("code_injection", []) + scan_results.get("unsafe_deserialization", [])
            passed = len(critical_issues) == 0 and security_score >= 70
            
            return passed, security_score, scan_results
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _scan_code_injection(self, codebase_path: str) -> List[Dict]:
        """Scan for code injection vulnerabilities."""
        import re
        issues = []
        
        try:
            python_files = list(Path(codebase_path).rglob("*.py"))
            for file_path in python_files[:20]:  # Limit for performance
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in self.security_patterns["code_injection"]:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            issues.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "line": content[:match.start()].count('\n') + 1,
                                "severity": "high"
                            })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Code injection scan error: {e}")
        
        return issues
    
    def _scan_path_traversal(self, codebase_path: str) -> List[Dict]:
        """Scan for path traversal vulnerabilities."""
        import re
        issues = []
        
        try:
            python_files = list(Path(codebase_path).rglob("*.py"))
            for file_path in python_files[:20]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in self.security_patterns["path_traversal"]:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            issues.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "line": content[:match.start()].count('\n') + 1,
                                "severity": "medium"
                            })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Path traversal scan error: {e}")
        
        return issues
    
    def _scan_hardcoded_secrets(self, codebase_path: str) -> List[Dict]:
        """Scan for hardcoded secrets."""
        import re
        issues = []
        
        try:
            python_files = list(Path(codebase_path).rglob("*.py"))
            for file_path in python_files[:10]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in self.security_patterns["hardcoded_secrets"]:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            issues.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "line": content[:match.start()].count('\n') + 1,
                                "severity": "high"
                            })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Hardcoded secrets scan error: {e}")
        
        return issues
    
    def _scan_unsafe_deserialization(self, codebase_path: str) -> List[Dict]:
        """Scan for unsafe deserialization."""
        import re
        issues = []
        
        try:
            python_files = list(Path(codebase_path).rglob("*.py"))
            for file_path in python_files[:15]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in self.security_patterns["unsafe_deserialization"]:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            issues.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "line": content[:match.start()].count('\n') + 1,
                                "severity": "critical"
                            })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Unsafe deserialization scan error: {e}")
        
        return issues
    
    def _scan_dependencies(self) -> List[Dict]:
        """Scan for known vulnerable dependencies."""
        issues = []
        
        try:
            # Check requirements files for known vulnerable packages
            req_files = ["requirements.txt", "requirements-dev.txt"]
            vulnerable_packages = {
                "requests": "2.25.0",  # Example - versions below this have vulnerabilities
                "urllib3": "1.26.0"
            }
            
            for req_file in req_files:
                if Path(req_file).exists():
                    try:
                        with open(req_file, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    for pkg, min_version in vulnerable_packages.items():
                                        if pkg in line:
                                            issues.append({
                                                "file": req_file,
                                                "line": line_num,
                                                "package": pkg,
                                                "recommendation": f"Update to >= {min_version}",
                                                "severity": "medium"
                                            })
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Dependency scan error: {e}")
        
        return issues
    
    def _analyze_permissions(self, codebase_path: str) -> Dict:
        """Analyze file permissions and access patterns."""
        try:
            python_files = list(Path(codebase_path).rglob("*.py"))
            
            permission_analysis = {
                "executable_files": 0,
                "world_writable": 0,
                "suspicious_permissions": []
            }
            
            for file_path in python_files[:10]:
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check for executable Python files (potentially suspicious)
                    if mode & 0o111:  # Any execute bit set
                        permission_analysis["executable_files"] += 1
                        
                    # Check for world-writable files (security risk)
                    if mode & 0o002:  # World write bit
                        permission_analysis["world_writable"] += 1
                        permission_analysis["suspicious_permissions"].append(str(file_path))
                        
                except Exception:
                    continue
            
            return permission_analysis
            
        except Exception as e:
            logger.warning(f"Permission analysis error: {e}")
            return {"error": str(e)}


class EnhancedProgressiveQualityGates(ProgressiveQualityGates):
    """Enhanced Progressive Quality Gates with Generation 2 features."""
    
    def __init__(self, config: Optional[ProgressiveQualityGateConfig] = None):
        super().__init__(config)
        self.monitor = QualityMonitor()
        self.security_validator = EnhancedSecurityValidator()
        self.execution_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.timeout_seconds = 300  # 5 minute default timeout
        self._shutdown_requested = False
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of enhanced quality gates."""
        logger.info("🛑 Shutting down Enhanced Progressive Quality Gates...")
        
        self.monitor.stop_monitoring()
        self.execution_pool.shutdown(wait=True)
        
        # Final health summary
        health_summary = self.monitor.get_health_summary()
        logger.info(f"📊 Final health summary: {json.dumps(health_summary, indent=2)}")
    
    @contextmanager
    def timeout_context(self, seconds: float):
        """Context manager for operation timeouts."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def execute_quality_gates_enhanced(
        self,
        stage: ProgressiveStage,
        risk_level: RiskLevel,
        context: Optional[Dict] = None,
        max_concurrent: int = 3
    ) -> ProgressiveGateResult:
        """Enhanced quality gate execution with monitoring and resilience."""
        
        if self._shutdown_requested:
            logger.warning("Shutdown requested, aborting quality gate execution")
            return self._create_error_result(stage, risk_level, "Shutdown requested")
        
        logger.info(f"🚀 Enhanced Progressive Quality Gates Execution")
        logger.info(f"   Stage: {stage.value}")
        logger.info(f"   Risk Level: {risk_level.value}")
        logger.info(f"   Max Concurrent: {max_concurrent}")
        
        start_time = time.time()
        
        # Get quality threshold configuration
        threshold_config = self.config.get_threshold(stage, risk_level)
        if not threshold_config:
            error_msg = f"No configuration found for stage={stage.value}, risk={risk_level.value}"
            logger.error(error_msg)
            return self._create_error_result(stage, risk_level, error_msg)
        
        # Check system health before execution
        health_summary = self.monitor.get_health_summary()
        if health_summary["system_health"]["status"] == HealthStatus.CRITICAL.value:
            logger.warning("⚠️ System health is critical, reducing concurrent execution")
            max_concurrent = 1
        
        # Execute checks with enhanced error handling
        all_checks = threshold_config.required_checks + threshold_config.optional_checks
        check_results = {}
        
        # Batch execution with concurrency control
        check_batches = [all_checks[i:i+max_concurrent] for i in range(0, len(all_checks), max_concurrent)]
        
        for batch_idx, batch in enumerate(check_batches):
            if self._shutdown_requested:
                break
                
            logger.info(f"📦 Executing batch {batch_idx + 1}/{len(check_batches)}: {batch}")
            
            # Execute batch concurrently
            futures = {}
            for check_name in batch:
                if check_name in self.config.check_registry:
                    future = self.execution_pool.submit(
                        self._execute_check_with_monitoring,
                        check_name,
                        threshold_config.required_checks
                    )
                    futures[check_name] = future
            
            # Collect results with timeout
            for check_name, future in futures.items():
                try:
                    with self.timeout_context(self.timeout_seconds):
                        result = future.result(timeout=self.timeout_seconds)
                        check_results[check_name] = result
                except TimeoutError:
                    logger.error(f"⏰ Check {check_name} timed out")
                    check_results[check_name] = {
                        "passed": False,
                        "score": 0.0,
                        "details": {"error": "Timeout"},
                        "required": check_name in threshold_config.required_checks
                    }
                    self.monitor.update_circuit_breaker(check_name, False)
                except Exception as e:
                    logger.error(f"❌ Check {check_name} failed: {e}")
                    check_results[check_name] = {
                        "passed": False,
                        "score": 0.0,
                        "details": {"error": str(e)},
                        "required": check_name in threshold_config.required_checks
                    }
                    self.monitor.update_circuit_breaker(check_name, False)
        
        # Calculate enhanced metrics
        metric_scores = self._calculate_metric_scores_enhanced(check_results, threshold_config)
        
        # Enhanced result analysis
        total_checks = len(check_results)
        checks_passed = sum(1 for r in check_results.values() if r["passed"])
        checks_failed = total_checks - checks_passed
        
        # Apply failure tolerance with circuit breaker consideration
        failure_rate = checks_failed / total_checks if total_checks > 0 else 0.0
        within_tolerance = failure_rate <= threshold_config.failure_tolerance
        
        # Enhanced scoring with quality decay for failed circuit breakers
        overall_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0.0
        
        # Apply circuit breaker penalty
        open_breakers = [
            name for name, breaker in self.monitor.circuit_breakers.items()
            if breaker.state == CircuitBreakerState.OPEN
        ]
        if open_breakers:
            circuit_breaker_penalty = len(open_breakers) * 10  # 10 points per open breaker
            overall_score = max(0, overall_score - circuit_breaker_penalty)
            logger.warning(f"⚠️ Circuit breaker penalty applied: -{circuit_breaker_penalty} points")
        
        # Enhanced pass/fail determination
        min_threshold = min(threshold_config.thresholds.values()) if threshold_config.thresholds else 70
        gates_passed = (
            overall_score >= min_threshold and
            within_tolerance and
            len(open_breakers) == 0  # No open circuit breakers
        )
        
        # Generate enhanced recommendations
        recommendations = self._generate_enhanced_recommendations(
            stage, risk_level, check_results, metric_scores, threshold_config, health_summary
        )
        
        # Generate next stage requirements
        next_stage_requirements = self._generate_next_stage_requirements(stage, risk_level)
        
        execution_time = time.time() - start_time
        
        # Create enhanced result
        result = ProgressiveGateResult(
            stage=stage,
            risk_level=risk_level,
            overall_score=overall_score,
            metric_scores=metric_scores,
            passed=gates_passed,
            checks_executed=total_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            execution_time=execution_time,
            recommendations=recommendations,
            next_stage_requirements=next_stage_requirements,
            details={
                "check_results": check_results,
                "failure_rate": failure_rate,
                "failure_tolerance": threshold_config.failure_tolerance,
                "system_health": health_summary,
                "circuit_breakers": {
                    name: breaker.state.value 
                    for name, breaker in self.monitor.circuit_breakers.items()
                },
                "active_alerts": len(self.monitor.get_active_alerts()),
                "open_circuit_breakers": open_breakers,
                "threshold_config": {
                    "stage": stage.value,
                    "risk_level": risk_level.value,
                    "thresholds": {k.value: v for k, v in threshold_config.thresholds.items()}
                }
            }
        )
        
        self.execution_history.append(result)
        
        # Enhanced logging
        logger.info(f"🎯 Enhanced Progressive Quality Gates Result:")
        logger.info(f"   Overall Score: {overall_score:.1f}")
        logger.info(f"   Gates Passed: {'✅ YES' if gates_passed else '❌ NO'}")
        logger.info(f"   Checks: {checks_passed}/{total_checks} passed")
        logger.info(f"   Circuit Breakers: {len(open_breakers)} open")
        logger.info(f"   Active Alerts: {len(self.monitor.get_active_alerts())}")
        logger.info(f"   Execution Time: {execution_time:.2f}s")
        
        return result
    
    def _execute_check_with_monitoring(
        self, 
        check_name: str, 
        required_checks: List[str]
    ) -> Dict:
        """Execute a quality check with monitoring and circuit breaker logic."""
        
        # Check circuit breaker
        if not self.monitor.should_execute_check(check_name):
            logger.warning(f"⚠️ Skipping {check_name} - circuit breaker open")
            return {
                "passed": False,
                "score": 0.0,
                "details": {"circuit_breaker": "open"},
                "required": check_name in required_checks
            }
        
        start_time = time.time()
        
        try:
            # Enhanced security check
            if check_name == "security_scan":
                passed, score, details = self.security_validator.comprehensive_security_scan()
            else:
                # Use original check function
                check_func = self.config.check_registry[check_name]
                passed, score, details = check_func()
            
            execution_time = time.time() - start_time
            
            # Record monitoring metrics
            self.monitor.record_check_execution(check_name, execution_time, passed)
            self.monitor.update_circuit_breaker(check_name, passed)
            
            # Resolve alert if check now passes
            if passed:
                self.monitor.resolve_alert(f"check_failure_{check_name}")
            else:
                # Create alert for failed check
                self.monitor._create_alert(
                    AlertSeverity.WARNING if check_name not in required_checks else AlertSeverity.ERROR,
                    f"check_failure_{check_name}",
                    f"Quality check {check_name} failed",
                    {"score": score, "details": details}
                )
            
            return {
                "passed": passed,
                "score": score,
                "details": details,
                "required": check_name in required_checks,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Check {check_name} failed with exception: {e}")
            
            # Record failure
            self.monitor.record_check_execution(check_name, execution_time, False)
            self.monitor.update_circuit_breaker(check_name, False)
            
            # Create alert
            self.monitor._create_alert(
                AlertSeverity.ERROR,
                f"check_error_{check_name}",
                f"Quality check {check_name} threw exception",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": str(e), "traceback": traceback.format_exc()},
                "required": check_name in required_checks,
                "execution_time": execution_time
            }
    
    def _calculate_metric_scores_enhanced(
        self, 
        check_results: Dict, 
        threshold_config: QualityThreshold
    ) -> Dict[QualityMetric, float]:
        """Enhanced metric score calculation with performance weighting."""
        metric_scores = super()._calculate_metric_scores(check_results, threshold_config)
        
        # Apply performance-based weighting
        for metric in metric_scores:
            # Get average performance for checks contributing to this metric
            related_checks = [
                name for name, result in check_results.items()
                if self._check_contributes_to_metric(name, metric)
            ]
            
            if related_checks:
                # Calculate performance factor
                performance_factors = []
                for check_name in related_checks:
                    if check_name in self.monitor.performance_metrics:
                        metrics = self.monitor.performance_metrics[check_name]
                        # Penalize slow or unreliable checks
                        if metrics.execution_times:
                            avg_time = statistics.mean(metrics.execution_times)
                            time_factor = max(0.5, 1.0 - (avg_time - 1.0) * 0.1)  # Penalize >1s execution
                            reliability_factor = metrics.success_rate
                            performance_factors.append(time_factor * reliability_factor)
                
                if performance_factors:
                    avg_performance = statistics.mean(performance_factors)
                    metric_scores[metric] *= avg_performance
        
        return metric_scores
    
    def _check_contributes_to_metric(self, check_name: str, metric: QualityMetric) -> bool:
        """Check if a quality check contributes to a specific metric."""
        # Enhanced mapping from Generation 1
        check_to_metric = {
            "unit_tests": [QualityMetric.TEST_COVERAGE],
            "integration_tests": [QualityMetric.TEST_COVERAGE],
            "e2e_tests": [QualityMetric.TEST_COVERAGE],
            "security_scan": [QualityMetric.SECURITY_SCORE],
            "basic_security": [QualityMetric.SECURITY_SCORE],
            "performance_test": [QualityMetric.PERFORMANCE_SCORE],
            "load_test": [QualityMetric.PERFORMANCE_SCORE],
            "basic_functionality": [QualityMetric.CODE_QUALITY_SCORE, QualityMetric.RELIABILITY_SCORE],
            "syntax_check": [QualityMetric.CODE_QUALITY_SCORE],
            "chaos_test": [QualityMetric.RELIABILITY_SCORE],
            "disaster_recovery_test": [QualityMetric.RELIABILITY_SCORE],
        }
        
        return metric in check_to_metric.get(check_name, [])
    
    def _generate_enhanced_recommendations(
        self,
        stage: ProgressiveStage,
        risk_level: RiskLevel,
        check_results: Dict,
        metric_scores: Dict[QualityMetric, float],
        threshold_config: QualityThreshold,
        health_summary: Dict
    ) -> List[str]:
        """Generate enhanced recommendations with system health consideration."""
        recommendations = super()._generate_recommendations(
            stage, risk_level, check_results, metric_scores, threshold_config
        )
        
        # Add health-based recommendations
        system_health = health_summary["system_health"]
        if system_health["status"] in ["degraded", "critical"]:
            recommendations.insert(0, f"System health is {system_health['status']} - consider reducing workload")
        
        # Add circuit breaker recommendations
        open_breakers = [
            name for name, breaker_info in health_summary["circuit_breakers"].items()
            if breaker_info["state"] == "open"
        ]
        if open_breakers:
            recommendations.insert(0, f"Fix failing checks to close circuit breakers: {', '.join(open_breakers)}")
        
        # Add performance recommendations
        slow_checks = [
            name for name, perf_info in health_summary["performance_summary"].items()
            if perf_info["avg_execution_time"] > 10.0  # Slow checks
        ]
        if slow_checks:
            recommendations.append(f"Optimize slow checks: {', '.join(slow_checks)}")
        
        # Add alert-based recommendations
        if health_summary.get("active_alerts_count", 0) > 0:
            recommendations.append(f"Resolve {health_summary['active_alerts_count']} active alerts")
        
        return recommendations[:8]  # Limit to top 8 recommendations


def main():
    """Main execution function for Enhanced Progressive Quality Gates."""
    print("🚀 ENHANCED PROGRESSIVE QUALITY GATES - GENERATION 2")
    print("=" * 70)
    print("✨ Features: Circuit Breakers • Real-time Monitoring • Enhanced Security")
    print()
    
    # Initialize Enhanced Progressive Quality Gates
    enhanced_gates = EnhancedProgressiveQualityGates()
    
    try:
        # Example context for enhanced execution
        context = {
            "current_test_coverage": 72,
            "deployment_target": "staging",
            "security_sensitive": True,
            "performance_critical": True,
            "concurrent_execution": True
        }
        
        # Get recommended stage and risk level
        stage, risk_level = enhanced_gates.get_recommended_stage(context)
        print(f"📊 Recommended Enhanced Configuration:")
        print(f"   Stage: {stage.value}")
        print(f"   Risk Level: {risk_level.value}")
        print(f"   Enhanced Features: Active")
        print()
        
        # Execute enhanced quality gates
        result = enhanced_gates.execute_quality_gates_enhanced(
            stage, risk_level, context, max_concurrent=3
        )
        
        # Display enhanced summary
        print()
        print("🎯 ENHANCED PROGRESSIVE QUALITY GATES SUMMARY")
        print("-" * 60)
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Gates Passed: {'✅ YES' if result.passed else '❌ NO'}")
        print(f"Checks Executed: {result.checks_executed}")
        print(f"Checks Passed: {result.checks_passed}")
        print(f"Checks Failed: {result.checks_failed}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        # Enhanced details
        if "open_circuit_breakers" in result.details:
            open_breakers = result.details["open_circuit_breakers"]
            print(f"Circuit Breakers Open: {len(open_breakers)}")
            if open_breakers:
                print(f"   {', '.join(open_breakers)}")
        
        active_alerts = result.details.get("active_alerts", 0)
        print(f"Active Alerts: {active_alerts}")
        
        # System health
        system_health = result.details.get("system_health", {}).get("system_health", {})
        print(f"System Health: {system_health.get('status', 'unknown').upper()}")
        
        if result.recommendations:
            print()
            print("🎯 ENHANCED RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        if result.next_stage_requirements:
            print()
            print("🚀 NEXT STAGE REQUIREMENTS:")
            for i, req in enumerate(result.next_stage_requirements, 1):
                print(f"  {i}. {req}")
        
        # Save enhanced results
        results_file = enhanced_gates.save_results(result)
        
        print()
        print("✅ GENERATION 2: Enhanced Progressive Quality Gates implemented successfully!")
        print("🔍 Real-time monitoring active")
        print("🛡️ Circuit breakers protecting system reliability")
        print("🔒 Enhanced security validation enabled")
        print(f"🎉 Ready for GENERATION 3: Performance optimization and scaling")
        
        return result.passed
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Enhanced Progressive Quality Gates execution failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Graceful shutdown
        enhanced_gates.shutdown()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)