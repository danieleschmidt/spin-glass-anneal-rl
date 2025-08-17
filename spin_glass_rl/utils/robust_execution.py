"""Robust execution framework with error handling and monitoring."""

import sys
import traceback
import time
import functools
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import json

# Simplified logging without conflicts
class SimpleLogger:
    """Simple logger without external dependencies."""
    
    def __init__(self, name="SimpleLogger", level="INFO"):
        self.name = name
        self.level = level
    
    def _log(self, level, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level} - {self.name} - {message}")
    
    def debug(self, message):
        if self.level in ["DEBUG"]:
            self._log("DEBUG", message)
    
    def info(self, message):
        if self.level in ["DEBUG", "INFO"]:
            self._log("INFO", message)
    
    def warning(self, message):
        if self.level in ["DEBUG", "INFO", "WARNING"]:
            self._log("WARNING", message)
    
    def error(self, message):
        if self.level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self._log("ERROR", message)
    
    def critical(self, message):
        self._log("CRITICAL", message)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionResult:
    """Result of robust execution."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_count: int = 0
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class RobustExecutor:
    """Robust execution with retry logic and error handling."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        log_level: str = "INFO"
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_multiplier = backoff_multiplier
        
        # Setup logging
        self.logger = SimpleLogger(f"RobustExecutor_{id(self)}", log_level)
    
    def execute(
        self,
        func: Callable,
        *args,
        retry_on: tuple = (Exception,),
        critical_exceptions: tuple = (MemoryError, KeyboardInterrupt),
        **kwargs
    ) -> ExecutionResult:
        """
        Execute function with robust error handling and retries.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_on: Exceptions to retry on
            critical_exceptions: Exceptions that should not be retried
            **kwargs: Function keyword arguments
            
        Returns:
            ExecutionResult with success status and results
        """
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Executing {func.__name__}, attempt {attempt + 1}")
                
                # Execute function
                function_result = func(*args, **kwargs)
                
                # Success
                result.success = True
                result.result = function_result
                result.execution_time = time.time() - start_time
                result.retry_count = attempt
                
                if attempt > 0:
                    self.logger.info(f"Function {func.__name__} succeeded after {attempt} retries")
                
                break
                
            except critical_exceptions as e:
                # Critical exceptions should not be retried
                self.logger.critical(f"Critical exception in {func.__name__}: {e}")
                result.error = e
                result.execution_time = time.time() - start_time
                result.retry_count = attempt
                break
                
            except retry_on as e:
                result.error = e
                result.retry_count = attempt
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_multiplier ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}: {e}"
                    )
                    result.execution_time = time.time() - start_time
                    
            except Exception as e:
                # Unexpected exception
                self.logger.error(f"Unexpected exception in {func.__name__}: {e}")
                self.logger.debug(traceback.format_exc())
                result.error = e
                result.execution_time = time.time() - start_time
                result.retry_count = attempt
                break
        
        return result
    
    def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        **kwargs
    ) -> ExecutionResult:
        """Execute with fallback function if primary fails."""
        # Try primary function
        result = self.execute(primary_func, *args, **kwargs)
        
        if not result.success:
            self.logger.info(f"Primary function failed, trying fallback")
            fallback_result = self.execute(fallback_func, *args, **kwargs)
            
            if fallback_result.success:
                fallback_result.warnings.append(
                    f"Primary function {primary_func.__name__} failed, used fallback {fallback_func.__name__}"
                )
                return fallback_result
            else:
                # Both failed
                result.warnings.append(f"Both primary and fallback functions failed")
                return result
        
        return result


def robust_operation(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    component: str = "unknown",
    operation: str = "unknown",
    critical_exceptions: tuple = (MemoryError, KeyboardInterrupt)
):
    """Decorator for robust operation execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            executor = RobustExecutor(max_retries=max_retries, retry_delay=retry_delay)
            
            result = executor.execute(
                func, *args, 
                critical_exceptions=critical_exceptions,
                **kwargs
            )
            
            if not result.success:
                # Log comprehensive error info
                error_info = {
                    "component": component,
                    "operation": operation,
                    "function": func.__name__,
                    "error": str(result.error),
                    "retry_count": result.retry_count,
                    "execution_time": result.execution_time
                }
                
                executor.logger.error(f"Robust operation failed: {json.dumps(error_info, indent=2)}")
                
                # Re-raise the exception with additional context
                raise RuntimeError(
                    f"Operation {operation} in component {component} failed after "
                    f"{result.retry_count} retries: {result.error}"
                ) from result.error
            
            return result.result
        
        return wrapper
    return decorator


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.metrics = {
            "memory_usage": [],
            "execution_times": [],
            "error_counts": {},
            "success_counts": {}
        }
        self.start_time = time.time()
    
    def record_execution(self, component: str, operation: str, success: bool, execution_time: float):
        """Record execution metrics."""
        self.metrics["execution_times"].append({
            "component": component,
            "operation": operation,
            "time": execution_time,
            "timestamp": time.time()
        })
        
        if success:
            key = f"{component}.{operation}"
            self.metrics["success_counts"][key] = self.metrics["success_counts"].get(key, 0) + 1
        else:
            key = f"{component}.{operation}"
            self.metrics["error_counts"][key] = self.metrics["error_counts"].get(key, 0) + 1
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        uptime = time.time() - self.start_time
        
        # Calculate averages
        total_executions = len(self.metrics["execution_times"])
        avg_execution_time = 0.0
        if total_executions > 0:
            avg_execution_time = sum(
                e["time"] for e in self.metrics["execution_times"]
            ) / total_executions
        
        # Error rates
        total_errors = sum(self.metrics["error_counts"].values())
        total_successes = sum(self.metrics["success_counts"].values())
        total_operations = total_errors + total_successes
        error_rate = total_errors / max(total_operations, 1)
        
        return {
            "uptime_seconds": uptime,
            "total_operations": total_operations,
            "success_rate": 1 - error_rate,
            "error_rate": error_rate,
            "avg_execution_time": avg_execution_time,
            "memory_usage": self.get_memory_usage(),
            "error_breakdown": self.metrics["error_counts"].copy(),
            "success_breakdown": self.metrics["success_counts"].copy(),
            "recent_executions": self.metrics["execution_times"][-10:]  # Last 10
        }


# Global health monitor instance
global_health_monitor = HealthMonitor()


class SecurityValidator:
    """Security validation for inputs and operations."""
    
    @staticmethod
    def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric input ranges."""
        try:
            num_val = float(value)
            if min_val is not None and num_val < min_val:
                return False
            if max_val is not None and num_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_array_input(arr: Any, max_size: int = 10000) -> bool:
        """Validate array inputs for size and content."""
        try:
            if hasattr(arr, '__len__'):
                if len(arr) > max_size:
                    return False
                
                # Check for reasonable numeric values
                if hasattr(arr, '__iter__'):
                    for item in arr:
                        if not SecurityValidator.validate_numeric_input(item, -1e6, 1e6):
                            return False
                return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file path to prevent directory traversal."""
        import os
        import pathlib
        
        # Remove any potential directory traversal
        clean_path = pathlib.Path(path).resolve()
        
        # Ensure it's within current working directory or a safe location
        cwd = pathlib.Path.cwd()
        try:
            clean_path.relative_to(cwd)
            return str(clean_path)
        except ValueError:
            # Path is outside CWD, use just the filename
            return str(pathlib.Path(clean_path.name))


def secure_operation(
    validate_inputs: bool = True,
    max_array_size: int = 10000,
    max_numeric_value: float = 1e6
):
    """Decorator for secure operation validation."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if validate_inputs:
                # Validate all numeric arguments
                for i, arg in enumerate(args):
                    if isinstance(arg, (int, float)):
                        if not SecurityValidator.validate_numeric_input(
                            arg, -max_numeric_value, max_numeric_value
                        ):
                            raise ValueError(f"Argument {i} out of safe range: {arg}")
                    elif hasattr(arg, '__iter__') and not isinstance(arg, str):
                        if not SecurityValidator.validate_array_input(arg, max_array_size):
                            raise ValueError(f"Argument {i} failed array validation")
                
                # Validate keyword arguments
                for key, value in kwargs.items():
                    if isinstance(value, (int, float)):
                        if not SecurityValidator.validate_numeric_input(
                            value, -max_numeric_value, max_numeric_value
                        ):
                            raise ValueError(f"Keyword argument {key} out of safe range: {value}")
                    elif hasattr(value, '__iter__') and not isinstance(value, str):
                        if not SecurityValidator.validate_array_input(value, max_array_size):
                            raise ValueError(f"Keyword argument {key} failed array validation")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Performance monitoring decorator
def monitor_performance(component: str = "unknown"):
    """Decorator to monitor performance of operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                raise e
            finally:
                execution_time = time.time() - start_time
                global_health_monitor.record_execution(
                    component, func.__name__, success, execution_time
                )
        
        return wrapper
    return decorator


# Quick test
def test_robust_execution():
    """Test robust execution framework."""
    print("🛡️ Testing Robust Execution Framework...")
    
    @robust_operation(max_retries=2, component="test", operation="math")
    def unreliable_function(x, fail_count=0):
        """Function that fails occasionally."""
        if fail_count > 0:
            fail_count -= 1
            raise ValueError("Intentional failure")
        return x * 2
    
    @secure_operation(validate_inputs=True)
    def secure_function(values):
        """Secure function with input validation."""
        return sum(values)
    
    @monitor_performance(component="test")
    def monitored_function(n):
        """Function with performance monitoring."""
        time.sleep(0.01)  # Simulate work
        return n ** 2
    
    # Test robust execution
    try:
        result = unreliable_function(5, fail_count=1)
        print(f"✅ Robust execution: {result}")
    except Exception as e:
        print(f"❌ Robust execution failed: {e}")
    
    # Test secure validation
    try:
        result = secure_function([1, 2, 3, 4, 5])
        print(f"✅ Secure operation: {result}")
    except Exception as e:
        print(f"❌ Secure operation failed: {e}")
    
    # Test performance monitoring
    for i in range(3):
        result = monitored_function(i)
        print(f"✅ Monitored operation {i}: {result}")
    
    # Get health report
    health = global_health_monitor.get_health_report()
    print(f"✅ Health report: {health['total_operations']} operations, "
          f"{health['success_rate']:.2%} success rate")
    
    print("🛡️ Robust Execution Framework test completed!")


if __name__ == "__main__":
    test_robust_execution()