"""Enhanced logging system with structured logging and monitoring integration."""

import logging
import logging.handlers
import json
import time
import threading
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import os

# Create custom log levels
TRACE_LEVEL = 5
SUCCESS_LEVEL = 25

logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'hostname': self.hostname,
            'thread_id': threading.current_thread().ident,
            'process_id': os.getpid(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items() 
                if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                           'filename', 'module', 'lineno', 'funcName', 'created', 
                           'msecs', 'relativeCreated', 'thread', 'threadName',
                           'processName', 'process', 'message', 'exc_info', 'exc_text', 'stack_info']
            }
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to record."""
        record.uptime = time.time() - self.start_time
        
        # Add memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except ImportError:
            pass
        
        return True


class SecurityFilter(logging.Filter):
    """Filter to sanitize sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        r'password["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'token["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'key["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        r'secret["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize sensitive information from log message."""
        message = record.getMessage()
        
        for pattern in self.SENSITIVE_PATTERNS:
            import re
            message = re.sub(pattern, r'\1' + '***REDACTED***', message, flags=re.IGNORECASE)
        
        # Update the record's args to reflect sanitized message
        record.args = ()
        record.msg = message
        
        return True


class RobustLogger:
    """Enhanced logger with monitoring integration and error recovery."""
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 log_dir: Optional[str] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5,
                 structured: bool = True,
                 console_output: bool = True):
        
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.structured = structured
        self.console_output = console_output
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Add filters
        self._setup_filters()
        
        # Performance tracking
        self.performance_metrics = {
            'log_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'start_time': time.time()
        }
    
    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        # Error file handler
        error_file = self.log_dir / f"{self.name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
        
        # Set formatters
        if self.structured:
            formatter = StructuredFormatter()
            file_handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)
        
        if self.console_output:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        if self.console_output:
            self.logger.addHandler(console_handler)
    
    def _setup_filters(self) -> None:
        """Setup log filters."""
        # Add performance filter
        perf_filter = PerformanceFilter()
        self.logger.addFilter(perf_filter)
        
        # Add security filter
        security_filter = SecurityFilter()
        self.logger.addFilter(security_filter)
    
    def trace(self, message: str, **kwargs) -> None:
        """Log trace level message."""
        self.logger.log(TRACE_LEVEL, message, extra=kwargs)
        self.performance_metrics['log_count'] += 1
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
        self.performance_metrics['log_count'] += 1
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)
        self.performance_metrics['log_count'] += 1
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message."""
        self.logger.log(SUCCESS_LEVEL, message, extra=kwargs)
        self.performance_metrics['log_count'] += 1
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
        self.performance_metrics['log_count'] += 1
        self.performance_metrics['warning_count'] += 1
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if exception:
            self.logger.error(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
        
        self.performance_metrics['log_count'] += 1
        self.performance_metrics['error_count'] += 1
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        if exception:
            self.logger.critical(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)
        
        self.performance_metrics['log_count'] += 1
        self.performance_metrics['error_count'] += 1
    
    def log_performance_metrics(self) -> None:
        """Log current performance metrics."""
        uptime = time.time() - self.performance_metrics['start_time']
        
        metrics = {
            'uptime_seconds': uptime,
            'total_logs': self.performance_metrics['log_count'],
            'error_count': self.performance_metrics['error_count'],
            'warning_count': self.performance_metrics['warning_count'],
            'logs_per_second': self.performance_metrics['log_count'] / max(uptime, 1),
            'error_rate': self.performance_metrics['error_count'] / max(self.performance_metrics['log_count'], 1)
        }
        
        self.info("Performance metrics", **metrics)
    
    def log_system_info(self) -> None:
        """Log system information."""
        try:
            import platform
            import psutil
            
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage('.').free / (1024**3)
            }
            
            # Add GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    system_info['gpu_count'] = torch.cuda.device_count()
                    system_info['gpu_name'] = torch.cuda.get_device_name(0)
                    system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                pass
            
            self.info("System information", **system_info)
            
        except ImportError:
            self.warning("Could not log system info - psutil not available")
    
    def create_child_logger(self, child_name: str) -> 'RobustLogger':
        """Create child logger with same configuration."""
        full_name = f"{self.name}.{child_name}"
        
        child = RobustLogger(
            name=full_name,
            level=logging.getLevelName(self.level),
            log_dir=str(self.log_dir),
            max_file_size=self.max_file_size,
            backup_count=self.backup_count,
            structured=self.structured,
            console_output=False  # Child loggers don't output to console by default
        )
        
        return child
    
    def configure_third_party_loggers(self, level: str = "WARNING") -> None:
        """Configure third-party library loggers to reduce noise."""
        third_party_loggers = [
            'urllib3', 'requests', 'matplotlib', 'PIL', 'h5py',
            'asyncio', 'concurrent.futures', 'multiprocessing'
        ]
        
        log_level = getattr(logging, level.upper())
        
        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
        
        self.info(f"Configured third-party loggers to {level} level")
    
    def emergency_log(self, message: str, **kwargs) -> None:
        """Emergency logging that bypasses filters and formatting."""
        # Create emergency handler that writes directly to stderr
        emergency_handler = logging.StreamHandler(sys.stderr)
        emergency_handler.setFormatter(logging.Formatter('EMERGENCY: %(message)s'))
        
        # Create temporary logger
        emergency_logger = logging.getLogger(f"{self.name}.emergency")
        emergency_logger.handlers.clear()
        emergency_logger.addHandler(emergency_handler)
        emergency_logger.setLevel(logging.CRITICAL)
        
        # Log the emergency message
        emergency_logger.critical(message, extra=kwargs)
    
    def flush_all_handlers(self) -> None:
        """Flush all log handlers."""
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'name': self.name,
            'level': logging.getLevelName(self.level),
            'metrics': self.performance_metrics.copy(),
            'handlers': len(self.logger.handlers),
            'filters': len(self.logger.filters)
        }


# Global logger registry
_logger_registry: Dict[str, RobustLogger] = {}


def get_logger(name: str, **kwargs) -> RobustLogger:
    """Get or create a robust logger."""
    if name not in _logger_registry:
        _logger_registry[name] = RobustLogger(name, **kwargs)
    return _logger_registry[name]


def setup_global_logging(level: str = "INFO", 
                        log_dir: Optional[str] = None,
                        structured: bool = True) -> RobustLogger:
    """Setup global logging configuration."""
    # Create main logger
    main_logger = get_logger(
        "spin_glass_rl",
        level=level,
        log_dir=log_dir,
        structured=structured,
        console_output=True
    )
    
    # Configure third-party loggers
    main_logger.configure_third_party_loggers("WARNING")
    
    # Log system information
    main_logger.log_system_info()
    
    return main_logger


def log_function_call(func):
    """Decorator to log function calls with performance metrics."""
    def wrapper(*args, **kwargs):
        logger = get_logger("function_calls")
        start_time = time.time()
        
        try:
            logger.trace(
                f"Calling {func.__module__}.{func.__name__}",
                function=func.__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.trace(
                f"Completed {func.__name__}",
                function=func.__name__,
                execution_time_ms=execution_time * 1000,
                success=True
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in {func.__name__}",
                exception=e,
                function=func.__name__,
                execution_time_ms=execution_time * 1000,
                success=False
            )
            raise
    
    return wrapper


# Context manager for logging operations
class LoggingContext:
    """Context manager for operation logging."""
    
    def __init__(self, operation_name: str, logger: Optional[RobustLogger] = None, **metadata):
        self.operation_name = operation_name
        self.logger = logger or get_logger("operations")
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Starting operation: {self.operation_name}",
            operation=self.operation_name,
            **self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.success(
                f"Completed operation: {self.operation_name}",
                operation=self.operation_name,
                execution_time_ms=execution_time * 1000,
                **self.metadata
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation_name}",
                operation=self.operation_name,
                execution_time_ms=execution_time * 1000,
                exception=exc_val,
                **self.metadata
            )


def cleanup_old_logs(log_dir: str, days_to_keep: int = 30) -> None:
    """Clean up old log files."""
    import datetime
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
    
    for log_file in log_path.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            try:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Failed to delete {log_file}: {e}")