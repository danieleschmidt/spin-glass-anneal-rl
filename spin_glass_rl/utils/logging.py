"""Logging utilities for Spin-Glass-Anneal-RL."""

import logging
import logging.config
import sys
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import time


# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "spin_glass_rl.utils.logging.JSONFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "spin_glass_rl.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "spin_glass_rl": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"]
    }
}


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", "msecs", 
                          "relativeCreated", "thread", "threadName", "processName", 
                          "process", "message", "exc_info", "exc_text", "stack_info"):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = "spin_glass_rl.performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log timing information."""
        self.logger.info(
            f"Performance: {operation} took {duration:.4f}s",
            extra={
                "operation": operation,
                "duration": duration,
                "metrics_type": "timing",
                **kwargs
            }
        )
    
    def log_memory(self, operation: str, memory_usage: Dict[str, Any]):
        """Log memory usage."""
        self.logger.info(
            f"Memory: {operation}",
            extra={
                "operation": operation,
                "memory_usage": memory_usage,
                "metrics_type": "memory"
            }
        )
    
    def log_gpu_stats(self, operation: str, gpu_stats: Dict[str, Any]):
        """Log GPU statistics."""
        self.logger.info(
            f"GPU: {operation}",
            extra={
                "operation": operation,
                "gpu_stats": gpu_stats,
                "metrics_type": "gpu"
            }
        )


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, **kwargs):
        self.operation = operation
        self.logger = logger or get_logger("spin_glass_rl.timing")
        self.kwargs = kwargs
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Operation '{self.operation}' completed in {duration:.4f}s",
                extra={
                    "operation": self.operation,
                    "duration": duration,
                    "success": True,
                    **self.kwargs
                }
            )
        else:
            self.logger.error(
                f"Operation '{self.operation}' failed after {duration:.4f}s: {exc_val}",
                extra={
                    "operation": self.operation,
                    "duration": duration,
                    "success": False,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    **self.kwargs
                }
            )


class ProgressLogger:
    """Logger for progress tracking."""
    
    def __init__(self, total_steps: int, logger_name: str = "spin_glass_rl.progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logging.getLogger(logger_name)
        self.last_log_time = time.time()
        self.log_interval = 1.0  # Log at most once per second
    
    def update(self, step: int, **kwargs):
        """Update progress."""
        self.current_step = step
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            progress = step / self.total_steps if self.total_steps > 0 else 0
            
            self.logger.info(
                f"Progress: {step}/{self.total_steps} ({progress:.1%})",
                extra={
                    "current_step": step,
                    "total_steps": self.total_steps,
                    "progress": progress,
                    "metrics_type": "progress",
                    **kwargs
                }
            )
            
            self.last_log_time = current_time
    
    def complete(self, **kwargs):
        """Mark progress as complete."""
        self.logger.info(
            f"Progress: {self.total_steps}/{self.total_steps} (100%) - Complete",
            extra={
                "current_step": self.total_steps,
                "total_steps": self.total_steps,
                "progress": 1.0,
                "metrics_type": "progress",
                "completed": True,
                **kwargs
            }
        )


def setup_logger(
    name: str = "spin_glass_rl",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "standard",
    config_dict: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_type: Format type ("standard", "detailed", "json")
        config_dict: Custom logging configuration
        
    Returns:
        Configured logger
    """
    if config_dict is not None:
        logging.config.dictConfig(config_dict)
    else:
        # Use default configuration with modifications
        config = DEFAULT_CONFIG.copy()
        
        # Set level
        config["loggers"]["spin_glass_rl"]["level"] = level.upper()
        config["handlers"]["console"]["level"] = level.upper()
        
        # Set format
        config["handlers"]["console"]["formatter"] = format_type
        config["handlers"]["file"]["formatter"] = format_type
        
        # Set log file
        if log_file:
            config["handlers"]["file"]["filename"] = log_file
            # Create log directory if it doesn't exist
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)
        
        logging.config.dictConfig(config)
    
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


def configure_logging_from_env():
    """Configure logging from environment variables."""
    level = os.getenv("SPIN_GLASS_LOG_LEVEL", "INFO")
    log_file = os.getenv("SPIN_GLASS_LOG_FILE")
    format_type = os.getenv("SPIN_GLASS_LOG_FORMAT", "standard")
    
    setup_logger(level=level, log_file=log_file, format_type=format_type)


def silence_external_loggers(level: str = "WARNING"):
    """Silence noisy external library loggers."""
    external_loggers = [
        "matplotlib",
        "PIL",
        "urllib3",
        "requests",
        "torch",
        "numpy"
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


class LogCapture:
    """Context manager to capture log messages for testing."""
    
    def __init__(self, logger_name: str, level: str = "DEBUG"):
        self.logger_name = logger_name
        self.level = getattr(logging, level.upper())
        self.handler = None
        self.records = []
    
    def __enter__(self):
        self.handler = LogCaptureHandler(self.records)
        self.handler.setLevel(self.level)
        
        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)
        
        return self.records
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)


class LogCaptureHandler(logging.Handler):
    """Handler that captures log records in a list."""
    
    def __init__(self, records_list: list):
        super().__init__()
        self.records = records_list
    
    def emit(self, record: logging.LogRecord):
        self.records.append(record)


def log_system_info():
    """Log system information."""
    import platform
    import torch
    
    logger = get_logger("spin_glass_rl.system")
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        system_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
    
    logger.info("System information", extra={"system_info": system_info})


def create_experiment_logger(experiment_name: str, output_dir: str) -> logging.Logger:
    """Create logger for experiment tracking."""
    log_file = Path(output_dir) / f"{experiment_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "experiment": {
                "format": "%(asctime)s [%(levelname)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "experiment_file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "experiment",
                "filename": str(log_file),
                "mode": "w",
                "encoding": "utf8"
            }
        },
        "loggers": {
            f"experiment.{experiment_name}": {
                "level": "DEBUG",
                "handlers": ["experiment_file"],
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(config)
    return logging.getLogger(f"experiment.{experiment_name}")


# Initialize logging when module is imported
try:
    configure_logging_from_env()
except Exception:
    # Fallback to basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )