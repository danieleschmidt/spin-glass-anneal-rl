"""Custom exceptions for Spin-Glass-Anneal-RL."""

from typing import Any, Dict, Optional


class SpinGlassError(Exception):
    """Base exception for spin-glass-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {detail_str})"
        return self.message


class ModelError(SpinGlassError):
    """Errors related to Ising model operations."""
    pass


class AnnealingError(SpinGlassError):
    """Errors related to annealing algorithms."""
    pass


class ConstraintError(SpinGlassError):
    """Errors related to constraint handling."""
    pass


class DeviceError(SpinGlassError):
    """Errors related to device/hardware operations."""
    pass


class ValidationError(SpinGlassError):
    """Errors related to input validation."""
    pass


class ConfigurationError(SpinGlassError):
    """Errors related to configuration parameters."""
    pass


class ConvergenceError(AnnealingError):
    """Errors related to convergence issues."""
    pass


class ResourceError(SpinGlassError):
    """Errors related to resource allocation or limits."""
    pass


class ProblemEncodingError(SpinGlassError):
    """Errors related to problem encoding to Ising models."""
    pass


class SolutionDecodingError(SpinGlassError):
    """Errors related to solution decoding from spin configurations."""
    pass


# Utility functions for error handling

def handle_torch_errors(func):
    """Decorator to handle common PyTorch errors."""
    import functools
    import torch
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            raise ResourceError(
                "GPU out of memory during operation",
                details={"function": func.__name__, "original_error": str(e)}
            ) from e
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise DeviceError(
                    f"CUDA error in {func.__name__}",
                    details={"original_error": str(e)}
                ) from e
            else:
                raise SpinGlassError(
                    f"Runtime error in {func.__name__}",
                    details={"original_error": str(e)}
                ) from e
    
    return wrapper


def handle_numpy_errors(func):
    """Decorator to handle common NumPy errors."""
    import functools
    import numpy as np
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except np.linalg.LinAlgError as e:
            raise ModelError(
                f"Linear algebra error in {func.__name__}",
                details={"original_error": str(e)}
            ) from e
        except ValueError as e:
            if "shape" in str(e).lower():
                raise ValidationError(
                    f"Shape mismatch in {func.__name__}",
                    details={"original_error": str(e)}
                ) from e
            else:
                raise ValidationError(
                    f"Value error in {func.__name__}",
                    details={"original_error": str(e)}
                ) from e
    
    return wrapper


def validate_and_handle_errors(validation_func):
    """Decorator to validate inputs and handle validation errors."""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Run validation
                validation_func(*args, **kwargs)
                return func(*args, **kwargs)
            except ValidationError:
                raise  # Re-raise validation errors as-is
            except Exception as e:
                raise ValidationError(
                    f"Validation failed for {func.__name__}",
                    details={"original_error": str(e)}
                ) from e
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with additional context."""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not issubclass(exc_type, SpinGlassError):
            # Convert generic exceptions to SpinGlassError with context
            details = self.context.copy()
            details["original_error"] = str(exc_val)
            details["original_type"] = exc_type.__name__
            
            raise SpinGlassError(
                f"Error during {self.operation}",
                details=details
            ) from exc_val
        
        return False  # Don't suppress SpinGlassError exceptions


def create_error_summary(error: Exception) -> Dict[str, Any]:
    """Create a summary dictionary from an exception."""
    summary = {
        "error_type": type(error).__name__,
        "message": str(error),
        "module": getattr(error, "__module__", "unknown")
    }
    
    if isinstance(error, SpinGlassError):
        summary["details"] = error.details
    
    # Add stack trace info if available
    import traceback
    summary["traceback"] = traceback.format_exception(type(error), error, error.__traceback__)
    
    return summary


def log_error(error: Exception, logger=None):
    """Log error with appropriate level and context."""
    if logger is None:
        from spin_glass_rl.utils.logging import get_logger
        logger = get_logger(__name__)
    
    error_summary = create_error_summary(error)
    
    if isinstance(error, (ValidationError, ConfigurationError)):
        logger.warning(f"Validation error: {error.message}", extra=error_summary)
    elif isinstance(error, ResourceError):
        logger.error(f"Resource error: {error.message}", extra=error_summary)
    elif isinstance(error, DeviceError):
        logger.error(f"Device error: {error.message}", extra=error_summary)
    else:
        logger.error(f"General error: {error.message}", extra=error_summary)