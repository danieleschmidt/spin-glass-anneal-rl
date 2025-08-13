"""Robust error handling and validation for spin-glass optimization."""

import sys
import traceback
import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, Type
from dataclasses import dataclass
import torch
import numpy as np
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    parameters: Dict[str, Any]
    stack_trace: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class SpinGlassError(Exception):
    """Base exception for spin-glass optimization errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context
        self.message = message


class ModelConfigurationError(SpinGlassError):
    """Raised when model configuration is invalid."""
    pass


class OptimizationError(SpinGlassError):
    """Raised when optimization fails."""
    pass


class GPUError(SpinGlassError):
    """Raised when GPU operations fail."""
    pass


class DataValidationError(SpinGlassError):
    """Raised when data validation fails."""
    pass


class ConvergenceError(SpinGlassError):
    """Raised when optimization fails to converge."""
    pass


class RobustErrorHandler:
    """Robust error handling with automatic recovery."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_retries = 3
        self.enable_fallbacks = True
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies[torch.cuda.OutOfMemoryError] = self._handle_gpu_oom
        self.recovery_strategies[RuntimeError] = self._handle_runtime_error
        self.recovery_strategies[ValueError] = self._handle_value_error
        self.recovery_strategies[TypeError] = self._handle_type_error
    
    def _handle_gpu_oom(self, error: Exception, context: ErrorContext) -> Any:
        """Handle GPU out of memory errors."""
        print("⚠️  GPU OOM detected - attempting recovery...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Suggest smaller batch size
        if 'batch_size' in context.parameters:
            new_batch_size = max(1, context.parameters['batch_size'] // 2)
            context.parameters['batch_size'] = new_batch_size
            print(f"🔧 Reduced batch size to {new_batch_size}")
        
        # Suggest CPU fallback
        if 'device' in context.parameters:
            context.parameters['device'] = 'cpu'
            print("🔧 Falling back to CPU computation")
        
        return context.parameters
    
    def _handle_runtime_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle runtime errors."""
        error_msg = str(error).lower()
        
        if 'cuda' in error_msg:
            return self._handle_gpu_oom(error, context)
        elif 'dimension' in error_msg or 'size' in error_msg:
            print("⚠️  Tensor dimension mismatch detected")
            # Log detailed tensor shapes if available
            if hasattr(error, 'args') and error.args:
                print(f"🔍 Error details: {error.args[0]}")
        
        return None
    
    def _handle_value_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle value errors."""
        error_msg = str(error).lower()
        
        if 'temperature' in error_msg:
            # Fix temperature bounds
            if 'temperature' in context.parameters:
                temp = context.parameters['temperature']
                context.parameters['temperature'] = max(0.001, min(1000.0, temp))
                print(f"🔧 Adjusted temperature to valid range: {context.parameters['temperature']}")
        
        elif 'coupling' in error_msg:
            # Fix coupling values
            if 'couplings' in context.parameters:
                couplings = context.parameters['couplings']
                if isinstance(couplings, torch.Tensor):
                    # Clamp extreme values
                    context.parameters['couplings'] = torch.clamp(couplings, -100.0, 100.0)
                    print("🔧 Clamped coupling values to reasonable range")
        
        return context.parameters
    
    def _handle_type_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle type errors."""
        error_msg = str(error).lower()
        
        if 'tensor' in error_msg:
            # Try to convert to appropriate tensor types
            for key, value in context.parameters.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    try:
                        context.parameters[key] = torch.tensor(value, dtype=torch.float32)
                        print(f"🔧 Converted {key} to tensor")
                    except Exception:
                        pass
        
        return context.parameters
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        retry_count: int = 0
    ) -> Optional[Any]:
        """Handle an error with automatic recovery attempts."""
        
        # Log error
        self.error_history.append(context)
        print(f"❌ Error in {context.component}.{context.operation}: {error}")
        
        # Determine severity
        if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
            context.severity = ErrorSeverity.HIGH
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            context.severity = ErrorSeverity.CRITICAL
            raise error  # Don't recover from user interruption
        
        # Attempt recovery
        if retry_count < self.max_retries and self.enable_fallbacks:
            error_type = type(error)
            if error_type in self.recovery_strategies:
                try:
                    recovery_result = self.recovery_strategies[error_type](error, context)
                    if recovery_result is not None:
                        print(f"✅ Recovery successful for {context.operation}")
                        return recovery_result
                except Exception as recovery_error:
                    print(f"⚠️  Recovery failed: {recovery_error}")
        
        # Re-raise if no recovery possible
        raise error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "recent_errors": []}
        
        recent = self.error_history[-10:]  # Last 10 errors
        severity_counts = {}
        for error_ctx in recent:
            severity = error_ctx.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent),
            "severity_breakdown": severity_counts,
            "most_common_operation": max(
                (ctx.operation for ctx in recent),
                key=lambda op: sum(1 for ctx in recent if ctx.operation == op),
                default="none"
            )
        }


def robust_operation(
    component: str = "unknown",
    operation: str = "unknown",
    max_retries: int = 3,
    fallback_value: Any = None
):
    """Decorator for robust operation execution with automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = RobustErrorHandler()
            
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    context = ErrorContext(
                        operation=operation,
                        component=component,
                        parameters={"args": args, "kwargs": kwargs},
                        stack_trace=traceback.format_exc()
                    )
                    
                    if retry == max_retries:
                        # Final attempt failed
                        if fallback_value is not None:
                            print(f"🔄 Using fallback value for {operation}")
                            return fallback_value
                        else:
                            print(f"💥 All retries exhausted for {operation}")
                            raise e
                    
                    # Attempt recovery
                    try:
                        recovered_params = error_handler.handle_error(e, context, retry)
                        if recovered_params:
                            # Update function arguments with recovered parameters
                            kwargs.update(recovered_params)
                            print(f"🔄 Retrying {operation} (attempt {retry + 2}/{max_retries + 1})")
                        else:
                            raise e
                    except Exception:
                        if retry == max_retries:
                            raise e
                        continue
            
            return fallback_value
        
        return wrapper
    return decorator


class InputValidator:
    """Comprehensive input validation for optimization components."""
    
    @staticmethod
    def validate_ising_config(config) -> None:
        """Validate Ising model configuration."""
        if config.n_spins <= 0:
            raise ModelConfigurationError(
                f"Number of spins must be positive, got {config.n_spins}"
            )
        
        if config.n_spins > 100000:
            warnings.warn(
                f"Large model size ({config.n_spins} spins) may cause memory issues",
                UserWarning
            )
        
        if not isinstance(config.coupling_strength, (int, float)):
            raise ModelConfigurationError(
                f"Coupling strength must be numeric, got {type(config.coupling_strength)}"
            )
        
        if abs(config.coupling_strength) > 1000:
            warnings.warn(
                f"Large coupling strength ({config.coupling_strength}) may cause numerical instability",
                UserWarning
            )
    
    @staticmethod
    def validate_annealer_config(config) -> None:
        """Validate annealer configuration."""
        if config.n_sweeps <= 0:
            raise ModelConfigurationError(
                f"Number of sweeps must be positive, got {config.n_sweeps}"
            )
        
        if config.initial_temp <= 0:
            raise ModelConfigurationError(
                f"Initial temperature must be positive, got {config.initial_temp}"
            )
        
        if config.final_temp <= 0:
            raise ModelConfigurationError(
                f"Final temperature must be positive, got {config.final_temp}"
            )
        
        if config.final_temp >= config.initial_temp:
            raise ModelConfigurationError(
                f"Final temperature ({config.final_temp}) must be less than "
                f"initial temperature ({config.initial_temp})"
            )
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            raise DataValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        if torch.isnan(tensor).any():
            raise DataValidationError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise DataValidationError(f"{name} contains infinite values")
        
        if tensor.numel() == 0:
            raise DataValidationError(f"{name} is empty")
    
    @staticmethod
    def validate_spins(spins: torch.Tensor) -> None:
        """Validate spin configuration."""
        InputValidator.validate_tensor(spins, "spins")
        
        # Check if spins are in valid range
        unique_values = torch.unique(spins)
        valid_values = torch.tensor([-1.0, 1.0])
        
        for val in unique_values:
            if not torch.any(torch.isclose(val, valid_values, atol=1e-6)):
                raise DataValidationError(
                    f"Invalid spin value {val.item():.6f}, spins must be ±1"
                )
    
    @staticmethod
    def validate_couplings(couplings: torch.Tensor, n_spins: int) -> None:
        """Validate coupling matrix."""
        InputValidator.validate_tensor(couplings, "couplings")
        
        expected_shape = (n_spins, n_spins)
        if couplings.shape != expected_shape:
            raise DataValidationError(
                f"Coupling matrix shape {couplings.shape} doesn't match "
                f"expected shape {expected_shape}"
            )
        
        # Check symmetry (within tolerance)
        if not torch.allclose(couplings, couplings.t(), atol=1e-6):
            raise DataValidationError("Coupling matrix must be symmetric")


def validate_optimization_result(result) -> None:
    """Validate optimization result."""
    if not hasattr(result, 'best_energy'):
        raise OptimizationError("Result missing best_energy attribute")
    
    if not hasattr(result, 'best_configuration'):
        raise OptimizationError("Result missing best_configuration attribute")
    
    if np.isnan(result.best_energy) or np.isinf(result.best_energy):
        raise OptimizationError(f"Invalid best energy: {result.best_energy}")
    
    if result.total_time < 0:
        raise OptimizationError(f"Invalid optimization time: {result.total_time}")
    
    if result.n_sweeps <= 0:
        raise OptimizationError(f"Invalid number of sweeps: {result.n_sweeps}")


# Global error handler instance
global_error_handler = RobustErrorHandler()