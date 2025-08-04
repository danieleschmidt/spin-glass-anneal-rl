"""Input validation utilities."""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import is_dataclass

from spin_glass_rl.utils.exceptions import ValidationError


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    dtype: Optional[torch.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    finite: bool = True,
    device: Optional[torch.device] = None
) -> None:
    """
    Validate tensor properties.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        dtype: Expected data type
        shape: Expected exact shape
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        finite: Whether all values must be finite
        device: Expected device
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"{name} must be a torch.Tensor, got {type(tensor)}",
            details={"expected_type": "torch.Tensor", "actual_type": type(tensor).__name__}
        )
    
    # Check dtype
    if dtype is not None and tensor.dtype != dtype:
        raise ValidationError(
            f"{name} must have dtype {dtype}, got {tensor.dtype}",
            details={"expected_dtype": str(dtype), "actual_dtype": str(tensor.dtype)}
        )
    
    # Check shape
    if shape is not None and tensor.shape != shape:
        raise ValidationError(
            f"{name} must have shape {shape}, got {tensor.shape}",
            details={"expected_shape": shape, "actual_shape": tuple(tensor.shape)}
        )
    
    # Check dimensions
    if min_dims is not None and tensor.ndim < min_dims:
        raise ValidationError(
            f"{name} must have at least {min_dims} dimensions, got {tensor.ndim}",
            details={"expected_min_dims": min_dims, "actual_dims": tensor.ndim}
        )
    
    if max_dims is not None and tensor.ndim > max_dims:
        raise ValidationError(
            f"{name} must have at most {max_dims} dimensions, got {tensor.ndim}",
            details={"expected_max_dims": max_dims, "actual_dims": tensor.ndim}
        )
    
    # Check values
    if finite and not torch.isfinite(tensor).all():
        raise ValidationError(
            f"{name} must contain only finite values",
            details={"has_nan": torch.isnan(tensor).any().item(), 
                    "has_inf": torch.isinf(tensor).any().item()}
        )
    
    if min_value is not None and tensor.min().item() < min_value:
        raise ValidationError(
            f"{name} values must be >= {min_value}, got min value {tensor.min().item()}",
            details={"expected_min": min_value, "actual_min": tensor.min().item()}
        )
    
    if max_value is not None and tensor.max().item() > max_value:
        raise ValidationError(
            f"{name} values must be <= {max_value}, got max value {tensor.max().item()}",
            details={"expected_max": max_value, "actual_max": tensor.max().item()}
        )
    
    # Check device
    if device is not None and tensor.device != device:
        raise ValidationError(
            f"{name} must be on device {device}, got {tensor.device}",
            details={"expected_device": str(device), "actual_device": str(tensor.device)}
        )


def validate_array(
    array: np.ndarray,
    name: str = "array", 
    dtype: Optional[np.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    finite: bool = True
) -> None:
    """
    Validate NumPy array properties.
    
    Args:
        array: Array to validate
        name: Name for error messages
        dtype: Expected data type
        shape: Expected exact shape
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        finite: Whether all values must be finite
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            f"{name} must be a numpy.ndarray, got {type(array)}",
            details={"expected_type": "numpy.ndarray", "actual_type": type(array).__name__}
        )
    
    # Check dtype
    if dtype is not None and array.dtype != dtype:
        raise ValidationError(
            f"{name} must have dtype {dtype}, got {array.dtype}",
            details={"expected_dtype": str(dtype), "actual_dtype": str(array.dtype)}
        )
    
    # Check shape
    if shape is not None and array.shape != shape:
        raise ValidationError(
            f"{name} must have shape {shape}, got {array.shape}",
            details={"expected_shape": shape, "actual_shape": array.shape}
        )
    
    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        raise ValidationError(
            f"{name} must have at least {min_dims} dimensions, got {array.ndim}",
            details={"expected_min_dims": min_dims, "actual_dims": array.ndim}
        )
    
    if max_dims is not None and array.ndim > max_dims:
        raise ValidationError(
            f"{name} must have at most {max_dims} dimensions, got {array.ndim}",
            details={"expected_max_dims": max_dims, "actual_dims": array.ndim}
        )
    
    # Check values
    if finite and not np.isfinite(array).all():
        raise ValidationError(
            f"{name} must contain only finite values",
            details={"has_nan": np.isnan(array).any(), "has_inf": np.isinf(array).any()}
        )
    
    if min_value is not None and array.min() < min_value:
        raise ValidationError(
            f"{name} values must be >= {min_value}, got min value {array.min()}",
            details={"expected_min": min_value, "actual_min": float(array.min())}
        )
    
    if max_value is not None and array.max() > max_value:
        raise ValidationError(
            f"{name} values must be <= {max_value}, got max value {array.max()}",
            details={"expected_max": max_value, "actual_max": float(array.max())}
        )


def validate_numeric(
    value: Union[int, float],
    name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer: bool = False,
    positive: bool = False,
    finite: bool = True
) -> None:
    """
    Validate numeric value.
    
    Args:
        value: Value to validate
        name: Name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer: Whether value must be integer
        positive: Whether value must be positive
        finite: Whether value must be finite
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be numeric, got {type(value)}",
            details={"expected_type": "numeric", "actual_type": type(value).__name__}
        )
    
    if integer and not isinstance(value, int):
        raise ValidationError(
            f"{name} must be integer, got {type(value)}",
            details={"expected_type": "int", "actual_type": type(value).__name__}
        )
    
    if finite and not np.isfinite(value):
        raise ValidationError(
            f"{name} must be finite, got {value}",
            details={"value": value, "is_nan": np.isnan(value), "is_inf": np.isinf(value)}
        )
    
    if positive and value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            details={"value": value}
        )
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{name} must be >= {min_value}, got {value}",
            details={"expected_min": min_value, "actual_value": value}
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{name} must be <= {max_value}, got {value}",
            details={"expected_max": max_value, "actual_value": value}
        )


def validate_string(
    value: str,
    name: str = "value",
    allowed_values: Optional[List[str]] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None
) -> None:
    """
    Validate string value.
    
    Args:
        value: String to validate
        name: Name for error messages
        allowed_values: List of allowed string values
        min_length: Minimum string length
        max_length: Maximum string length  
        pattern: Regex pattern to match
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{name} must be string, got {type(value)}",
            details={"expected_type": "str", "actual_type": type(value).__name__}
        )
    
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"{name} must be one of {allowed_values}, got '{value}'",
            details={"allowed_values": allowed_values, "actual_value": value}
        )
    
    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"{name} must be at least {min_length} characters, got {len(value)}",
            details={"expected_min_length": min_length, "actual_length": len(value)}
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{name} must be at most {max_length} characters, got {len(value)}",
            details={"expected_max_length": max_length, "actual_length": len(value)}
        )
    
    if pattern is not None:
        import re
        if not re.match(pattern, value):
            raise ValidationError(
                f"{name} must match pattern '{pattern}', got '{value}'",
                details={"pattern": pattern, "actual_value": value}
            )


def validate_config(config: Any, required_fields: Optional[List[str]] = None) -> None:
    """
    Validate configuration object.
    
    Args:
        config: Configuration object to validate
        required_fields: List of required field names
        
    Raises:
        ValidationError: If validation fails
    """
    if config is None:
        raise ValidationError("Configuration cannot be None")
    
    # Check if it's a dataclass or has required attributes
    if not (is_dataclass(config) or hasattr(config, '__dict__')):
        raise ValidationError(
            f"Configuration must be dataclass or have attributes, got {type(config)}",
            details={"config_type": type(config).__name__}
        )
    
    if required_fields is not None:
        config_dict = config.__dict__ if hasattr(config, '__dict__') else {}
        
        missing_fields = []
        for field in required_fields:
            if field not in config_dict:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValidationError(
                f"Configuration missing required fields: {missing_fields}",
                details={"missing_fields": missing_fields, "available_fields": list(config_dict.keys())}
            )


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """
    Validate and normalize device specification.
    
    Args:
        device: Device specification
        
    Returns:
        Normalized torch.device
        
    Raises:
        ValidationError: If device is invalid
    """
    try:
        device_obj = torch.device(device)
    except Exception as e:
        raise ValidationError(
            f"Invalid device specification: {device}",
            details={"device": str(device), "error": str(e)}
        ) from e
    
    # Check if CUDA device is available if requested
    if device_obj.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValidationError(
                "CUDA device requested but CUDA is not available",
                details={"requested_device": str(device_obj)}
            )
        
        if device_obj.index is not None and device_obj.index >= torch.cuda.device_count():
            raise ValidationError(
                f"CUDA device {device_obj.index} requested but only {torch.cuda.device_count()} devices available",
                details={"requested_index": device_obj.index, "available_devices": torch.cuda.device_count()}
            )
    
    return device_obj


def validate_spin_configuration(spins: torch.Tensor, name: str = "spins") -> None:
    """
    Validate spin configuration tensor.
    
    Args:
        spins: Spin configuration tensor
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_tensor(spins, name=name, min_dims=1, max_dims=2, finite=True)
    
    # Check that spins are binary {-1, +1} or {0, 1}
    unique_values = torch.unique(spins)
    
    valid_binary_1 = torch.allclose(torch.sort(unique_values)[0], torch.tensor([-1, 1], dtype=spins.dtype, device=spins.device))
    valid_binary_2 = torch.allclose(torch.sort(unique_values)[0], torch.tensor([0, 1], dtype=spins.dtype, device=spins.device))
    valid_single_1 = len(unique_values) == 1 and unique_values[0] in [-1, 1]
    valid_single_2 = len(unique_values) == 1 and unique_values[0] in [0, 1]
    
    if not (valid_binary_1 or valid_binary_2 or valid_single_1 or valid_single_2):
        raise ValidationError(
            f"{name} must contain only binary values {{-1, +1}} or {{0, 1}}, got unique values: {unique_values.tolist()}",
            details={"unique_values": unique_values.tolist()}
        )


def validate_coupling_matrix(couplings: torch.Tensor, n_spins: int, name: str = "couplings") -> None:
    """
    Validate coupling matrix.
    
    Args:
        couplings: Coupling matrix tensor
        n_spins: Expected number of spins
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if couplings.is_sparse:
        # Sparse tensor validation
        validate_tensor(couplings._indices(), name=f"{name}_indices", min_dims=2, max_dims=2)
        validate_tensor(couplings._values(), name=f"{name}_values", min_dims=1, finite=True)
        
        if couplings.shape != (n_spins, n_spins):
            raise ValidationError(
                f"{name} must have shape ({n_spins}, {n_spins}), got {couplings.shape}",
                details={"expected_shape": (n_spins, n_spins), "actual_shape": tuple(couplings.shape)}
            )
    else:
        # Dense tensor validation
        validate_tensor(couplings, name=name, shape=(n_spins, n_spins), finite=True)
        
        # Check symmetry (couplings should be symmetric)
        if not torch.allclose(couplings, couplings.t(), atol=1e-6):
            raise ValidationError(
                f"{name} matrix must be symmetric",
                details={"max_asymmetry": torch.max(torch.abs(couplings - couplings.t())).item()}
            )


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate probability value.
    
    Args:
        value: Probability value
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_numeric(value, name=name, min_value=0.0, max_value=1.0, finite=True)


def validate_temperature(value: float, name: str = "temperature") -> None:
    """
    Validate temperature value.
    
    Args:
        value: Temperature value
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_numeric(value, name=name, min_value=1e-10, finite=True)  # Temperature must be positive


def validate_list_of_type(values: List[Any], expected_type: type, name: str = "list") -> None:
    """
    Validate list contains only elements of expected type.
    
    Args:
        values: List to validate
        expected_type: Expected type of elements
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(values, list):
        raise ValidationError(
            f"{name} must be a list, got {type(values)}",
            details={"expected_type": "list", "actual_type": type(values).__name__}
        )
    
    for i, value in enumerate(values):
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"{name}[{i}] must be {expected_type.__name__}, got {type(value).__name__}",
                details={"index": i, "expected_type": expected_type.__name__, "actual_type": type(value).__name__}
            )


class Validator:
    """Context manager for batch validation."""
    
    def __init__(self):
        self.errors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors:
            error_messages = [str(e) for e in self.errors]
            raise ValidationError(
                f"Multiple validation errors: {'; '.join(error_messages)}",
                details={"error_count": len(self.errors), "errors": error_messages}
            )
        return False
    
    def validate(self, validation_func, *args, **kwargs):
        """Run validation function and collect any errors."""
        try:
            validation_func(*args, **kwargs)
        except ValidationError as e:
            self.errors.append(e)