"""Input validation and security for optimization problems."""

import re
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRule:
    """Input validation rule."""
    name: str
    description: str
    validator: callable
    severity: str = "error"  # error, warning, info
    

class InputValidator:
    """Comprehensive input validation for optimization problems."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        self.level = level
        self.rules = self._initialize_rules()
        self.violations = []
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize validation rules based on level."""
        rules = [
            # Basic rules
            ValidationRule(
                "non_negative_size",
                "Problem size must be non-negative",
                lambda x: isinstance(x, int) and x >= 0,
                "error"
            ),
            ValidationRule(
                "reasonable_size",
                "Problem size should be reasonable (< 100k)",
                lambda x: isinstance(x, int) and x < 100000,
                "warning"
            ),
            ValidationRule(
                "finite_numeric",
                "Numeric values must be finite",
                lambda x: self._is_finite_numeric(x),
                "error"
            ),
            ValidationRule(
                "safe_array_size",
                "Arrays should not exceed memory limits",
                lambda x: self._check_array_size(x),
                "error"
            )
        ]
        
        if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            rules.extend([
                ValidationRule(
                    "no_injection_patterns",
                    "String inputs should not contain injection patterns",
                    lambda x: self._check_injection_patterns(x),
                    "error"
                ),
                ValidationRule(
                    "bounded_numeric_range",
                    "Numeric values should be in reasonable range",
                    lambda x: self._check_numeric_bounds(x),
                    "warning"
                )
            ])
        
        if self.level == ValidationLevel.PARANOID:
            rules.extend([
                ValidationRule(
                    "no_suspicious_paths",
                    "File paths should not contain suspicious patterns",
                    lambda x: self._check_file_path_safety(x),
                    "error"
                ),
                ValidationRule(
                    "limited_memory_usage",
                    "Input should not cause excessive memory usage",
                    lambda x: self._estimate_memory_usage(x),
                    "error"
                )
            ])
        
        return rules
    
    def validate(self, value: Any, context: str = "unknown") -> Dict[str, Any]:
        """
        Validate input value against all applicable rules.
        
        Args:
            value: Value to validate
            context: Context description for error reporting
            
        Returns:
            Validation result with status and messages
        """
        self.violations.clear()
        errors = []
        warnings = []
        
        for rule in self.rules:
            try:
                if not rule.validator(value):
                    violation = {
                        "rule": rule.name,
                        "description": rule.description,
                        "severity": rule.severity,
                        "context": context,
                        "value_type": type(value).__name__
                    }
                    
                    self.violations.append(violation)
                    
                    if rule.severity == "error":
                        errors.append(violation)
                    elif rule.severity == "warning":
                        warnings.append(violation)
                        
            except Exception as e:
                # Validation rule itself failed
                error = {
                    "rule": rule.name,
                    "description": f"Validation rule failed: {e}",
                    "severity": "error",
                    "context": context,
                    "value_type": type(value).__name__
                }
                errors.append(error)
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "context": context,
            "validation_level": self.level.value
        }
    
    def validate_problem_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete problem instance."""
        all_results = {}
        overall_valid = True
        all_errors = []
        all_warnings = []
        
        for key, value in instance.items():
            result = self.validate(value, context=f"problem_instance.{key}")
            all_results[key] = result
            
            if not result["valid"]:
                overall_valid = False
            
            all_errors.extend(result["errors"])
            all_warnings.extend(result["warnings"])
        
        return {
            "valid": overall_valid,
            "field_results": all_results,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "errors": all_errors,
            "warnings": all_warnings
        }
    
    def _is_finite_numeric(self, value: Any) -> bool:
        """Check if value is finite numeric."""
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                import math
                return math.isfinite(value)
            return True
        elif hasattr(value, '__iter__') and not isinstance(value, str):
            try:
                for item in value:
                    if not self._is_finite_numeric(item):
                        return False
                return True
            except Exception:
                return False
        return True
    
    def _check_array_size(self, value: Any) -> bool:
        """Check if array size is reasonable."""
        if hasattr(value, '__len__'):
            try:
                size = len(value)
                # Check for different limits based on validation level
                if self.level == ValidationLevel.BASIC:
                    return size < 1000000  # 1M elements
                elif self.level == ValidationLevel.STRICT:
                    return size < 100000   # 100K elements
                else:  # PARANOID
                    return size < 10000    # 10K elements
            except Exception:
                return False
        return True
    
    def _check_injection_patterns(self, value: Any) -> bool:
        """Check for potential injection patterns in strings."""
        if isinstance(value, str):
            # Common injection patterns
            dangerous_patterns = [
                r'[;&|`$()]',  # Shell injection
                r'<script',     # XSS
                r'javascript:',  # JavaScript injection
                r'\.\./.*',     # Directory traversal
                r'file://',     # File protocol
                r'exec\(',      # Code execution
                r'eval\(',      # Code evaluation
                r'import\s+os', # Python import
                r'__.*__',      # Python magic methods
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False
        
        return True
    
    def _check_numeric_bounds(self, value: Any) -> bool:
        """Check if numeric values are in reasonable bounds."""
        if isinstance(value, (int, float)):
            # Reasonable bounds for optimization problems
            return -1e12 <= value <= 1e12
        elif hasattr(value, '__iter__') and not isinstance(value, str):
            try:
                for item in value:
                    if not self._check_numeric_bounds(item):
                        return False
                return True
            except Exception:
                return False
        return True
    
    def _check_file_path_safety(self, value: Any) -> bool:
        """Check if file path is safe."""
        if isinstance(value, str) and ('/' in value or '\\' in value):
            # Looks like a file path
            normalized = os.path.normpath(value)
            
            # Check for directory traversal
            if '..' in normalized:
                return False
            
            # Check for absolute paths in paranoid mode
            if os.path.isabs(normalized):
                return False
            
            # Check for system directories
            dangerous_prefixes = ['/etc', '/sys', '/proc', '/dev', 'C:\\Windows']
            for prefix in dangerous_prefixes:
                if normalized.startswith(prefix):
                    return False
        
        return True
    
    def _estimate_memory_usage(self, value: Any) -> bool:
        """Estimate if value would cause excessive memory usage."""
        try:
            import sys
            size = sys.getsizeof(value)
            
            # Different limits based on validation level
            if self.level == ValidationLevel.PARANOID:
                limit = 10 * 1024 * 1024  # 10MB
            else:
                limit = 100 * 1024 * 1024  # 100MB
            
            return size < limit
        except Exception:
            # If we can't estimate, be conservative in paranoid mode
            return self.level != ValidationLevel.PARANOID
    
    def get_validation_report(self) -> str:
        """Generate human-readable validation report."""
        if not self.violations:
            return "✅ All validations passed"
        
        report = ["🛡️ Input Validation Report", "=" * 30]
        
        errors = [v for v in self.violations if v["severity"] == "error"]
        warnings = [v for v in self.violations if v["severity"] == "warning"]
        
        if errors:
            report.append(f"\n❌ Errors ({len(errors)}):")
            for error in errors:
                report.append(f"  - {error['context']}: {error['description']}")
        
        if warnings:
            report.append(f"\n⚠️ Warnings ({len(warnings)}):")
            for warning in warnings:
                report.append(f"  - {warning['context']}: {warning['description']}")
        
        return "\n".join(report)


class SecureConfigValidator:
    """Validator for configuration objects."""
    
    @staticmethod
    def validate_annealer_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate annealer configuration."""
        validator = InputValidator(ValidationLevel.STRICT)
        results = {}
        
        # Check required fields
        required_fields = ["n_sweeps", "initial_temp", "final_temp"]
        for field in required_fields:
            if field not in config:
                results[field] = {
                    "valid": False,
                    "errors": [{"description": f"Required field {field} missing"}]
                }
            else:
                results[field] = validator.validate(config[field], f"config.{field}")
        
        # Validate ranges
        if "n_sweeps" in config:
            if not (1 <= config["n_sweeps"] <= 1000000):
                results["n_sweeps"]["errors"].append({
                    "description": "n_sweeps should be between 1 and 1,000,000"
                })
        
        if "initial_temp" in config and "final_temp" in config:
            if config["initial_temp"] <= config["final_temp"]:
                results["temperature_schedule"] = {
                    "valid": False,
                    "errors": [{"description": "initial_temp must be > final_temp"}]
                }
        
        return results
    
    @staticmethod
    def validate_ising_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Ising model configuration."""
        validator = InputValidator(ValidationLevel.STRICT)
        results = {}
        
        # Check n_spins
        if "n_spins" in config:
            result = validator.validate(config["n_spins"], "config.n_spins")
            if config["n_spins"] <= 0 or config["n_spins"] > 100000:
                result["errors"].append({
                    "description": "n_spins should be between 1 and 100,000"
                })
            results["n_spins"] = result
        
        return results


def validate_optimization_input(
    problem_data: Dict[str, Any],
    level: ValidationLevel = ValidationLevel.STRICT
) -> Dict[str, Any]:
    """
    Comprehensive validation for optimization problem inputs.
    
    Args:
        problem_data: Input data for optimization problem
        level: Validation strictness level
        
    Returns:
        Validation result with detailed feedback
    """
    validator = InputValidator(level)
    result = validator.validate_problem_instance(problem_data)
    
    # Add validation summary
    result["summary"] = {
        "total_fields": len(problem_data),
        "valid_fields": sum(1 for r in result["field_results"].values() if r["valid"]),
        "error_fields": sum(1 for r in result["field_results"].values() if not r["valid"]),
        "validation_level": level.value,
        "passed": result["valid"]
    }
    
    return result


# Quick test function
def test_input_validation():
    """Test input validation framework."""
    print("🛡️ Testing Input Validation Framework...")
    
    # Test basic validation
    validator = InputValidator(ValidationLevel.STRICT)
    
    # Test valid inputs
    valid_tests = [
        (42, "positive_integer"),
        ([1, 2, 3, 4], "small_array"),
        ({"n_spins": 100, "temp": 1.0}, "config_dict"),
        ("safe_filename.txt", "filename")
    ]
    
    for value, context in valid_tests:
        result = validator.validate(value, context)
        print(f"✅ {context}: {'PASS' if result['valid'] else 'FAIL'}")
    
    # Test invalid inputs
    invalid_tests = [
        (-1, "negative_integer"),
        (float('inf'), "infinite_value"),
        ("../../../etc/passwd", "dangerous_path"),
        (list(range(200000)), "huge_array")
    ]
    
    for value, context in invalid_tests:
        result = validator.validate(value, context)
        print(f"❌ {context}: {'FAIL' if not result['valid'] else 'UNEXPECTED_PASS'}")
        if result['errors']:
            print(f"   Error: {result['errors'][0]['description']}")
    
    # Test problem instance validation
    problem_instance = {
        "n_tasks": 10,
        "n_agents": 3,
        "time_horizon": 100.0,
        "task_durations": [5.0, 8.0, 12.0, 6.0, 9.0, 15.0, 7.0, 11.0, 4.0, 13.0]
    }
    
    instance_result = validate_optimization_input(problem_instance)
    print(f"✅ Problem instance validation: {'PASS' if instance_result['valid'] else 'FAIL'}")
    print(f"   Summary: {instance_result['summary']}")
    
    # Test configuration validation
    config = {
        "n_sweeps": 1000,
        "initial_temp": 10.0,
        "final_temp": 0.01
    }
    
    config_result = SecureConfigValidator.validate_annealer_config(config)
    all_valid = all(r.get("valid", True) for r in config_result.values())
    print(f"✅ Config validation: {'PASS' if all_valid else 'FAIL'}")
    
    print("🛡️ Input Validation Framework test completed!")


if __name__ == "__main__":
    test_input_validation()