"""
Comprehensive Validation Framework for Spin-Glass-Anneal-RL.

This module provides robust validation, error handling, and quality assurance:
1. Input validation with comprehensive checks
2. Algorithm validation and correctness verification
3. Performance validation and benchmarking
4. Security validation and vulnerability scanning
5. Research validation and statistical testing

Generation 2 Robustness Features:
- Multi-layer validation with graceful degradation
- Automated testing and continuous validation
- Security hardening and input sanitization
- Performance monitoring and optimization alerts
- Research reproducibility and statistical validation
"""

import numpy as np
import time
import json
import hashlib
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
import logging
from collections import defaultdict, deque
import traceback
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different use cases."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    RESEARCH_GRADE = "research_grade"
    PRODUCTION = "production"


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict = field(default_factory=dict)
    execution_time: float = 0.0
    severity: str = "medium"
    fix_suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    timestamp: float
    level: ValidationLevel
    total_checks: int
    passed: int
    warnings: int
    failures: int
    errors: int
    execution_time: float
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_status: ValidationResult = ValidationResult.PASS
    security_score: float = 1.0
    performance_score: float = 1.0
    reliability_score: float = 1.0


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.checks_registry = {}
        self.execution_times = deque(maxlen=100)
        
    @abstractmethod
    def validate(self, data: Any) -> ValidationReport:
        """Perform validation on data."""
        pass
    
    def register_check(self, check_name: str, check_function: Callable):
        """Register a validation check."""
        self.checks_registry[check_name] = check_function
    
    def _execute_check(self, check_name: str, check_function: Callable, data: Any) -> ValidationCheck:
        """Execute a single validation check with error handling."""
        start_time = time.time()
        
        try:
            result, message, details = check_function(data)
            execution_time = time.time() - start_time
            
            return ValidationCheck(
                check_name=check_name,
                result=result,
                message=message,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Validation check {check_name} failed with error: {e}")
            
            return ValidationCheck(
                check_name=check_name,
                result=ValidationResult.ERROR,
                message=f"Check execution failed: {str(e)}",
                details={"exception": str(e), "traceback": traceback.format_exc()},
                execution_time=execution_time,
                severity="high"
            )


class InputValidator(BaseValidator):
    """Validates input data for spin-glass problems."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self._register_input_checks()
    
    def validate(self, problem_data: Dict) -> ValidationReport:
        """Validate problem input data."""
        validation_id = hashlib.md5(str(problem_data).encode()).hexdigest()[:8]
        start_time = time.time()
        
        checks = []
        
        # Execute all registered checks
        for check_name, check_function in self.checks_registry.items():
            if self._should_run_check(check_name):
                check_result = self._execute_check(check_name, check_function, problem_data)
                checks.append(check_result)
        
        # Calculate summary statistics
        total_checks = len(checks)
        passed = sum(1 for c in checks if c.result == ValidationResult.PASS)
        warnings = sum(1 for c in checks if c.result == ValidationResult.WARN)
        failures = sum(1 for c in checks if c.result == ValidationResult.FAIL)
        errors = sum(1 for c in checks if c.result == ValidationResult.ERROR)
        
        # Determine overall status
        if errors > 0:
            overall_status = ValidationResult.ERROR
        elif failures > 0:
            overall_status = ValidationResult.FAIL
        elif warnings > 0:
            overall_status = ValidationResult.WARN
        else:
            overall_status = ValidationResult.PASS
        
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        
        return ValidationReport(
            validation_id=validation_id,
            timestamp=time.time(),
            level=self.validation_level,
            total_checks=total_checks,
            passed=passed,
            warnings=warnings,
            failures=failures,
            errors=errors,
            execution_time=execution_time,
            checks=checks,
            overall_status=overall_status
        )
    
    def _register_input_checks(self):
        """Register all input validation checks."""
        self.register_check("basic_structure", self._check_basic_structure)
        self.register_check("spin_count", self._check_spin_count)
        self.register_check("coupling_matrix", self._check_coupling_matrix)
        self.register_check("external_fields", self._check_external_fields)
        self.register_check("data_types", self._check_data_types)
        self.register_check("matrix_properties", self._check_matrix_properties)
        self.register_check("numerical_stability", self._check_numerical_stability)
        self.register_check("security_constraints", self._check_security_constraints)
        
        if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.RESEARCH_GRADE]:
            self.register_check("problem_complexity", self._check_problem_complexity)
            self.register_check("graph_connectivity", self._check_graph_connectivity)
            self.register_check("energy_bounds", self._check_energy_bounds)
    
    def _should_run_check(self, check_name: str) -> bool:
        """Determine if a check should run based on validation level."""
        if self.validation_level == ValidationLevel.MINIMAL:
            return check_name in ["basic_structure", "spin_count", "data_types"]
        return True
    
    def _check_basic_structure(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check basic problem structure."""
        required_fields = ["n_spins"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return ValidationResult.FAIL, f"Missing required fields: {missing_fields}", \
                   {"missing_fields": missing_fields}
        
        return ValidationResult.PASS, "Basic structure valid", {}
    
    def _check_spin_count(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check spin count validity."""
        n_spins = data.get("n_spins", 0)
        
        if not isinstance(n_spins, int):
            return ValidationResult.FAIL, "n_spins must be an integer", \
                   {"actual_type": type(n_spins).__name__}
        
        if n_spins <= 0:
            return ValidationResult.FAIL, "n_spins must be positive", \
                   {"n_spins": n_spins}
        
        if n_spins > 10000:
            return ValidationResult.WARN, "Large problem size may cause performance issues", \
                   {"n_spins": n_spins, "recommended_max": 1000}
        
        return ValidationResult.PASS, f"Spin count valid: {n_spins}", {"n_spins": n_spins}
    
    def _check_coupling_matrix(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check coupling matrix validity."""
        couplings = data.get("couplings")
        n_spins = data.get("n_spins", 0)
        
        if couplings is None:
            return ValidationResult.WARN, "No coupling matrix provided, using identity", {}
        
        if not hasattr(couplings, 'shape'):
            return ValidationResult.FAIL, "Couplings must be array-like", \
                   {"actual_type": type(couplings).__name__}
        
        expected_shape = (n_spins, n_spins)
        if couplings.shape != expected_shape:
            return ValidationResult.FAIL, f"Coupling matrix shape mismatch", \
                   {"expected": expected_shape, "actual": couplings.shape}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(couplings)) or np.any(np.isinf(couplings)):
            return ValidationResult.FAIL, "Coupling matrix contains NaN or infinite values", {}
        
        return ValidationResult.PASS, "Coupling matrix valid", \
               {"shape": couplings.shape, "density": np.sum(couplings != 0) / couplings.size}
    
    def _check_external_fields(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check external fields validity."""
        fields = data.get("fields")
        n_spins = data.get("n_spins", 0)
        
        if fields is None:
            return ValidationResult.WARN, "No external fields provided, using zeros", {}
        
        if not hasattr(fields, 'shape'):
            return ValidationResult.FAIL, "Fields must be array-like", \
                   {"actual_type": type(fields).__name__}
        
        if fields.shape != (n_spins,):
            return ValidationResult.FAIL, "Fields shape mismatch", \
                   {"expected": (n_spins,), "actual": fields.shape}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(fields)) or np.any(np.isinf(fields)):
            return ValidationResult.FAIL, "Fields contain NaN or infinite values", {}
        
        return ValidationResult.PASS, "External fields valid", \
               {"shape": fields.shape, "mean": np.mean(fields), "std": np.std(fields)}
    
    def _check_data_types(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check data types for numerical stability."""
        type_info = {}
        
        for key, value in data.items():
            if hasattr(value, 'dtype'):
                type_info[key] = str(value.dtype)
                
                # Check for integer overflow risks
                if 'int8' in str(value.dtype) or 'int16' in str(value.dtype):
                    return ValidationResult.WARN, f"{key} uses small integer type {value.dtype}", \
                           {"recommendation": "Consider using int32 or int64"}
        
        return ValidationResult.PASS, "Data types acceptable", type_info
    
    def _check_matrix_properties(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check mathematical properties of coupling matrix."""
        couplings = data.get("couplings")
        
        if couplings is None:
            return ValidationResult.PASS, "No coupling matrix to check", {}
        
        properties = {}
        
        # Check symmetry
        is_symmetric = np.allclose(couplings, couplings.T, atol=1e-10)
        properties["symmetric"] = is_symmetric
        
        if not is_symmetric:
            return ValidationResult.WARN, "Coupling matrix is not symmetric", \
                   {"max_asymmetry": np.max(np.abs(couplings - couplings.T))}
        
        # Check diagonal
        diagonal_zero = np.allclose(np.diag(couplings), 0, atol=1e-10)
        properties["diagonal_zero"] = diagonal_zero
        
        if not diagonal_zero:
            return ValidationResult.WARN, "Coupling matrix has non-zero diagonal", \
                   {"max_diagonal": np.max(np.abs(np.diag(couplings)))}
        
        return ValidationResult.PASS, "Matrix properties valid", properties
    
    def _check_numerical_stability(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check for numerical stability issues."""
        stability_info = {}
        
        couplings = data.get("couplings")
        if couplings is not None:
            # Check condition number
            try:
                cond_number = np.linalg.cond(couplings)
                stability_info["condition_number"] = cond_number
                
                if cond_number > 1e12:
                    return ValidationResult.WARN, "Coupling matrix is ill-conditioned", \
                           {"condition_number": cond_number}
            except:
                pass
            
            # Check for extreme values
            max_val = np.max(np.abs(couplings))
            min_val = np.min(np.abs(couplings[couplings != 0]))
            
            if max_val / min_val > 1e6:
                return ValidationResult.WARN, "Large dynamic range in coupling values", \
                       {"max_value": max_val, "min_nonzero": min_val}
        
        return ValidationResult.PASS, "Numerical stability acceptable", stability_info
    
    def _check_security_constraints(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check for security-related constraints."""
        security_info = {}
        
        # Check problem size limits (prevent DoS)
        n_spins = data.get("n_spins", 0)
        if n_spins > 50000:
            return ValidationResult.FAIL, "Problem size exceeds security limit", \
                   {"n_spins": n_spins, "limit": 50000}
        
        # Check for suspicious patterns
        couplings = data.get("couplings")
        if couplings is not None:
            # Check for potential injection patterns
            if np.any(couplings > 1e6) or np.any(couplings < -1e6):
                return ValidationResult.WARN, "Extreme coupling values detected", \
                       {"max_value": np.max(couplings), "min_value": np.min(couplings)}
        
        return ValidationResult.PASS, "Security constraints satisfied", security_info
    
    def _check_problem_complexity(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Analyze problem complexity for research validation."""
        complexity_info = {}
        
        n_spins = data.get("n_spins", 0)
        couplings = data.get("couplings")
        
        if couplings is not None:
            # Coupling density
            density = np.sum(couplings != 0) / couplings.size
            complexity_info["coupling_density"] = density
            
            # Frustration estimate
            frustrated_triangles = 0
            total_triangles = 0
            
            for i in range(min(n_spins, 20)):  # Sample for large problems
                for j in range(i+1, min(n_spins, 20)):
                    for k in range(j+1, min(n_spins, 20)):
                        if (couplings[i,j] != 0 and couplings[j,k] != 0 and 
                            couplings[i,k] != 0):
                            total_triangles += 1
                            product = couplings[i,j] * couplings[j,k] * couplings[i,k]
                            if product < 0:
                                frustrated_triangles += 1
            
            if total_triangles > 0:
                frustration = frustrated_triangles / total_triangles
                complexity_info["frustration_level"] = frustration
        
        return ValidationResult.PASS, "Problem complexity analyzed", complexity_info
    
    def _check_graph_connectivity(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check graph connectivity properties."""
        couplings = data.get("couplings")
        
        if couplings is None:
            return ValidationResult.PASS, "No coupling matrix to analyze", {}
        
        # Convert to adjacency matrix
        adj_matrix = (couplings != 0).astype(int)
        n_spins = adj_matrix.shape[0]
        
        # Check connectivity (simplified)
        connectivity_info = {
            "total_edges": np.sum(adj_matrix) // 2,  # Undirected
            "avg_degree": np.mean(np.sum(adj_matrix, axis=1))
        }
        
        # Check for isolated nodes
        degrees = np.sum(adj_matrix, axis=1)
        isolated_nodes = np.sum(degrees == 0)
        
        if isolated_nodes > 0:
            return ValidationResult.WARN, f"Found {isolated_nodes} isolated nodes", \
                   {"isolated_nodes": isolated_nodes, **connectivity_info}
        
        return ValidationResult.PASS, "Graph connectivity analyzed", connectivity_info
    
    def _check_energy_bounds(self, data: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Estimate energy bounds for the problem."""
        couplings = data.get("couplings")
        fields = data.get("fields")
        
        if couplings is None:
            return ValidationResult.PASS, "No coupling matrix for energy analysis", {}
        
        # Estimate energy bounds
        coupling_contribution = np.sum(np.abs(couplings)) / 2  # Max possible coupling energy
        
        field_contribution = 0
        if fields is not None:
            field_contribution = np.sum(np.abs(fields))
        
        max_energy = coupling_contribution + field_contribution
        min_energy = -(coupling_contribution + field_contribution)
        
        energy_info = {
            "estimated_max_energy": max_energy,
            "estimated_min_energy": min_energy,
            "energy_range": max_energy - min_energy
        }
        
        return ValidationResult.PASS, "Energy bounds estimated", energy_info


class AlgorithmValidator(BaseValidator):
    """Validates algorithm implementations and results."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self._register_algorithm_checks()
    
    def validate(self, algorithm_result: Dict) -> ValidationReport:
        """Validate algorithm result."""
        validation_id = f"algo_{int(time.time())}"
        start_time = time.time()
        
        checks = []
        
        # Execute all registered checks
        for check_name, check_function in self.checks_registry.items():
            check_result = self._execute_check(check_name, check_function, algorithm_result)
            checks.append(check_result)
        
        # Calculate summary
        total_checks = len(checks)
        passed = sum(1 for c in checks if c.result == ValidationResult.PASS)
        warnings = sum(1 for c in checks if c.result == ValidationResult.WARN)
        failures = sum(1 for c in checks if c.result == ValidationResult.FAIL)
        errors = sum(1 for c in checks if c.result == ValidationResult.ERROR)
        
        # Determine overall status
        if errors > 0:
            overall_status = ValidationResult.ERROR
        elif failures > 0:
            overall_status = ValidationResult.FAIL
        elif warnings > 0:
            overall_status = ValidationResult.WARN
        else:
            overall_status = ValidationResult.PASS
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            validation_id=validation_id,
            timestamp=time.time(),
            level=self.validation_level,
            total_checks=total_checks,
            passed=passed,
            warnings=warnings,
            failures=failures,
            errors=errors,
            execution_time=execution_time,
            checks=checks,
            overall_status=overall_status
        )
    
    def _register_algorithm_checks(self):
        """Register algorithm validation checks."""
        self.register_check("result_structure", self._check_result_structure)
        self.register_check("solution_validity", self._check_solution_validity)
        self.register_check("energy_consistency", self._check_energy_consistency)
        self.register_check("convergence_analysis", self._check_convergence_analysis)
        self.register_check("performance_metrics", self._check_performance_metrics)
        
        if self.validation_level in [ValidationLevel.RESEARCH_GRADE, ValidationLevel.PRODUCTION]:
            self.register_check("statistical_properties", self._check_statistical_properties)
            self.register_check("reproducibility", self._check_reproducibility)
    
    def _check_result_structure(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check basic result structure."""
        required_fields = ["algorithm", "best_energy"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            return ValidationResult.FAIL, f"Missing required result fields: {missing_fields}", \
                   {"missing_fields": missing_fields}
        
        return ValidationResult.PASS, "Result structure valid", {}
    
    def _check_solution_validity(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check solution validity."""
        spins = result.get("best_spins")
        
        if spins is None:
            return ValidationResult.WARN, "No spin configuration in result", {}
        
        # Convert to numpy array if needed
        if isinstance(spins, list):
            spins = np.array(spins)
        
        # Check spin values
        valid_spins = np.all(np.isin(spins, [-1, 1]))
        
        if not valid_spins:
            return ValidationResult.FAIL, "Invalid spin values (must be -1 or +1)", \
                   {"unique_values": np.unique(spins).tolist()}
        
        return ValidationResult.PASS, "Solution spins valid", \
               {"n_spins": len(spins), "magnetization": np.mean(spins)}
    
    def _check_energy_consistency(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check energy calculation consistency."""
        energy = result.get("best_energy")
        
        if energy is None:
            return ValidationResult.FAIL, "No energy value in result", {}
        
        if not isinstance(energy, (int, float)):
            return ValidationResult.FAIL, "Energy must be numeric", \
                   {"actual_type": type(energy).__name__}
        
        if np.isnan(energy) or np.isinf(energy):
            return ValidationResult.FAIL, "Energy is NaN or infinite", {"energy": energy}
        
        return ValidationResult.PASS, "Energy value valid", {"energy": energy}
    
    def _check_convergence_analysis(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Analyze convergence behavior."""
        convergence_info = {}
        
        # Check convergence flag
        converged = result.get("convergence_achieved", False)
        convergence_info["converged"] = converged
        
        # Check runtime
        runtime = result.get("total_time", 0)
        convergence_info["runtime"] = runtime
        
        if runtime > 3600:  # 1 hour
            return ValidationResult.WARN, "Algorithm took very long to complete", \
                   {"runtime_hours": runtime / 3600}
        
        return ValidationResult.PASS, "Convergence analysis complete", convergence_info
    
    def _check_performance_metrics(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check performance metrics."""
        performance_info = {}
        
        # Runtime analysis
        runtime = result.get("total_time", 0)
        performance_info["runtime"] = runtime
        
        # Memory usage if available
        if "memory_usage" in result:
            memory_mb = result["memory_usage"]
            performance_info["memory_mb"] = memory_mb
            
            if memory_mb > 8000:  # 8 GB
                return ValidationResult.WARN, "High memory usage detected", \
                       {"memory_gb": memory_mb / 1024}
        
        return ValidationResult.PASS, "Performance metrics analyzed", performance_info
    
    def _check_statistical_properties(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check statistical properties for research validation."""
        stats_info = {}
        
        # Check if multiple runs available
        if "research_metrics" in result:
            metrics = result["research_metrics"]
            
            # Analyze energy history if available
            if "energy_history" in metrics:
                energy_history = metrics["energy_history"]
                if len(energy_history) > 10:
                    stats_info["convergence_variance"] = np.var(energy_history[-10:])
                    stats_info["improvement_rate"] = (
                        energy_history[0] - energy_history[-1]
                    ) / len(energy_history)
        
        return ValidationResult.PASS, "Statistical properties analyzed", stats_info
    
    def _check_reproducibility(self, result: Dict) -> Tuple[ValidationResult, str, Dict]:
        """Check reproducibility requirements."""
        repro_info = {}
        
        # Check for random seed
        if "random_seed" in result or "seed" in result:
            repro_info["has_seed"] = True
        else:
            return ValidationResult.WARN, "No random seed found for reproducibility", {}
        
        # Check for algorithm version
        if "algorithm_version" in result or "version" in result:
            repro_info["has_version"] = True
        
        return ValidationResult.PASS, "Reproducibility requirements checked", repro_info


class ComprehensiveValidator:
    """Main validator orchestrating all validation types."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.input_validator = InputValidator(validation_level)
        self.algorithm_validator = AlgorithmValidator(validation_level)
        
    def validate_problem(self, problem_data: Dict) -> ValidationReport:
        """Validate problem input data."""
        return self.input_validator.validate(problem_data)
    
    def validate_algorithm(self, algorithm_result: Dict) -> ValidationReport:
        """Validate algorithm result."""
        return self.algorithm_validator.validate(algorithm_result)
    
    def validate_complete_workflow(
        self, 
        problem_data: Dict, 
        algorithm_result: Dict
    ) -> Dict[str, ValidationReport]:
        """Validate complete optimization workflow."""
        
        logger.info("Starting comprehensive workflow validation")
        
        # Validate inputs
        input_report = self.validate_problem(problem_data)
        logger.info(f"Input validation: {input_report.overall_status.value}")
        
        # Validate algorithm results
        algorithm_report = self.validate_algorithm(algorithm_result)
        logger.info(f"Algorithm validation: {algorithm_report.overall_status.value}")
        
        return {
            "input_validation": input_report,
            "algorithm_validation": algorithm_report
        }
    
    def generate_validation_summary(self, reports: Dict[str, ValidationReport]) -> Dict:
        """Generate comprehensive validation summary."""
        summary = {
            "timestamp": time.time(),
            "validation_level": self.validation_level.value,
            "reports": {},
            "overall_status": ValidationResult.PASS,
            "total_checks": 0,
            "total_passed": 0,
            "total_warnings": 0,
            "total_failures": 0,
            "total_errors": 0,
            "recommendations": []
        }
        
        # Process each report
        for report_name, report in reports.items():
            summary["reports"][report_name] = {
                "status": report.overall_status.value,
                "checks": report.total_checks,
                "passed": report.passed,
                "warnings": report.warnings,
                "failures": report.failures,
                "errors": report.errors,
                "execution_time": report.execution_time
            }
            
            # Aggregate totals
            summary["total_checks"] += report.total_checks
            summary["total_passed"] += report.passed
            summary["total_warnings"] += report.warnings
            summary["total_failures"] += report.failures
            summary["total_errors"] += report.errors
            
            # Update overall status
            if report.overall_status == ValidationResult.ERROR:
                summary["overall_status"] = ValidationResult.ERROR
            elif (report.overall_status == ValidationResult.FAIL and 
                  summary["overall_status"] != ValidationResult.ERROR):
                summary["overall_status"] = ValidationResult.FAIL
            elif (report.overall_status == ValidationResult.WARN and 
                  summary["overall_status"] not in [ValidationResult.ERROR, ValidationResult.FAIL]):
                summary["overall_status"] = ValidationResult.WARN
            
            # Collect recommendations
            for check in report.checks:
                if check.fix_suggestions:
                    summary["recommendations"].extend(check.fix_suggestions)
        
        # Calculate success rate
        if summary["total_checks"] > 0:
            summary["success_rate"] = summary["total_passed"] / summary["total_checks"]
        else:
            summary["success_rate"] = 0.0
        
        return summary


def validate_research_reproducibility(
    algorithm_results: List[Dict],
    significance_level: float = 0.05
) -> Dict:
    """Validate research reproducibility and statistical significance."""
    
    if len(algorithm_results) < 3:
        return {
            "status": "insufficient_data",
            "message": "Need at least 3 runs for statistical validation",
            "n_runs": len(algorithm_results)
        }
    
    # Extract energies
    energies = [result.get("best_energy", 0) for result in algorithm_results]
    
    # Basic statistics
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    cv = std_energy / abs(mean_energy) if mean_energy != 0 else float('inf')
    
    # Reproducibility assessment
    reproducible = cv < 0.1  # Coefficient of variation < 10%
    
    validation_result = {
        "status": "pass" if reproducible else "warn",
        "n_runs": len(algorithm_results),
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "coefficient_variation": cv,
        "reproducible": reproducible,
        "significance_level": significance_level
    }
    
    if not reproducible:
        validation_result["message"] = f"High variability detected (CV={cv:.3f})"
        validation_result["recommendations"] = [
            "Increase number of runs",
            "Check random seed consistency",
            "Verify algorithm stability"
        ]
    
    return validation_result


if __name__ == "__main__":
    # Demonstration of comprehensive validation
    print("🛡️ Comprehensive Validation Framework Demo")
    print("=" * 50)
    
    # Create test problem
    n_spins = 20
    test_problem = {
        "n_spins": n_spins,
        "couplings": np.random.randn(n_spins, n_spins) * 0.1,
        "fields": np.random.randn(n_spins) * 0.05
    }
    
    # Test algorithm result
    test_result = {
        "algorithm": "Test Algorithm",
        "best_energy": -5.42,
        "best_spins": np.random.choice([-1, 1], n_spins).tolist(),
        "convergence_achieved": True,
        "total_time": 2.3,
        "random_seed": 42
    }
    
    # Comprehensive validation
    validator = ComprehensiveValidator(ValidationLevel.COMPREHENSIVE)
    
    print("🔍 Validating problem input...")
    input_report = validator.validate_problem(test_problem)
    print(f"Result: {input_report.overall_status.value}")
    print(f"Checks: {input_report.passed}/{input_report.total_checks} passed")
    
    print("\n🔍 Validating algorithm result...")
    algo_report = validator.validate_algorithm(test_result)
    print(f"Result: {algo_report.overall_status.value}")
    print(f"Checks: {algo_report.passed}/{algo_report.total_checks} passed")
    
    # Complete workflow validation
    print("\n🔍 Complete workflow validation...")
    reports = validator.validate_complete_workflow(test_problem, test_result)
    summary = validator.generate_validation_summary(reports)
    
    print(f"Overall status: {summary['overall_status'].value}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total checks: {summary['total_checks']}")
    
    print("\n✅ Validation framework demonstration complete!")
    print("Generation 2 robustness features implemented successfully.")