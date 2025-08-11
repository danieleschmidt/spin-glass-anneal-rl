"""Constraint encoding and penalty methods for Ising models."""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from spin_glass_rl.core.ising_model import IsingModel


class ConstraintType(Enum):
    """Types of constraints."""
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    LOGICAL = "logical"
    CARDINALITY = "cardinality"
    CUSTOM = "custom"


@dataclass
class ConstraintTerm:
    """Individual constraint term."""
    spins: List[int]
    coefficients: List[float]
    constraint_type: ConstraintType
    target_value: float
    penalty_weight: float
    description: str = ""


class Constraint(ABC):
    """Abstract base class for constraints."""
    
    def __init__(self, penalty_weight: float = 1.0, description: str = ""):
        self.penalty_weight = penalty_weight
        self.description = description
    
    @abstractmethod
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate constraint violation."""
        pass
    
    @abstractmethod
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Get penalty terms to add to Ising model."""
        pass


class EqualityConstraint(Constraint):
    """Equality constraint: Σ c_i * s_i = target."""
    
    def __init__(
        self,
        spins: List[int],
        coefficients: List[float],
        target: float,
        penalty_weight: float = 1.0,
        description: str = ""
    ):
        super().__init__(penalty_weight, description)
        self.spins = spins
        self.coefficients = coefficients
        self.target = target
    
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate constraint violation."""
        value = sum(c * spins[i].item() for i, c in zip(self.spins, self.coefficients))
        violation = (value - self.target) ** 2
        return self.penalty_weight * violation
    
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Convert to quadratic penalty terms."""
        # Quadratic penalty: λ(Σ c_i * s_i - target)²
        # Expanded: λ[Σ c_i² * s_i² + 2*Σ c_i*c_j*s_i*s_j + target² - 2*target*Σ c_i*s_i]
        
        terms = []
        
        # Quadratic terms (s_i² = 1 for Ising spins)
        for i, c_i in zip(self.spins, self.coefficients):
            linear_coeff = self.penalty_weight * (c_i**2 - 2*self.target*c_i)
            terms.append(([i], [linear_coeff]))
        
        # Cross terms
        for idx1, (i, c_i) in enumerate(zip(self.spins, self.coefficients)):
            for idx2, (j, c_j) in enumerate(zip(self.spins, self.coefficients)):
                if idx1 < idx2:  # Avoid double counting
                    coupling_coeff = 2 * self.penalty_weight * c_i * c_j
                    terms.append(([i, j], [coupling_coeff]))
        
        return terms


class InequalityConstraint(Constraint):
    """Inequality constraint: Σ c_i * s_i ≤ target."""
    
    def __init__(
        self,
        spins: List[int],
        coefficients: List[float],
        target: float,
        penalty_weight: float = 1.0,
        description: str = ""
    ):
        super().__init__(penalty_weight, description)
        self.spins = spins
        self.coefficients = coefficients
        self.target = target
    
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate constraint violation."""
        value = sum(c * spins[i].item() for i, c in zip(self.spins, self.coefficients))
        violation = max(0, value - self.target) ** 2
        return self.penalty_weight * violation
    
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Convert to quadratic penalty using slack variables."""
        # For now, use simple quadratic penalty
        # In full implementation, would introduce auxiliary spins
        return EqualityConstraint(
            self.spins, self.coefficients, self.target, self.penalty_weight
        ).get_penalty_terms()


class CardinalityConstraint(Constraint):
    """Cardinality constraint: exactly k spins should be +1."""
    
    def __init__(
        self,
        spins: List[int],
        k: int,
        penalty_weight: float = 1.0,
        description: str = ""
    ):
        super().__init__(penalty_weight, description)
        self.spins = spins
        self.k = k
    
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate constraint violation."""
        # Count spins in +1 state
        count = sum(1 for i in self.spins if spins[i].item() == 1)
        violation = (count - self.k) ** 2
        return self.penalty_weight * violation
    
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Convert to quadratic penalty terms."""
        # (Σ (1+s_i)/2 - k)² = (Σ s_i - (2k - n))²/4
        # where n is number of spins in constraint
        n = len(self.spins)
        target = 2 * self.k - n
        coefficients = [1.0] * n
        
        eq_constraint = EqualityConstraint(
            self.spins, coefficients, target, self.penalty_weight / 4
        )
        return eq_constraint.get_penalty_terms()


class LogicalConstraint(Constraint):
    """Logical constraints (AND, OR, NOT)."""
    
    def __init__(
        self,
        operation: str,
        spins: List[int],
        penalty_weight: float = 1.0,
        description: str = ""
    ):
        super().__init__(penalty_weight, description)
        self.operation = operation.upper()
        self.spins = spins
        
        if self.operation not in ["AND", "OR", "NOT", "XOR"]:
            raise ValueError(f"Unknown logical operation: {operation}")
    
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate logical constraint violation."""
        # Convert Ising spins {-1, +1} to binary {0, 1}
        binary_spins = [(spins[i].item() + 1) // 2 for i in self.spins]
        
        if self.operation == "AND":
            # All spins should be +1
            violation = sum(1 - b for b in binary_spins)
        elif self.operation == "OR":
            # At least one spin should be +1
            violation = 1 if sum(binary_spins) == 0 else 0
        elif self.operation == "NOT":
            # First spin should be -1
            violation = binary_spins[0] if len(binary_spins) > 0 else 0
        elif self.operation == "XOR":
            # Odd number of spins should be +1
            violation = (sum(binary_spins) % 2) == 0
        
        return self.penalty_weight * violation
    
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Convert logical constraints to quadratic terms."""
        terms = []
        
        if self.operation == "AND":
            # All spins +1: minimize Σ(1-s_i)/2 = minimize Σ(1-s_i)
            for i in self.spins:
                terms.append(([i], [-self.penalty_weight]))
        
        elif self.operation == "OR":
            # At least one +1: minimize when all are -1
            # Penalty when all s_i = -1: Π(1-s_i)/2
            # Approximation: linear penalty on sum
            coeff = -self.penalty_weight / len(self.spins)
            for i in self.spins:
                terms.append(([i], [coeff]))
        
        # Add more logical operations as needed
        
        return terms


class CustomConstraint(Constraint):
    """Custom constraint with user-defined evaluation function."""
    
    def __init__(
        self,
        evaluation_func: Callable[[torch.Tensor], float],
        penalty_terms: List[Tuple[List[int], List[float]]],
        penalty_weight: float = 1.0,
        description: str = ""
    ):
        super().__init__(penalty_weight, description)
        self.evaluation_func = evaluation_func
        self.penalty_terms = penalty_terms
    
    def evaluate(self, spins: torch.Tensor) -> float:
        """Evaluate using custom function."""
        return self.penalty_weight * self.evaluation_func(spins)
    
    def get_penalty_terms(self) -> List[Tuple[List[int], List[float]]]:
        """Return pre-defined penalty terms."""
        return [(spins, [w * self.penalty_weight for w in weights]) 
                for spins, weights in self.penalty_terms]


class ConstraintEncoder:
    """
    Encodes constraints into Ising model penalty terms.
    
    Converts high-level constraints into quadratic penalty terms
    that can be added to the Ising Hamiltonian.
    """
    
    def __init__(self, model: IsingModel):
        self.model = model
        self.constraints: List[Constraint] = []
        self.penalty_total = 0.0
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add constraint to the model."""
        self.constraints.append(constraint)
        self._apply_constraint_to_model(constraint)
    
    def add_equality_constraint(
        self,
        spins: List[int],
        coefficients: List[float],
        target: float,
        penalty_weight: float = 1.0,
        description: str = ""
    ) -> None:
        """Add equality constraint."""
        constraint = EqualityConstraint(
            spins, coefficients, target, penalty_weight, description
        )
        self.add_constraint(constraint)
    
    def add_inequality_constraint(
        self,
        spins: List[int],
        coefficients: List[float],
        target: float,
        penalty_weight: float = 1.0,
        description: str = ""
    ) -> None:
        """Add inequality constraint."""
        constraint = InequalityConstraint(
            spins, coefficients, target, penalty_weight, description
        )
        self.add_constraint(constraint)
    
    def add_cardinality_constraint(
        self,
        spins: List[int],
        k: int,
        penalty_weight: float = 1.0,
        description: str = ""
    ) -> None:
        """Add cardinality constraint."""
        constraint = CardinalityConstraint(spins, k, penalty_weight, description)
        self.add_constraint(constraint)
    
    def add_logical_constraint(
        self,
        operation: str,
        spins: List[int],
        penalty_weight: float = 1.0,
        description: str = ""
    ) -> None:
        """Add logical constraint."""
        constraint = LogicalConstraint(operation, spins, penalty_weight, description)
        self.add_constraint(constraint)
    
    def evaluate_all_constraints(self, spins: Optional[torch.Tensor] = None) -> Dict:
        """Evaluate all constraints and return violations."""
        if spins is None:
            spins = self.model.spins
        
        violations = {}
        total_violation = 0.0
        
        for i, constraint in enumerate(self.constraints):
            violation = constraint.evaluate(spins)
            violations[f"constraint_{i}"] = {
                "violation": violation,
                "description": constraint.description,
                "type": type(constraint).__name__,
            }
            total_violation += violation
        
        violations["total_violation"] = total_violation
        return violations
    
    def get_feasible_solution(self, max_iterations: int = 1000) -> Optional[torch.Tensor]:
        """
        Find a feasible solution that satisfies all constraints.
        
        Uses simple random search for demonstration.
        In practice, would use more sophisticated methods.
        """
        best_spins = None
        best_violation = float('inf')
        
        for _ in range(max_iterations):
            # Random spin configuration
            test_spins = torch.randint(0, 2, (self.model.n_spins,)) * 2 - 1
            test_spins = test_spins.to(self.model.device)
            
            # Evaluate constraints
            violations = self.evaluate_all_constraints(test_spins)
            total_violation = violations["total_violation"]
            
            if total_violation < best_violation:
                best_violation = total_violation
                best_spins = test_spins.clone()
                
                if total_violation == 0:
                    break  # Found feasible solution
        
        return best_spins if best_violation == 0 else None
    
    def _apply_constraint_to_model(self, constraint: Constraint) -> None:
        """Apply constraint penalty terms to the Ising model."""
        penalty_terms = constraint.get_penalty_terms()
        
        for spins, coefficients in penalty_terms:
            if len(spins) == 1:
                # Linear term (external field)
                i = spins[0]
                coeff = coefficients[0]
                current_field = self.model.external_fields[i].item()
                self.model.set_external_field(i, current_field + coeff)
            
            elif len(spins) == 2:
                # Quadratic term (coupling)
                i, j = spins[0], spins[1]
                coeff = coefficients[0]
                current_coupling = self.model.couplings[i, j].item() if not self.model.config.use_sparse else 0
                self.model.set_coupling(i, j, current_coupling + coeff)
            
            else:
                # Higher-order terms using auxiliary variables
                # For now, approximate using penalty method
                print(f"Warning: Higher-order constraint approximated with penalty method")
                
                # Create auxiliary penalty term for higher-order constraints
                aux_penalty = penalty_weight * 10.0  # Increased penalty for complex constraints
                for spin_idx in spin_indices:
                    current_field = self.model.external_fields[spin_idx].item()
                    self.model.set_external_field(spin_idx, current_field + aux_penalty)
    
    def remove_constraint(self, index: int) -> None:
        """Remove constraint by index."""
        if 0 <= index < len(self.constraints):
            # Note: This doesn't remove the penalty terms from the model
            # In practice, would need to rebuild the model
            del self.constraints[index]
    
    def clear_constraints(self) -> None:
        """Clear all constraints."""
        self.constraints.clear()
    
    def get_constraint_summary(self) -> Dict:
        """Get summary of all constraints."""
        summary = {
            "n_constraints": len(self.constraints),
            "constraint_types": {},
            "total_penalty_weight": 0.0,
        }
        
        for constraint in self.constraints:
            constraint_type = type(constraint).__name__
            if constraint_type not in summary["constraint_types"]:
                summary["constraint_types"][constraint_type] = 0
            summary["constraint_types"][constraint_type] += 1
            summary["total_penalty_weight"] += constraint.penalty_weight
        
        return summary
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConstraintEncoder(n_constraints={len(self.constraints)}, "
            f"model_spins={self.model.n_spins})"
        )