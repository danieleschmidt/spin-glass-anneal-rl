"""Base classes for problem formulations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.constraints import ConstraintEncoder


@dataclass
class ProblemSolution:
    """Generic solution representation."""
    variables: Dict[str, Any]
    objective_value: float
    is_feasible: bool
    constraint_violations: Dict[str, float]
    metadata: Dict[str, Any]


class ProblemTemplate(ABC):
    """
    Abstract base class for optimization problems.
    
    Provides interface for converting problem instances to Ising models
    and extracting solutions from spin configurations.
    """
    
    def __init__(self, name: str = "Generic Problem"):
        self.name = name
        self.variables = {}
        self.constraints = []
        self.objective_function = None
        self.ising_model: Optional[IsingModel] = None
        self.constraint_encoder: Optional[ConstraintEncoder] = None
    
    @abstractmethod
    def encode_to_ising(self, **problem_params) -> IsingModel:
        """
        Encode problem instance as Ising model.
        
        Args:
            **problem_params: Problem-specific parameters
            
        Returns:
            IsingModel representing the problem
        """
        pass
    
    @abstractmethod
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """
        Decode spin configuration to problem solution.
        
        Args:
            spins: Spin configuration from Ising model
            
        Returns:
            ProblemSolution with interpreted results
        """
        pass
    
    @abstractmethod
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """
        Validate that solution satisfies problem constraints.
        
        Args:
            solution: Problem solution to validate
            
        Returns:
            True if solution is feasible
        """
        pass
    
    def add_variable(self, name: str, var_type: str, bounds: Optional[Tuple] = None, **kwargs) -> None:
        """Add decision variable to problem."""
        self.variables[name] = {
            "type": var_type,
            "bounds": bounds,
            "kwargs": kwargs
        }
    
    def add_constraint(self, name: str, constraint_func: callable, constraint_type: str = "equality") -> None:
        """Add constraint to problem."""
        self.constraints.append({
            "name": name,
            "function": constraint_func,
            "type": constraint_type
        })
    
    def set_objective(self, objective_func: callable, sense: str = "minimize") -> None:
        """Set objective function."""
        self.objective_function = {
            "function": objective_func,
            "sense": sense
        }
    
    def get_variable_mapping(self) -> Dict[str, List[int]]:
        """Get mapping from variable names to spin indices."""
        if not hasattr(self, '_variable_mapping'):
            return {}
        return self._variable_mapping
    
    def create_ising_model(self, n_spins: int, device: str = "cpu") -> IsingModel:
        """Create empty Ising model for problem."""
        config = IsingModelConfig(
            n_spins=n_spins,
            use_sparse=True,
            device=device
        )
        model = IsingModel(config)
        self.constraint_encoder = ConstraintEncoder(model)
        return model
    
    def solve_with_annealer(self, annealer, **annealer_params) -> ProblemSolution:
        """
        Solve problem using provided annealer.
        
        Args:
            annealer: Annealing algorithm instance
            **annealer_params: Parameters for annealer
            
        Returns:
            Problem solution
        """
        if self.ising_model is None:
            raise ValueError("Problem not encoded as Ising model. Call encode_to_ising() first.")
        
        # Run annealing
        result = annealer.anneal(self.ising_model, **annealer_params)
        
        # Decode solution
        solution = self.decode_solution(result.best_configuration)
        
        # Add annealing metadata
        solution.metadata.update({
            "annealing_result": result,
            "best_energy": result.best_energy,
            "convergence_sweep": result.convergence_sweep,
            "total_time": result.total_time
        })
        
        return solution
    
    def generate_random_instance(self, **params) -> Dict:
        """Generate random problem instance for testing."""
        # Default implementation - subclasses should override
        return {"random_seed": np.random.randint(0, 1000000)}
    
    def benchmark_instance(self, instance_params: Dict, annealer, n_trials: int = 5) -> Dict:
        """
        Benchmark problem instance.
        
        Args:
            instance_params: Problem instance parameters
            annealer: Annealer to use
            n_trials: Number of trials
            
        Returns:
            Benchmark statistics
        """
        results = {
            "objective_values": [],
            "solve_times": [],
            "feasibility_rates": [],
            "constraint_violations": []
        }
        
        for trial in range(n_trials):
            # Encode problem
            self.encode_to_ising(**instance_params)
            
            # Solve
            solution = self.solve_with_annealer(annealer)
            
            # Record results
            results["objective_values"].append(solution.objective_value)
            results["solve_times"].append(solution.metadata.get("total_time", 0))
            results["feasibility_rates"].append(1.0 if solution.is_feasible else 0.0)
            results["constraint_violations"].append(sum(solution.constraint_violations.values()))
        
        # Compute statistics
        return {
            "mean_objective": np.mean(results["objective_values"]),
            "std_objective": np.std(results["objective_values"]),
            "best_objective": np.min(results["objective_values"]),
            "worst_objective": np.max(results["objective_values"]),
            "mean_time": np.mean(results["solve_times"]),
            "std_time": np.std(results["solve_times"]),
            "feasibility_rate": np.mean(results["feasibility_rates"]),
            "mean_violations": np.mean(results["constraint_violations"]),
            "n_trials": n_trials
        }
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None, **plot_params) -> None:
        """Plot solution visualization (to be implemented by subclasses)."""
        print(f"Solution visualization not implemented for {self.name}")
        print(f"Objective value: {solution.objective_value}")
        print(f"Feasible: {solution.is_feasible}")
    
    def export_solution(self, solution: ProblemSolution, filepath: str, format: str = "json") -> None:
        """Export solution to file."""
        import json
        
        # Convert solution to serializable format
        export_data = {
            "problem_name": self.name,
            "objective_value": solution.objective_value,
            "is_feasible": solution.is_feasible,
            "constraint_violations": solution.constraint_violations,
            "variables": self._serialize_variables(solution.variables),
            "metadata": self._serialize_metadata(solution.metadata)
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _serialize_variables(self, variables: Dict) -> Dict:
        """Serialize variables for export."""
        serialized = {}
        for key, value in variables.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Serialize metadata for export."""
        serialized = {}
        for key, value in metadata.items():
            if key == "annealing_result":
                # Skip annealing result object
                continue
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def get_problem_info(self) -> Dict:
        """Get problem information summary."""
        return {
            "name": self.name,
            "n_variables": len(self.variables),
            "n_constraints": len(self.constraints),
            "variable_types": [v["type"] for v in self.variables.values()],
            "constraint_types": [c["type"] for c in self.constraints],
            "objective_sense": self.objective_function["sense"] if self.objective_function else None,
            "ising_spins": self.ising_model.n_spins if self.ising_model else None
        }
    
    def __repr__(self) -> str:
        """String representation."""
        info = self.get_problem_info()
        return (
            f"{self.name}("
            f"variables={info['n_variables']}, "
            f"constraints={info['n_constraints']}, "
            f"spins={info['ising_spins']})"
        )