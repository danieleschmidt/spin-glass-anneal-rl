"""Standard optimization benchmarks for spin-glass annealing."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.coupling_matrix import CouplingMatrix
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution


@dataclass
class BenchmarkInstance:
    """A standard benchmark problem instance."""
    name: str
    problem_type: str
    size: int
    optimal_value: Optional[float]
    description: str
    parameters: Dict[str, Any]


class StandardBenchmarkProblem(ProblemTemplate):
    """Base class for standard benchmark problems."""
    
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size = size
        self.optimal_value: Optional[float] = None
    
    @abstractmethod
    def generate_instance(self, seed: int = 42) -> Dict[str, Any]:
        """Generate benchmark instance."""
        pass
    
    def get_approximation_ratio(self, solution_value: float) -> float:
        """Get approximation ratio vs optimal (if known)."""
        if self.optimal_value is None:
            return float('inf')
        if self.optimal_value == 0:
            return float('inf') if solution_value != 0 else 1.0
        return solution_value / self.optimal_value


class MaxCutProblem(StandardBenchmarkProblem):
    """Maximum Cut problem benchmark."""
    
    def __init__(self, n_vertices: int = 50):
        super().__init__(f"MaxCut-{n_vertices}", n_vertices)
        self.n_vertices = n_vertices
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.edge_weights: Optional[np.ndarray] = None
    
    def generate_instance(self, seed: int = 42) -> Dict[str, Any]:
        """Generate random MaxCut instance."""
        np.random.seed(seed)
        
        # Generate random graph (Erdős–Rényi model)
        edge_prob = 0.5  # Dense graph for harder instances
        self.adjacency_matrix = np.random.rand(self.n_vertices, self.n_vertices)
        self.adjacency_matrix = (self.adjacency_matrix < edge_prob).astype(float)
        
        # Make symmetric and remove self-loops
        self.adjacency_matrix = (self.adjacency_matrix + self.adjacency_matrix.T) / 2
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        # Random edge weights
        self.edge_weights = np.random.uniform(0.5, 2.0, (self.n_vertices, self.n_vertices))
        self.edge_weights = (self.edge_weights + self.edge_weights.T) / 2
        self.edge_weights *= self.adjacency_matrix
        
        return {
            "n_vertices": self.n_vertices,
            "edge_prob": edge_prob,
            "seed": seed
        }
    
    def encode_to_ising(
        self,
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """Encode MaxCut as Ising model."""
        if self.adjacency_matrix is None:
            self.generate_instance()
        
        # For MaxCut: maximize sum of edge weights across cut
        # Ising formulation: minimize -0.5 * sum(w_ij * (1 - s_i * s_j))
        config = IsingModelConfig(
            n_spins=self.n_vertices,
            use_sparse=True,
            device="cpu"
        )
        
        self.ising_model = IsingModel(config)
        
        # Set couplings: J_ij = -0.5 * w_ij
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if self.adjacency_matrix[i, j] > 0:
                    coupling = -0.5 * self.edge_weights[i, j]
                    self.ising_model.set_coupling(i, j, coupling)
        
        return self.ising_model
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode MaxCut solution."""
        binary_spins = (spins + 1) // 2  # Convert {-1,1} to {0,1}
        
        # Calculate cut value
        cut_value = 0.0
        cut_edges = 0
        
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if self.adjacency_matrix[i, j] > 0:
                    if binary_spins[i] != binary_spins[j]:  # Edge crosses cut
                        cut_value += self.edge_weights[i, j]
                        cut_edges += 1
        
        # Get partition sizes
        partition_0 = (binary_spins == 0).sum().item()
        partition_1 = (binary_spins == 1).sum().item()
        
        return ProblemSolution(
            variables={"partition": binary_spins.cpu().numpy()},
            objective_value=cut_value,
            is_feasible=True,  # MaxCut has no hard constraints
            constraint_violations={},
            metadata={
                "cut_edges": cut_edges,
                "partition_sizes": [partition_0, partition_1],
                "balance": abs(partition_0 - partition_1) / self.n_vertices
            }
        )


class QuadraticAssignmentProblem(StandardBenchmarkProblem):
    """Quadratic Assignment Problem benchmark."""
    
    def __init__(self, n_facilities: int = 20):
        super().__init__(f"QAP-{n_facilities}", n_facilities)
        self.n_facilities = n_facilities
        self.flow_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
    
    def generate_instance(self, seed: int = 42) -> Dict[str, Any]:
        """Generate random QAP instance."""
        np.random.seed(seed)
        n = self.n_facilities
        
        # Generate flow matrix (facility interactions)
        self.flow_matrix = np.random.randint(0, 100, (n, n))
        np.fill_diagonal(self.flow_matrix, 0)  # No self-flow
        
        # Generate distance matrix (location distances)
        # Use random points on unit square
        locations = np.random.rand(n, 2)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(locations[i] - locations[j])
                    self.distance_matrix[i, j] = dist
        
        return {
            "n_facilities": n,
            "seed": seed
        }
    
    def encode_to_ising(
        self,
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """Encode QAP as Ising model using binary variables."""
        if self.flow_matrix is None:
            self.generate_instance()
        
        if penalty_weights is None:
            penalty_weights = {"assignment": 1000.0}
        
        n = self.n_facilities
        
        # Binary variables: x_{i,j} = 1 if facility i assigned to location j
        n_spins = n * n
        
        config = IsingModelConfig(
            n_spins=n_spins,
            use_sparse=True,
            device="cpu"
        )
        
        self.ising_model = IsingModel(config)
        
        # Variable mapping
        self._variable_mapping = {}
        spin_idx = 0
        for facility in range(n):
            for location in range(n):
                self._variable_mapping[(facility, location)] = spin_idx
                spin_idx += 1
        
        # Add objective: minimize sum(flow[i,j] * distance[k,l] * x[i,k] * x[j,l])
        # This creates 4-body interactions, approximated with auxiliary variables
        self._add_qap_objective()
        
        # Add constraints: each facility assigned to exactly one location
        self._add_assignment_constraints(penalty_weights["assignment"])
        
        return self.ising_model
    
    def _get_spin_index(self, facility: int, location: int) -> int:
        """Get spin index for facility-location assignment."""
        return self._variable_mapping[(facility, location)]
    
    def _add_qap_objective(self):
        """Add QAP objective using pairwise approximation."""
        n = self.n_facilities
        
        # Simplified quadratic approximation
        for facility_i in range(n):
            for facility_j in range(n):
                if facility_i != facility_j:
                    flow = self.flow_matrix[facility_i, facility_j]
                    
                    for loc_k in range(n):
                        for loc_l in range(n):
                            if loc_k != loc_l:
                                distance = self.distance_matrix[loc_k, loc_l]
                                cost = flow * distance
                                
                                spin_i = self._get_spin_index(facility_i, loc_k)
                                spin_j = self._get_spin_index(facility_j, loc_l)
                                
                                # Add coupling
                                current = self.ising_model.couplings[spin_i, spin_j].item()
                                self.ising_model.set_coupling(spin_i, spin_j, current + cost)
    
    def _add_assignment_constraints(self, penalty_weight: float):
        """Add assignment constraints using penalty method."""
        n = self.n_facilities
        
        # Each facility assigned to exactly one location
        for facility in range(n):
            facility_spins = []
            for location in range(n):
                spin_idx = self._get_spin_index(facility, location)
                facility_spins.append(spin_idx)
            
            # Add quadratic penalty: (sum(x_i) - 1)^2
            for i, spin_i in enumerate(facility_spins):
                # Linear term: -2 * penalty_weight
                current_field = self.ising_model.external_fields[spin_i].item()
                self.ising_model.set_external_field(spin_i, current_field - 2 * penalty_weight)
                
                # Quadratic terms: +penalty_weight for all pairs
                for j, spin_j in enumerate(facility_spins):
                    if i != j:
                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                    current_coupling + penalty_weight)
        
        # Each location assigned to exactly one facility
        for location in range(n):
            location_spins = []
            for facility in range(n):
                spin_idx = self._get_spin_index(facility, location)
                location_spins.append(spin_idx)
            
            # Add quadratic penalty
            for i, spin_i in enumerate(location_spins):
                current_field = self.ising_model.external_fields[spin_i].item()
                self.ising_model.set_external_field(spin_i, current_field - 2 * penalty_weight)
                
                for j, spin_j in enumerate(location_spins):
                    if i != j:
                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                    current_coupling + penalty_weight)
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode QAP solution."""
        binary_spins = (spins + 1) // 2
        n = self.n_facilities
        
        # Extract assignment
        assignment = [-1] * n  # assignment[facility] = location
        assignment_matrix = np.zeros((n, n))
        
        for facility in range(n):
            for location in range(n):
                spin_idx = self._get_spin_index(facility, location)
                if binary_spins[spin_idx].item() == 1:
                    assignment[facility] = location
                    assignment_matrix[facility, location] = 1
        
        # Calculate objective value
        total_cost = 0.0
        for i in range(n):
            for j in range(n):
                if assignment[i] != -1 and assignment[j] != -1:
                    flow = self.flow_matrix[i, j]
                    distance = self.distance_matrix[assignment[i], assignment[j]]
                    total_cost += flow * distance
        
        # Check feasibility
        is_feasible = all(a != -1 for a in assignment)
        constraint_violations = {}
        
        # Check assignment constraints
        for facility in range(n):
            assignments = assignment_matrix[facility, :].sum()
            if assignments != 1:
                constraint_violations[f"facility_{facility}"] = abs(assignments - 1)
        
        for location in range(n):
            assignments = assignment_matrix[:, location].sum()
            if assignments != 1:
                constraint_violations[f"location_{location}"] = abs(assignments - 1)
        
        return ProblemSolution(
            variables={"assignment": assignment},
            objective_value=total_cost,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "n_facilities": n,
                "valid_assignments": sum(1 for a in assignment if a != -1)
            }
        )


class StandardBenchmarkSuite:
    """Suite of standard optimization benchmarks."""
    
    def __init__(self):
        self.problems = {}
        self._register_standard_problems()
    
    def _register_standard_problems(self):
        """Register standard benchmark problems."""
        # MaxCut instances
        for size in [20, 50, 100]:
            problem = MaxCutProblem(size)
            self.problems[f"maxcut_{size}"] = problem
        
        # QAP instances
        for size in [10, 15, 20]:
            problem = QuadraticAssignmentProblem(size)
            self.problems[f"qap_{size}"] = problem
    
    def get_problem(self, name: str) -> StandardBenchmarkProblem:
        """Get problem by name."""
        if name not in self.problems:
            raise ValueError(f"Unknown problem: {name}. Available: {list(self.problems.keys())}")
        return self.problems[name]
    
    def list_problems(self) -> List[str]:
        """List available problems."""
        return list(self.problems.keys())
    
    def run_benchmark(
        self,
        problem_name: str,
        annealer_config: Optional[GPUAnnealerConfig] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Run benchmark on specified problem."""
        problem = self.get_problem(problem_name)
        
        # Generate instance
        instance_params = problem.generate_instance(seed)
        
        # Create default annealer config if not provided
        if annealer_config is None:
            annealer_config = GPUAnnealerConfig(
                n_sweeps=5000,
                initial_temp=10.0,
                final_temp=0.01,
                schedule_type=ScheduleType.GEOMETRIC,
                random_seed=seed
            )
        
        # Encode problem
        ising_model = problem.encode_to_ising()
        
        # Solve
        annealer = GPUAnnealer(annealer_config)
        annealing_result = annealer.anneal(ising_model)
        
        # Decode solution
        solution = problem.decode_solution(annealing_result.best_configuration)
        
        # Compute metrics
        approx_ratio = problem.get_approximation_ratio(solution.objective_value)
        
        return {
            "problem_name": problem_name,
            "problem_size": problem.size,
            "instance_params": instance_params,
            "solution": solution,
            "annealing_result": annealing_result,
            "approximation_ratio": approx_ratio,
            "solution_quality": "optimal" if approx_ratio == 1.0 else "approximate",
            "runtime": annealing_result.total_time,
            "convergence_sweep": annealing_result.convergence_sweep
        }
    
    def run_benchmark_suite(
        self,
        problems: Optional[List[str]] = None,
        annealer_config: Optional[GPUAnnealerConfig] = None,
        n_runs: int = 1
    ) -> Dict[str, List[Dict]]:
        """Run benchmarks on multiple problems."""
        if problems is None:
            problems = self.list_problems()
        
        results = {}
        
        for problem_name in problems:
            print(f"Running benchmark: {problem_name}")
            problem_results = []
            
            for run in range(n_runs):
                seed = 42 + run  # Different seed for each run
                result = self.run_benchmark(problem_name, annealer_config, seed)
                problem_results.append(result)
            
            results[problem_name] = problem_results
        
        return results
    
    def analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        analysis = {}
        
        for problem_name, problem_results in results.items():
            objective_values = [r["solution"].objective_value for r in problem_results]
            runtimes = [r["runtime"] for r in problem_results]
            approx_ratios = [r["approximation_ratio"] for r in problem_results 
                           if np.isfinite(r["approximation_ratio"])]
            
            analysis[problem_name] = {
                "n_runs": len(problem_results),
                "best_objective": min(objective_values),
                "mean_objective": np.mean(objective_values),
                "std_objective": np.std(objective_values),
                "mean_runtime": np.mean(runtimes),
                "std_runtime": np.std(runtimes),
                "mean_approx_ratio": np.mean(approx_ratios) if approx_ratios else float('inf'),
                "success_rate": len([r for r in problem_results if r["solution"].is_feasible]) / len(problem_results)
            }
        
        return analysis


def demo_standard_benchmarks():
    """Demonstration of standard benchmark suite."""
    print("Spin-Glass-Anneal-RL Standard Benchmark Suite")
    print("=" * 50)
    
    suite = StandardBenchmarkSuite()
    
    # List available problems
    print(f"Available problems: {suite.list_problems()}")
    print()
    
    # Run small benchmark
    config = GPUAnnealerConfig(
        n_sweeps=1000,
        initial_temp=5.0,
        final_temp=0.1,
        random_seed=42
    )
    
    # Test MaxCut
    result = suite.run_benchmark("maxcut_20", config)
    print(f"MaxCut-20 Result:")
    print(f"  Objective: {result['solution'].objective_value:.2f}")
    print(f"  Runtime: {result['runtime']:.4f}s")
    print(f"  Feasible: {result['solution'].is_feasible}")
    print()
    
    # Test QAP
    result = suite.run_benchmark("qap_10", config)
    print(f"QAP-10 Result:")
    print(f"  Objective: {result['solution'].objective_value:.2f}")
    print(f"  Runtime: {result['runtime']:.4f}s")
    print(f"  Feasible: {result['solution'].is_feasible}")
    print()


if __name__ == "__main__":
    demo_standard_benchmarks()