"""Problem-specific benchmarks for optimization domains."""

import time
import numpy as np
import torch
from typing import Dict, Any, List

from .benchmark_runner import Benchmark, BenchmarkConfig, BenchmarkSuite
from spin_glass_rl.problems.routing import TSPProblem, VRPProblem
from spin_glass_rl.problems.scheduling import SchedulingProblem
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, AnnealerConfig


class TSPBenchmark(Benchmark):
    """Benchmark for Traveling Salesman Problem."""
    
    def __init__(self):
        super().__init__("TSP", "Traveling Salesman Problem benchmark")
        self.problem = None
        self.annealer = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup TSP problem and annealer."""
        self.problem = TSPProblem()
        
        annealer_config = AnnealerConfig(
            n_sweeps=1000,
            initial_temperature=2.0,
            final_temperature=0.01,
            device=config.device
        )
        self.annealer = GPUAnnealer(annealer_config)
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run single TSP trial."""
        # Generate random instance
        n_cities = 20 + trial_id % 10  # Vary between 20-30 cities
        instance_params = self.problem.generate_random_instance(
            n_locations=n_cities,
            area_size=100.0
        )
        
        # Solve
        start_time = time.time()
        solution = self.problem.solve_with_annealer(self.annealer)
        solve_time = time.time() - start_time
        
        return {
            'n_cities': n_cities,
            'objective_value': solution.objective_value,
            'is_feasible': solution.is_feasible,
            'solve_time': solve_time,
            'total_distance': solution.metadata.get('total_distance', solution.objective_value),
            'convergence_sweep': solution.metadata.get('annealing_result', {}).get('convergence_sweep', -1)
        }


class VRPBenchmark(Benchmark):
    """Benchmark for Vehicle Routing Problem."""
    
    def __init__(self):
        super().__init__("VRP", "Vehicle Routing Problem benchmark")
        self.problem = None
        self.annealer = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup VRP problem and annealer."""
        self.problem = VRPProblem()
        
        annealer_config = AnnealerConfig(
            n_sweeps=2000,
            initial_temperature=3.0,
            final_temperature=0.01,
            device=config.device
        )
        self.annealer = GPUAnnealer(annealer_config)
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run single VRP trial."""
        # Generate random instance
        n_locations = 15 + trial_id % 5
        n_vehicles = 2 + trial_id % 3
        instance_params = self.problem.generate_random_instance(
            n_locations=n_locations,
            n_vehicles=n_vehicles,
            area_size=100.0
        )
        
        # Solve
        start_time = time.time()
        solution = self.problem.solve_with_annealer(self.annealer)
        solve_time = time.time() - start_time
        
        return {
            'n_locations': n_locations,
            'n_vehicles': n_vehicles,
            'objective_value': solution.objective_value,
            'is_feasible': solution.is_feasible,
            'solve_time': solve_time,
            'n_routes': solution.metadata.get('n_routes', 0),
            'served_customers': solution.metadata.get('served_customers', 0),
            'unserved_customers': solution.metadata.get('unserved_customers', 0)
        }


class SchedulingBenchmark(Benchmark):
    """Benchmark for Multi-Agent Scheduling."""
    
    def __init__(self):
        super().__init__("Scheduling", "Multi-agent scheduling benchmark")
        self.problem = None
        self.annealer = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup scheduling problem and annealer."""
        self.problem = SchedulingProblem()
        
        annealer_config = AnnealerConfig(
            n_sweeps=1500,
            initial_temperature=2.5,
            final_temperature=0.01,
            device=config.device
        )
        self.annealer = GPUAnnealer(annealer_config)
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run single scheduling trial."""
        # Generate random instance
        n_tasks = 10 + trial_id % 8
        n_agents = 3 + trial_id % 2
        instance_params = self.problem.generate_random_instance(
            n_tasks=n_tasks,
            n_agents=n_agents,
            time_horizon=50.0
        )
        
        # Solve with different objectives
        objectives = ['makespan', 'total_time', 'weighted_completion']
        objective = objectives[trial_id % len(objectives)]
        
        self.problem.encode_to_ising(objective=objective)
        
        # Solve
        start_time = time.time()
        solution = self.problem.solve_with_annealer(self.annealer)
        solve_time = time.time() - start_time
        
        return {
            'n_tasks': n_tasks,
            'n_agents': n_agents,
            'objective': objective,
            'objective_value': solution.objective_value,
            'is_feasible': solution.is_feasible,
            'solve_time': solve_time,
            'makespan': solution.metadata.get('makespan', 0),
            'total_completion_time': solution.metadata.get('total_completion_time', 0),
            'n_assigned_tasks': solution.metadata.get('n_assigned_tasks', 0)
        }


def create_standard_suite() -> BenchmarkSuite:
    """Create standard problem benchmark suite."""
    suite = BenchmarkSuite("StandardProblems", "Standard optimization problems")
    
    suite.add_benchmark(TSPBenchmark())
    suite.add_benchmark(VRPBenchmark()) 
    suite.add_benchmark(SchedulingBenchmark())
    
    return suite


def create_problem_suite() -> BenchmarkSuite:
    """Create comprehensive problem-specific benchmark suite."""
    suite = BenchmarkSuite("ProblemSpecific", "Domain-specific problem benchmarks")
    
    # Add problem-specific benchmarks
    suite.add_benchmark(TSPBenchmark())
    suite.add_benchmark(VRPBenchmark())
    suite.add_benchmark(SchedulingBenchmark())
    
    # Add specialized variants
    suite.add_benchmark(TSPScalabilityBenchmark())
    suite.add_benchmark(SchedulingObjectiveBenchmark())
    
    return suite


class TSPScalabilityBenchmark(Benchmark):
    """TSP scalability benchmark with increasing problem sizes."""
    
    def __init__(self):
        super().__init__("TSP_Scalability", "TSP scalability across different city counts")
        self.problem = None
        self.annealer = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup TSP problem and annealer."""
        self.problem = TSPProblem()
        
        annealer_config = AnnealerConfig(
            n_sweeps=500,  # Reduced for scalability testing
            initial_temperature=2.0,
            final_temperature=0.01,
            device=config.device
        )
        self.annealer = GPUAnnealer(annealer_config)
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run TSP trial with varying sizes."""
        # Test different problem sizes
        sizes = [10, 15, 20, 25, 30, 40, 50]
        n_cities = sizes[trial_id % len(sizes)]
        
        instance_params = self.problem.generate_random_instance(
            n_locations=n_cities,
            area_size=100.0
        )
        
        # Solve
        start_time = time.time()
        solution = self.problem.solve_with_annealer(self.annealer)
        solve_time = time.time() - start_time
        
        # Calculate efficiency metrics
        distance_per_city = solution.objective_value / n_cities if n_cities > 0 else 0
        time_per_city = solve_time / n_cities if n_cities > 0 else 0
        
        return {
            'n_cities': n_cities,
            'objective_value': solution.objective_value,
            'is_feasible': solution.is_feasible,
            'solve_time': solve_time,
            'distance_per_city': distance_per_city,
            'time_per_city': time_per_city,
            'efficiency_ratio': distance_per_city / time_per_city if time_per_city > 0 else 0
        }


class SchedulingObjectiveBenchmark(Benchmark):
    """Scheduling benchmark comparing different objectives."""
    
    def __init__(self):
        super().__init__("Scheduling_Objectives", "Compare scheduling objectives")
        self.problem = None
        self.annealer = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup scheduling problem and annealer.""" 
        self.problem = SchedulingProblem()
        
        annealer_config = AnnealerConfig(
            n_sweeps=1000,
            initial_temperature=2.0,
            final_temperature=0.01,
            device=config.device
        )
        self.annealer = GPUAnnealer(annealer_config)
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run scheduling trial with different objectives."""
        # Fixed problem size for fair comparison
        n_tasks = 12
        n_agents = 3
        
        # Generate consistent instance
        np.random.seed(42 + trial_id // 3)  # Same instance for every 3 trials
        instance_params = self.problem.generate_random_instance(
            n_tasks=n_tasks,
            n_agents=n_agents, 
            time_horizon=60.0
        )
        
        # Cycle through objectives
        objectives = ['makespan', 'total_time', 'weighted_completion']
        objective = objectives[trial_id % len(objectives)]
        
        self.problem.encode_to_ising(objective=objective)
        
        # Solve
        start_time = time.time()
        solution = self.problem.solve_with_annealer(self.annealer)
        solve_time = time.time() - start_time
        
        return {
            'instance_id': trial_id // 3,
            'objective_type': objective,
            'objective_value': solution.objective_value,
            'is_feasible': solution.is_feasible,
            'solve_time': solve_time,
            'makespan': solution.metadata.get('makespan', 0),
            'total_completion_time': solution.metadata.get('total_completion_time', 0),
            'n_assigned_tasks': solution.metadata.get('n_assigned_tasks', 0)
        }