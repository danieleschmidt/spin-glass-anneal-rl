#!/usr/bin/env python3
"""Interactive optimization CLI for real-time problem solving."""

import time
import argparse
from typing import Dict, Any, Optional
import torch
import numpy as np
from pathlib import Path
import json

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.scheduling import SchedulingProblem
from spin_glass_rl.problems.routing import TSPProblem


class InteractiveOptimizer:
    """Interactive command-line optimizer for immediate problem solving."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model: Optional[IsingModel] = None
        self.last_result = None
        
    def quick_solve(self, problem_type: str, **kwargs) -> Dict[str, Any]:
        """Solve common problems with sensible defaults."""
        print(f"🚀 Quick solving {problem_type}...")
        
        if problem_type == "tsp":
            return self._solve_tsp_quick(**kwargs)
        elif problem_type == "scheduling":
            return self._solve_scheduling_quick(**kwargs)
        elif problem_type == "random":
            return self._solve_random_quick(**kwargs)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def _solve_tsp_quick(self, n_cities: int = 10, **kwargs) -> Dict[str, Any]:
        """Quick TSP solve."""
        tsp = TSPProblem()
        tsp.generate_random_instance(n_locations=n_cities, area_size=100.0)
        
        ising_model = tsp.encode_to_ising()
        
        # Quick annealer config
        config = GPUAnnealerConfig(
            n_sweeps=min(2000, n_cities * 200),
            initial_temp=5.0 * np.sqrt(n_cities),
            final_temp=0.001,
            schedule_type=ScheduleType.GEOMETRIC,
            random_seed=int(time.time()) % 1000
        )
        
        annealer = GPUAnnealer(config)
        result = annealer.anneal(ising_model)
        
        # Decode solution
        try:
            solution = tsp.decode_solution(result.best_configuration)
            return {
                "problem_type": "tsp",
                "n_cities": n_cities,
                "tour_length": solution.objective_value,
                "solution_time": result.total_time,
                "energy": result.best_energy,
                "is_feasible": solution.is_feasible,
                "tour": solution.variables.get("tour", [])
            }
        except Exception as e:
            return {
                "problem_type": "tsp",
                "n_cities": n_cities,
                "error": str(e),
                "energy": result.best_energy,
                "solution_time": result.total_time
            }
    
    def _solve_scheduling_quick(self, n_tasks: int = 8, n_agents: int = 3, **kwargs) -> Dict[str, Any]:
        """Quick scheduling solve."""
        scheduler = SchedulingProblem()
        scheduler.generate_random_instance(
            n_tasks=n_tasks,
            n_agents=n_agents,
            time_horizon=50.0
        )
        
        ising_model = scheduler.encode_to_ising(objective="makespan")
        
        # Quick annealer config
        config = GPUAnnealerConfig(
            n_sweeps=min(3000, n_tasks * n_agents * 100),
            initial_temp=10.0,
            final_temp=0.01,
            schedule_type=ScheduleType.ADAPTIVE,
            random_seed=int(time.time()) % 1000
        )
        
        annealer = GPUAnnealer(config)
        result = annealer.anneal(ising_model)
        
        # Decode solution
        try:
            solution = scheduler.decode_solution(result.best_configuration)
            return {
                "problem_type": "scheduling",
                "n_tasks": n_tasks,
                "n_agents": n_agents,
                "makespan": solution.metadata.get("makespan", 0),
                "solution_time": result.total_time,
                "energy": result.best_energy,
                "is_feasible": solution.is_feasible,
                "n_assigned": solution.metadata.get("n_assigned_tasks", 0)
            }
        except Exception as e:
            return {
                "problem_type": "scheduling",
                "n_tasks": n_tasks,
                "n_agents": n_agents,
                "error": str(e),
                "energy": result.best_energy,
                "solution_time": result.total_time
            }
    
    def _solve_random_quick(self, n_spins: int = 100, **kwargs) -> Dict[str, Any]:
        """Quick random problem solve."""
        config = IsingModelConfig(
            n_spins=n_spins,
            use_sparse=True,
            device=str(self.device)
        )
        
        model = IsingModel(config)
        
        # Add random couplings
        n_couplings = min(n_spins * 2, n_spins * (n_spins - 1) // 4)
        for _ in range(n_couplings):
            i, j = np.random.randint(0, n_spins, 2)
            if i != j:
                strength = np.random.uniform(-1, 1)
                model.set_coupling(i, j, strength)
        
        # Add random fields
        fields = torch.randn(n_spins) * 0.5
        model.set_external_fields(fields)
        
        initial_energy = model.compute_energy()
        
        # Quick annealer config
        config = GPUAnnealerConfig(
            n_sweeps=min(5000, n_spins * 50),
            initial_temp=5.0,
            final_temp=0.01,
            schedule_type=ScheduleType.GEOMETRIC,
            random_seed=int(time.time()) % 1000
        )
        
        annealer = GPUAnnealer(config)
        result = annealer.anneal(model)
        
        return {
            "problem_type": "random",
            "n_spins": n_spins,
            "n_couplings": n_couplings,
            "initial_energy": initial_energy,
            "final_energy": result.best_energy,
            "improvement": initial_energy - result.best_energy,
            "solution_time": result.total_time,
            "sweeps": result.n_sweeps
        }
    
    def benchmark_scaling(self, problem_type: str = "random", max_size: int = 1000) -> Dict[str, Any]:
        """Quick scaling benchmark."""
        print(f"📊 Running scaling benchmark for {problem_type}...")
        
        sizes = [50, 100, 200, 500]
        if max_size > 500:
            sizes.append(max_size)
        
        results = {}
        
        for size in sizes:
            print(f"  Testing size {size}...")
            
            if problem_type == "random":
                result = self._solve_random_quick(n_spins=size)
            elif problem_type == "tsp":
                result = self._solve_tsp_quick(n_cities=size // 10)
            else:
                continue
            
            results[size] = {
                "time": result["solution_time"],
                "energy": result.get("final_energy", result.get("energy", 0)),
                "improvement": result.get("improvement", 0)
            }
            
            # Time limit to prevent runaway
            if result["solution_time"] > 30.0:
                print(f"    Stopping benchmark - time limit exceeded")
                break
        
        return {
            "problem_type": problem_type,
            "sizes": list(results.keys()),
            "results": results,
            "device": str(self.device)
        }


def main():
    """Interactive optimizer CLI."""
    parser = argparse.ArgumentParser(description="Interactive Spin-Glass Optimizer")
    parser.add_argument("problem", choices=["tsp", "scheduling", "random", "benchmark"], 
                        help="Problem type to solve")
    parser.add_argument("--size", type=int, default=50, help="Problem size")
    parser.add_argument("--n-cities", type=int, default=10, help="Number of cities for TSP")
    parser.add_argument("--n-tasks", type=int, default=8, help="Number of tasks for scheduling")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents for scheduling")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    optimizer = InteractiveOptimizer()
    
    print("🧠 Spin-Glass Interactive Optimizer")
    print(f"Device: {optimizer.device}")
    print("=" * 50)
    
    start_time = time.time()
    
    if args.problem == "benchmark":
        result = optimizer.benchmark_scaling("random", max_size=args.size)
    elif args.problem == "tsp":
        result = optimizer.quick_solve("tsp", n_cities=args.n_cities)
    elif args.problem == "scheduling":
        result = optimizer.quick_solve("scheduling", n_tasks=args.n_tasks, n_agents=args.n_agents)
    elif args.problem == "random":
        result = optimizer.quick_solve("random", n_spins=args.size)
    
    total_time = time.time() - start_time
    
    # Display results
    print("\n✅ Results:")
    print("-" * 30)
    
    if args.problem == "benchmark":
        print(f"Benchmark completed in {total_time:.2f}s")
        for size, data in result["results"].items():
            print(f"  Size {size}: {data['time']:.3f}s, Energy: {data['energy']:.4f}")
    else:
        for key, value in result.items():
            if key not in ["tour"]:  # Skip long outputs
                print(f"{key}: {value}")
        
        if "tour" in result and args.verbose:
            print(f"tour: {result['tour']}")
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_path}")
    
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()