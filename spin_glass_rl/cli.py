"""Command-line interface for Spin-Glass-Anneal-RL."""

import click
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.annealing.parallel_tempering import ParallelTempering, ParallelTemperingConfig
from spin_glass_rl.problems.scheduling import SchedulingProblem
from spin_glass_rl.problems.routing import TSPProblem, VRPProblem
from spin_glass_rl.problems.resource_allocation import ResourceAllocationProblem
from spin_glass_rl.problems.coordination import CoordinationProblem


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Spin-Glass-Anneal-RL: GPU-accelerated optimization via physics-inspired RL."""
    pass


@main.command()
@click.option('--problem', '-p', type=click.Choice(['tsp', 'vrp', 'scheduling', 'resource', 'coordination']), 
              required=True, help='Problem type to solve')
@click.option('--size', '-s', default=10, help='Problem size (number of cities, tasks, etc.)')
@click.option('--algorithm', '-a', type=click.Choice(['sa', 'pt']), default='sa', 
              help='Algorithm: sa=Simulated Annealing, pt=Parallel Tempering')
@click.option('--sweeps', default=1000, help='Number of annealing sweeps')
@click.option('--temp-max', default=10.0, help='Initial temperature')
@click.option('--temp-min', default=0.01, help='Final temperature')
@click.option('--schedule', type=click.Choice(['linear', 'exponential', 'geometric', 'adaptive']), 
              default='geometric', help='Temperature schedule')
@click.option('--output', '-o', help='Output file for results')
@click.option('--plot', is_flag=True, help='Generate solution plot')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def solve(problem, size, algorithm, sweeps, temp_max, temp_min, schedule, output, plot, seed, verbose):
    """Solve optimization problem using spin-glass annealing."""
    
    if verbose:
        click.echo(f"Setting up {problem} problem with size {size}")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create problem instance
    problem_instance = create_problem_instance(problem, size, verbose)
    
    # Configure annealer
    if algorithm == 'sa':
        annealer = configure_simulated_annealing(sweeps, temp_max, temp_min, schedule, seed, verbose)
        result = problem_instance.solve_with_annealer(annealer)
    elif algorithm == 'pt':
        annealer = configure_parallel_tempering(sweeps, temp_max, temp_min, seed, verbose)
        result = problem_instance.solve_with_annealer(annealer)
    
    # Display results
    display_results(result, verbose)
    
    # Save results
    if output:
        save_results(result, output, verbose)
    
    # Generate plot
    if plot:
        plot_path = output.replace('.json', '_plot.png') if output else f'{problem}_solution.png'
        problem_instance.plot_solution(result, plot_path)
        if verbose:
            click.echo(f"Solution plot saved to {plot_path}")


@main.command()
@click.option('--problem', '-p', type=click.Choice(['tsp', 'vrp', 'scheduling', 'resource', 'coordination']), 
              required=True, help='Problem type to benchmark')
@click.option('--sizes', default='10,20,50', help='Comma-separated problem sizes')
@click.option('--trials', default=3, help='Number of trials per size')
@click.option('--output', '-o', help='Output file for benchmark results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def benchmark(problem, sizes, trials, output, verbose):
    """Benchmark annealing performance on different problem sizes."""
    
    size_list = [int(s.strip()) for s in sizes.split(',')]
    
    if verbose:
        click.echo(f"Benchmarking {problem} problem on sizes: {size_list}")
    
    results = {}
    
    for size in size_list:
        if verbose:
            click.echo(f"Benchmarking size {size}...")
        
        # Create problem and annealer
        problem_instance = create_problem_instance(problem, size, verbose=False)
        annealer_config = GPUAnnealerConfig(n_sweeps=500, random_seed=42)
        annealer = GPUAnnealer(annealer_config)
        
        # Run benchmark
        size_results = problem_instance.benchmark_instance(
            {"size": size}, annealer, n_trials=trials
        )
        results[size] = size_results
        
        if verbose:
            click.echo(f"  Mean objective: {size_results['mean_objective']:.4f}")
            click.echo(f"  Mean time: {size_results['mean_time']:.4f}s")
    
    # Display summary
    click.echo("\nBenchmark Results:")
    click.echo("=" * 50)
    for size, result in results.items():
        click.echo(f"Size {size:3d}: obj={result['mean_objective']:8.4f}, "
                  f"time={result['mean_time']:6.4f}s, "
                  f"feasible={result['feasibility_rate']:5.1%}")
    
    # Save results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            click.echo(f"Benchmark results saved to {output}")


@main.command()
@click.option('--n-spins', default=100, help='Number of spins')
@click.option('--coupling-strength', default=1.0, help='Coupling strength')
@click.option('--field-strength', default=0.5, help='External field strength')
@click.option('--pattern', type=click.Choice(['random', 'nearest_neighbor', 'fully_connected']), 
              default='random', help='Coupling pattern')
@click.option('--sweeps', default=1000, help='Number of sweeps')
@click.option('--output', '-o', help='Output file for results')
@click.option('--plot', is_flag=True, help='Plot energy trajectory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def ising(n_spins, coupling_strength, field_strength, pattern, sweeps, output, plot, verbose):
    """Solve generic Ising model."""
    
    if verbose:
        click.echo(f"Creating Ising model: {n_spins} spins, {pattern} coupling")
    
    # Create Ising model
    config = IsingModelConfig(
        n_spins=n_spins,
        coupling_strength=coupling_strength,
        external_field_strength=field_strength,
        use_sparse=True
    )
    model = IsingModel(config)
    
    # Set coupling pattern
    from spin_glass_rl.core.coupling_matrix import CouplingMatrix
    coupling_matrix = CouplingMatrix(n_spins, use_sparse=True)
    
    if pattern == 'random':
        coupling_matrix.generate_pattern('random_graph', (-coupling_strength, coupling_strength), 
                                        edge_probability=0.1)
    elif pattern == 'nearest_neighbor':
        coupling_matrix.generate_pattern('nearest_neighbor', (-coupling_strength, coupling_strength))
    elif pattern == 'fully_connected':
        coupling_matrix.generate_pattern('fully_connected', (-coupling_strength, coupling_strength))
    
    model.set_couplings_from_matrix(coupling_matrix.matrix)
    
    # Set random external fields
    fields = torch.randn(n_spins) * field_strength
    model.set_external_fields(fields)
    
    if verbose:
        click.echo(f"Initial energy: {model.compute_energy():.6f}")
    
    # Run annealing
    annealer_config = GPUAnnealerConfig(n_sweeps=sweeps, random_seed=42)
    annealer = GPUAnnealer(annealer_config)
    result = annealer.anneal(model)
    
    # Display results
    click.echo(f"Final energy: {result.best_energy:.6f}")
    click.echo(f"Energy improvement: {model.compute_energy() - result.best_energy:.6f}")
    click.echo(f"Total time: {result.total_time:.4f}s")
    click.echo(f"Convergence: sweep {result.convergence_sweep}" if result.convergence_sweep else "No convergence detected")
    
    # Save results
    if output:
        result.save(output)
        if verbose:
            click.echo(f"Results saved to {output}")
    
    # Plot trajectory
    if plot:
        plot_path = output.replace('.npz', '_trajectory.png') if output else 'ising_trajectory.png'
        result.plot_trajectory(plot_path)
        if verbose:
            click.echo(f"Trajectory plot saved to {plot_path}")


@main.command()
@click.option('--input', '-i', required=True, help='Input problem file (JSON)')
@click.option('--output', '-o', help='Output solution file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def load(input, output, verbose):
    """Load and solve problem from file."""
    
    if verbose:
        click.echo(f"Loading problem from {input}")
    
    try:
        with open(input, 'r') as f:
            problem_data = json.load(f)
        
        problem_type = problem_data.get('type')
        if not problem_type:
            raise ValueError("Problem file must specify 'type' field")
        
        # Create problem instance based on type
        if problem_type == 'tsp':
            problem_instance = load_tsp_problem(problem_data)
        elif problem_type == 'scheduling':
            problem_instance = load_scheduling_problem(problem_data)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        
        # Solve problem
        annealer_config = GPUAnnealerConfig(
            n_sweeps=problem_data.get('sweeps', 1000),
            random_seed=problem_data.get('seed', 42)
        )
        annealer = GPUAnnealer(annealer_config)
        result = problem_instance.solve_with_annealer(annealer)
        
        # Display results
        display_results(result, verbose)
        
        # Save results
        if output:
            save_results(result, output, verbose)
            
    except Exception as e:
        click.echo(f"Error loading problem: {e}", err=True)
        return 1


def create_problem_instance(problem_type: str, size: int, verbose: bool = False):
    """Create problem instance of specified type and size."""
    
    if problem_type == 'tsp':
        problem = TSPProblem()
        problem.generate_random_instance(n_locations=size, area_size=100.0)
    
    elif problem_type == 'vrp':
        problem = VRPProblem()
        problem.generate_random_instance(n_locations=size, n_vehicles=max(1, size // 5), area_size=100.0)
    
    elif problem_type == 'scheduling':
        problem = SchedulingProblem()
        problem.generate_random_instance(n_tasks=size, n_agents=max(1, size // 3), time_horizon=100.0)
    
    elif problem_type == 'resource':
        problem = ResourceAllocationProblem()
        problem.generate_random_instance(n_demands=size, n_resources=max(1, size // 2))
    
    elif problem_type == 'coordination':
        problem = CoordinationProblem()
        problem.generate_random_instance(n_agents=max(2, size // 2), n_tasks=size, area_size=100.0)
    
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Encode as Ising model
    problem.encode_to_ising()
    
    if verbose:
        info = problem.get_problem_info()
        click.echo(f"Created {info['name']} with {info['ising_spins']} spins")
    
    return problem


def configure_simulated_annealing(sweeps: int, temp_max: float, temp_min: float, 
                                schedule: str, seed: Optional[int], verbose: bool = False):
    """Configure simulated annealing."""
    
    schedule_type = {
        'linear': ScheduleType.LINEAR,
        'exponential': ScheduleType.EXPONENTIAL,
        'geometric': ScheduleType.GEOMETRIC,
        'adaptive': ScheduleType.ADAPTIVE
    }[schedule]
    
    config = GPUAnnealerConfig(
        n_sweeps=sweeps,
        initial_temp=temp_max,
        final_temp=temp_min,
        schedule_type=schedule_type,
        random_seed=seed
    )
    
    annealer = GPUAnnealer(config)
    
    if verbose:
        click.echo(f"Configured simulated annealing: {sweeps} sweeps, {schedule} schedule")
    
    return annealer


def configure_parallel_tempering(sweeps: int, temp_max: float, temp_min: float, 
                                seed: Optional[int], verbose: bool = False):
    """Configure parallel tempering."""
    
    config = ParallelTemperingConfig(
        n_replicas=8,
        n_sweeps=sweeps,
        temp_min=temp_min,
        temp_max=temp_max,
        random_seed=seed
    )
    
    annealer = ParallelTempering(config)
    
    if verbose:
        click.echo(f"Configured parallel tempering: {config.n_replicas} replicas, {sweeps} sweeps")
    
    return annealer


def display_results(result, verbose: bool = False):
    """Display optimization results."""
    
    click.echo(f"Objective value: {result.objective_value:.6f}")
    click.echo(f"Feasible: {result.is_feasible}")
    click.echo(f"Total time: {result.metadata.get('total_time', 0):.4f}s")
    
    if verbose:
        click.echo(f"Best energy: {result.metadata.get('best_energy', 'N/A')}")
        click.echo(f"Convergence: sweep {result.metadata.get('convergence_sweep', 'N/A')}")
        
        if result.constraint_violations:
            click.echo("Constraint violations:")
            for constraint, violation in result.constraint_violations.items():
                if violation > 0:
                    click.echo(f"  {constraint}: {violation}")


def save_results(result, output_path: str, verbose: bool = False):
    """Save results to file."""
    
    # Prepare serializable data
    result_data = {
        "objective_value": result.objective_value,
        "is_feasible": result.is_feasible,
        "constraint_violations": result.constraint_violations,
        "metadata": result.metadata
    }
    
    # Save variables (simplified)
    if hasattr(result.variables, 'items'):
        result_data["variables"] = {}
        for key, value in result.variables.items():
            if isinstance(value, (list, dict, str, int, float)):
                result_data["variables"][key] = value
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    
    if verbose:
        click.echo(f"Results saved to {output_path}")


def load_tsp_problem(problem_data):
    """Load TSP problem from data."""
    problem = TSPProblem()
    
    # Add locations from data
    for loc_data in problem_data.get('locations', []):
        from spin_glass_rl.problems.routing import Location
        location = Location(
            id=loc_data['id'],
            name=loc_data.get('name', f"City_{loc_data['id']}"),
            x=loc_data['x'],
            y=loc_data['y']
        )
        problem.add_location(location)
    
    return problem


def load_scheduling_problem(problem_data):
    """Load scheduling problem from data."""
    problem = SchedulingProblem()
    
    # Add tasks
    for task_data in problem_data.get('tasks', []):
        from spin_glass_rl.problems.scheduling import Task
        task = Task(
            id=task_data['id'],
            duration=task_data['duration'],
            priority=task_data.get('priority', 1.0)
        )
        problem.add_task(task)
    
    # Add agents
    for agent_data in problem_data.get('agents', []):
        from spin_glass_rl.problems.scheduling import Agent
        agent = Agent(
            id=agent_data['id'],
            name=agent_data.get('name', f"Agent_{agent_data['id']}")
        )
        problem.add_agent(agent)
    
    return problem


if __name__ == '__main__':
    main()