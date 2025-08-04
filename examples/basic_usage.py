#!/usr/bin/env python3
"""Basic usage examples for Spin-Glass-Anneal-RL."""

import numpy as np
import torch

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.coupling_matrix import CouplingMatrix
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.scheduling import SchedulingProblem
from spin_glass_rl.problems.routing import TSPProblem


def example_basic_ising_model():
    """Example: Basic Ising model creation and optimization."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Ising Model")
    print("=" * 60)
    
    # Create Ising model
    config = IsingModelConfig(
        n_spins=50,
        coupling_strength=1.0,
        external_field_strength=0.5,
        use_sparse=True,
        device="cpu"  # Use CPU for basic example
    )
    
    model = IsingModel(config)
    print(f"Created Ising model with {model.n_spins} spins")
    
    # Add random couplings
    coupling_matrix = CouplingMatrix(model.n_spins, use_sparse=True)
    coupling_matrix.generate_pattern(
        "random_graph", 
        strength_range=(-1.0, 1.0),
        edge_probability=0.1
    )
    
    model.set_couplings_from_matrix(coupling_matrix.matrix)
    print(f"Added {coupling_matrix.n_couplings} random couplings")
    
    # Set random external fields
    fields = torch.randn(model.n_spins) * 0.5
    model.set_external_fields(fields)
    
    # Initial energy
    initial_energy = model.compute_energy()
    print(f"Initial energy: {initial_energy:.6f}")
    
    # Configure annealer
    annealer_config = GPUAnnealerConfig(
        n_sweeps=2000,
        initial_temp=5.0,
        final_temp=0.01,
        schedule_type=ScheduleType.GEOMETRIC,
        random_seed=42
    )
    
    annealer = GPUAnnealer(annealer_config)
    
    # Run optimization
    print("Running simulated annealing...")
    result = annealer.anneal(model)
    
    # Results
    print(f"Final energy: {result.best_energy:.6f}")
    print(f"Energy improvement: {initial_energy - result.best_energy:.6f}")
    print(f"Optimization time: {result.total_time:.4f} seconds")
    print(f"Final acceptance rate: {result.final_acceptance_rate:.4f}")
    
    if result.convergence_sweep:
        print(f"Converged at sweep: {result.convergence_sweep}")
    
    print()


def example_tsp_problem():
    """Example: Traveling Salesman Problem."""
    print("=" * 60)
    print("EXAMPLE 2: Traveling Salesman Problem")
    print("=" * 60)
    
    # Create TSP problem
    tsp = TSPProblem()
    
    # Generate random instance
    instance_params = tsp.generate_random_instance(
        n_locations=12,
        area_size=100.0
    )
    print(f"Generated TSP instance: {instance_params['n_locations']} cities")
    
    # Display city locations
    print("City locations:")
    for i, location in enumerate(tsp.locations):
        print(f"  City {i}: ({location.x:.1f}, {location.y:.1f})")
    
    # Encode as Ising model
    ising_model = tsp.encode_to_ising(
        penalty_weights={
            "city_visit": 100.0,
            "position_fill": 100.0
        }
    )
    print(f"Encoded as Ising model with {ising_model.n_spins} spins")
    
    # Configure annealer
    annealer_config = GPUAnnealerConfig(
        n_sweeps=3000,
        initial_temp=10.0,
        final_temp=0.001,
        schedule_type=ScheduleType.GEOMETRIC,
        random_seed=42
    )
    
    annealer = GPUAnnealer(annealer_config)
    
    # Solve
    print("Solving TSP...")
    solution = tsp.solve_with_annealer(annealer)
    
    # Results
    print(f"Total distance: {solution.objective_value:.2f}")
    print(f"Solution feasible: {solution.is_feasible}")
    print(f"Optimization time: {solution.metadata['total_time']:.4f} seconds")
    
    if solution.is_feasible:
        tour = solution.variables["tour"]
        print(f"Optimal tour: {' -> '.join(map(str, tour))} -> {tour[0]}")
    
    # Constraint violations
    if solution.constraint_violations:
        print("Constraint violations:")
        for constraint, violation in solution.constraint_violations.items():
            if violation > 0:
                print(f"  {constraint}: {violation}")
    
    print()


def example_scheduling_problem():
    """Example: Multi-Agent Scheduling Problem."""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Agent Scheduling")
    print("=" * 60)
    
    # Create scheduling problem
    scheduler = SchedulingProblem()
    
    # Generate random instance
    instance_params = scheduler.generate_random_instance(
        n_tasks=8,
        n_agents=3,
        time_horizon=50.0
    )
    print(f"Generated scheduling instance:")
    print(f"  Tasks: {instance_params['n_tasks']}")
    print(f"  Agents: {instance_params['n_agents']}")
    print(f"  Time horizon: {instance_params['time_horizon']}")
    
    # Display tasks
    print("\nTasks:")
    for task in scheduler.tasks:
        print(f"  Task {task.id}: duration={task.duration:.1f}, priority={task.priority:.2f}")
        if task.due_date:
            print(f"    Due date: {task.due_date:.1f}")
    
    # Display agents
    print("\nAgents:")
    for agent in scheduler.agents:
        print(f"  {agent.name}: cost=${agent.cost_per_hour:.1f}/hour")
    
    # Encode as Ising model
    ising_model = scheduler.encode_to_ising(
        objective="makespan",
        penalty_weights={
            "assignment": 150.0,
            "capacity": 100.0,
            "time_window": 75.0
        }
    )
    print(f"\nEncoded as Ising model with {ising_model.n_spins} spins")
    
    # Configure annealer
    annealer_config = GPUAnnealerConfig(
        n_sweeps=2500,
        initial_temp=8.0,
        final_temp=0.01,
        schedule_type=ScheduleType.ADAPTIVE,
        random_seed=42
    )
    
    annealer = GPUAnnealer(annealer_config)
    
    # Solve
    print("Solving scheduling problem...")
    solution = scheduler.solve_with_annealer(annealer)
    
    # Results
    print(f"Makespan: {solution.metadata['makespan']:.2f}")
    print(f"Total completion time: {solution.metadata['total_completion_time']:.2f}")
    print(f"Assigned tasks: {solution.metadata['n_assigned_tasks']}/{len(scheduler.tasks)}")
    print(f"Solution feasible: {solution.is_feasible}")
    print(f"Optimization time: {solution.metadata['total_time']:.4f} seconds")
    
    # Display schedule
    if solution.is_feasible:
        schedule = solution.variables["schedule"]
        print("\nSchedule:")
        for agent_id, tasks in schedule.items():
            print(f"  Agent {agent_id}:")
            for task_info in sorted(tasks, key=lambda x: x["start_time"]):
                print(f"    Task {task_info['task_id']}: "
                      f"{task_info['start_time']:.1f} - {task_info['end_time']:.1f}")
    
    print()


def example_temperature_schedules():
    """Example: Comparing different temperature schedules."""
    print("=" * 60)
    print("EXAMPLE 4: Temperature Schedule Comparison")
    print("=" * 60)
    
    from spin_glass_rl.annealing.temperature_scheduler import TemperatureScheduler
    
    # Create small Ising model for comparison
    config = IsingModelConfig(n_spins=20, use_sparse=True)
    model = IsingModel(config)
    
    # Add some couplings
    for i in range(model.n_spins - 1):
        model.set_coupling(i, i + 1, -1.0)  # Ferromagnetic chain
    
    schedules = [ScheduleType.LINEAR, ScheduleType.EXPONENTIAL, 
                ScheduleType.GEOMETRIC, ScheduleType.ADAPTIVE]
    
    results = {}
    
    for schedule_type in schedules:
        print(f"Testing {schedule_type.value} schedule...")
        
        annealer_config = GPUAnnealerConfig(
            n_sweeps=1000,
            initial_temp=5.0,
            final_temp=0.01,
            schedule_type=schedule_type,
            random_seed=42
        )
        
        annealer = GPUAnnealer(annealer_config)
        result = annealer.anneal(model.copy())  # Use copy to ensure fair comparison
        
        results[schedule_type.value] = {
            "final_energy": result.best_energy,
            "time": result.total_time,
            "convergence": result.convergence_sweep
        }
    
    # Compare results
    print("\nSchedule Comparison:")
    print("-" * 70)
    print(f"{'Schedule':<12} {'Final Energy':<15} {'Time (s)':<10} {'Convergence':<12}")
    print("-" * 70)
    
    for schedule, result in results.items():
        convergence = result["convergence"] if result["convergence"] else "No"
        print(f"{schedule:<12} {result['final_energy']:<15.6f} "
              f"{result['time']:<10.4f} {convergence:<12}")
    
    print()


def example_parallel_tempering():
    """Example: Parallel Tempering."""
    print("=" * 60)
    print("EXAMPLE 5: Parallel Tempering")
    print("=" * 60)
    
    from spin_glass_rl.annealing.parallel_tempering import ParallelTempering, ParallelTemperingConfig
    
    # Create frustrated Ising model (harder to optimize)
    config = IsingModelConfig(n_spins=30, use_sparse=True)
    model = IsingModel(config)
    
    # Add random frustrated couplings
    np.random.seed(42)
    for _ in range(50):
        i, j = np.random.randint(0, model.n_spins, 2)
        if i != j:
            strength = np.random.choice([-1.0, 1.0])  # Random +/- coupling
            model.set_coupling(i, j, strength)
    
    print(f"Created frustrated Ising model with {model.n_spins} spins")
    initial_energy = model.compute_energy()
    print(f"Initial energy: {initial_energy:.6f}")
    
    # Configure parallel tempering
    pt_config = ParallelTemperingConfig(
        n_replicas=6,
        n_sweeps=2000,
        temp_min=0.1,
        temp_max=5.0,
        temp_distribution="geometric",
        exchange_interval=10,
        random_seed=42
    )
    
    pt = ParallelTempering(pt_config)
    
    # Run parallel tempering
    print(f"Running parallel tempering with {pt_config.n_replicas} replicas...")
    result = pt.run(model)
    
    # Results
    print(f"Final energy: {result.best_energy:.6f}")
    print(f"Energy improvement: {initial_energy - result.best_energy:.6f}")
    print(f"Optimization time: {result.total_time:.4f} seconds")
    
    # Exchange statistics
    exchange_rates = pt.get_exchange_rates()
    if len(exchange_rates) > 0:
        print(f"Mean exchange rate: {np.mean(exchange_rates):.4f}")
        print("Exchange rates between replicas:")
        for i, rate in enumerate(exchange_rates):
            print(f"  Replicas {i}-{i+1}: {rate:.4f}")
    
    print()


if __name__ == "__main__":
    print("Spin-Glass-Anneal-RL Examples")
    print("=============================")
    print()
    
    # Run examples
    example_basic_ising_model()
    example_tsp_problem()
    example_scheduling_problem()
    example_temperature_schedules()
    example_parallel_tempering()
    
    print("All examples completed successfully!")
    print("Try running with different parameters or problem sizes.")
    print("Use 'spin-glass-rl --help' for CLI options.")