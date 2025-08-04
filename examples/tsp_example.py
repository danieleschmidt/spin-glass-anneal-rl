#!/usr/bin/env python3
"""Detailed TSP example with visualization and analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from spin_glass_rl.problems.routing import TSPProblem, Location
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType


def create_tsp_instances() -> List[TSPProblem]:
    """Create different TSP instances for comparison."""
    
    instances = []
    
    # 1. Small random instance
    tsp1 = TSPProblem()
    tsp1.generate_random_instance(n_locations=8, area_size=100.0)
    tsp1.name = "Random 8-City TSP"
    instances.append(tsp1)
    
    # 2. Clustered cities
    tsp2 = TSPProblem()
    # Create two clusters
    cluster1_center = (20, 20)
    cluster2_center = (80, 80)
    
    locations = []
    for i in range(6):
        if i < 3:
            # Cluster 1
            x = cluster1_center[0] + np.random.normal(0, 8)
            y = cluster1_center[1] + np.random.normal(0, 8)
        else:
            # Cluster 2
            x = cluster2_center[0] + np.random.normal(0, 8)
            y = cluster2_center[1] + np.random.normal(0, 8)
        
        location = Location(id=i, name=f"City_{i}", x=x, y=y)
        tsp2.add_location(location)
    
    tsp2.name = "Clustered 6-City TSP"
    instances.append(tsp2)
    
    # 3. Grid-based cities
    tsp3 = TSPProblem()
    grid_size = 3
    spacing = 30
    
    loc_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * spacing + 10
            y = j * spacing + 10
            location = Location(id=loc_id, name=f"Grid_{i}_{j}", x=x, y=y)
            tsp3.add_location(location)
            loc_id += 1
    
    tsp3.name = "Grid 9-City TSP"
    instances.append(tsp3)
    
    return instances


def solve_with_different_methods(tsp: TSPProblem) -> dict:
    """Solve TSP with different annealing configurations."""
    
    results = {}
    
    # Method 1: Fast cooling
    config1 = GPUAnnealerConfig(
        n_sweeps=1000,
        initial_temp=20.0,
        final_temp=0.001,
        schedule_type=ScheduleType.EXPONENTIAL,
        random_seed=42
    )
    annealer1 = GPUAnnealer(config1)
    tsp.encode_to_ising()
    solution1 = tsp.solve_with_annealer(annealer1)
    results["Fast Cooling"] = solution1
    
    # Method 2: Slow cooling
    config2 = GPUAnnealerConfig(
        n_sweeps=3000,
        initial_temp=10.0,
        final_temp=0.01,
        schedule_type=ScheduleType.GEOMETRIC,
        schedule_params={"alpha": 0.99},
        random_seed=42
    )
    annealer2 = GPUAnnealer(config2)
    tsp.encode_to_ising()
    solution2 = tsp.solve_with_annealer(annealer2)
    results["Slow Cooling"] = solution2
    
    # Method 3: Adaptive cooling
    config3 = GPUAnnealerConfig(
        n_sweeps=2000,
        initial_temp=15.0,
        final_temp=0.01,
        schedule_type=ScheduleType.ADAPTIVE,
        random_seed=42
    )
    annealer3 = GPUAnnealer(config3)
    tsp.encode_to_ising()
    solution3 = tsp.solve_with_annealer(annealer3)
    results["Adaptive Cooling"] = solution3
    
    return results


def analyze_solution_quality(tsp: TSPProblem, solution) -> dict:
    """Analyze TSP solution quality."""
    
    if not solution.is_feasible:
        return {"feasible": False, "analysis": "Solution not feasible"}
    
    tour = solution.variables["tour"]
    n_cities = len(tsp.locations)
    
    analysis = {
        "feasible": True,
        "tour_length": len(tour),
        "total_distance": solution.objective_value,
        "avg_distance_per_edge": solution.objective_value / n_cities,
    }
    
    # Calculate tour properties
    distances = []
    for i in range(n_cities):
        city1 = tour[i]
        city2 = tour[(i + 1) % n_cities]
        distance = tsp.get_distance(city1, city2)
        distances.append(distance)
    
    analysis.update({
        "min_edge_distance": min(distances),
        "max_edge_distance": max(distances),
        "std_edge_distance": np.std(distances),
    })
    
    # Check for obvious improvements (crossing edges)
    crossings = count_edge_crossings(tsp, tour)
    analysis["edge_crossings"] = crossings
    
    return analysis


def count_edge_crossings(tsp: TSPProblem, tour: List[int]) -> int:
    """Count crossing edges in TSP tour."""
    
    n_cities = len(tour)
    crossings = 0
    
    for i in range(n_cities):
        for j in range(i + 2, n_cities):
            if j == (i - 1) % n_cities:  # Adjacent edges don't count
                continue
            
            # Check if edge (tour[i], tour[i+1]) crosses edge (tour[j], tour[j+1])
            p1 = tsp.locations[tour[i]]
            p2 = tsp.locations[tour[(i + 1) % n_cities]]
            p3 = tsp.locations[tour[j]]
            p4 = tsp.locations[tour[(j + 1) % n_cities]]
            
            if edges_intersect((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)):
                crossings += 1
    
    return crossings // 2  # Each crossing counted twice


def edges_intersect(p1: Tuple[float, float], p2: Tuple[float, float], 
                   p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Check if two line segments intersect."""
    
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def plot_detailed_comparison(tsp: TSPProblem, results: dict, save_path: str = None):
    """Plot detailed comparison of TSP solutions."""
    
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    # Plot solutions
    for idx, (method, solution) in enumerate(results.items()):
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]
        
        # Plot cities
        x_coords = [loc.x for loc in tsp.locations]
        y_coords = [loc.y for loc in tsp.locations]
        
        ax1.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
        
        # Add city labels
        for i, loc in enumerate(tsp.locations):
            ax1.annotate(f'{i}', (loc.x, loc.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        # Plot tour if feasible
        if solution.is_feasible:
            tour = solution.variables["tour"]
            tour_x = [tsp.locations[city].x for city in tour] + [tsp.locations[tour[0]].x]
            tour_y = [tsp.locations[city].y for city in tour] + [tsp.locations[tour[0]].y]
            
            ax1.plot(tour_x, tour_y, 'b-', linewidth=2, alpha=0.7)
            ax1.set_title(f'{method}\nDistance: {solution.objective_value:.2f}')
        else:
            ax1.set_title(f'{method}\n(Infeasible)')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot energy trajectory
        if hasattr(solution.metadata.get('annealing_result'), 'energy_history'):
            annealing_result = solution.metadata['annealing_result']
            ax2.plot(annealing_result.energy_history)
            ax2.set_xlabel('Sweep')
            ax2.set_ylabel('Energy')
            ax2.set_title(f'Energy Trajectory\nFinal: {annealing_result.best_energy:.4f}')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title('Energy Trajectory')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main TSP example with detailed analysis."""
    
    print("Detailed TSP Example")
    print("===================")
    print()
    
    # Create TSP instances
    instances = create_tsp_instances()
    
    for instance in instances:
        print(f"Solving {instance.name}...")
        print("-" * 50)
        
        # Solve with different methods
        results = solve_with_different_methods(instance)
        
        # Compare results
        print("Method Comparison:")
        print(f"{'Method':<15} {'Distance':<12} {'Time (s)':<10} {'Feasible':<10}")
        print("-" * 50)
        
        best_distance = float('inf')
        best_method = None
        
        for method, solution in results.items():
            distance = solution.objective_value if solution.is_feasible else float('inf')
            time = solution.metadata.get('total_time', 0)
            feasible = "Yes" if solution.is_feasible else "No"
            
            print(f"{method:<15} {distance:<12.2f} {time:<10.4f} {feasible:<10}")
            
            if solution.is_feasible and distance < best_distance:
                best_distance = distance
                best_method = method
        
        print(f"\nBest method: {best_method} (distance: {best_distance:.2f})")
        
        # Analyze best solution
        if best_method:
            best_solution = results[best_method]
            analysis = analyze_solution_quality(instance, best_solution)
            
            print("\nSolution Analysis:")
            if analysis["feasible"]:
                print(f"  Average edge length: {analysis['avg_distance_per_edge']:.2f}")
                print(f"  Shortest edge: {analysis['min_edge_distance']:.2f}")
                print(f"  Longest edge: {analysis['max_edge_distance']:.2f}")
                print(f"  Edge length std: {analysis['std_edge_distance']:.2f}")
                print(f"  Edge crossings: {analysis['edge_crossings']}")
                
                tour = best_solution.variables["tour"]
                print(f"  Optimal tour: {' → '.join(map(str, tour))} → {tour[0]}")
        
        # Generate visualization
        plot_path = f"tsp_{instance.name.lower().replace(' ', '_').replace('-', '_')}_comparison.png"
        plot_detailed_comparison(instance, results, plot_path)
        
        print()
    
    print("TSP examples completed!")
    print("Check the generated plots for visual comparison of methods.")


if __name__ == "__main__":
    main()