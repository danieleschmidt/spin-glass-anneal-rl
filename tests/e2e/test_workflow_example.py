"""Example end-to-end workflow tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tests import E2E_TEST


@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_scheduling_problem_workflow(self, large_scheduling_problem, temp_dir):
        """Test complete scheduling problem solving workflow."""
        problem = large_scheduling_problem
        
        # Step 1: Problem setup and validation
        n_agents = problem["n_agents"]
        n_tasks = problem["n_tasks"]
        n_resources = problem["n_resources"]
        
        assert n_agents > 0
        assert n_tasks > 0
        assert n_resources > 0
        
        # Step 2: Problem encoding to Ising model
        # In a real implementation, this would be more sophisticated
        n_spins = n_agents * n_tasks
        
        # Mock coupling matrix representing scheduling constraints
        coupling_matrix = np.zeros((n_spins, n_spins))
        
        # Add coupling for constraint violations
        for task in range(n_tasks):
            for agent1 in range(n_agents):
                for agent2 in range(agent1 + 1, n_agents):
                    idx1 = agent1 * n_tasks + task
                    idx2 = agent2 * n_tasks + task
                    # Penalize multiple agents on same task
                    coupling_matrix[idx1, idx2] = 10.0
                    coupling_matrix[idx2, idx1] = 10.0
        
        # External field for preferences
        external_field = np.random.randn(n_spins) * 0.1
        
        # Step 3: Initialize annealing
        spins = np.random.choice([-1, 1], size=n_spins)
        
        # Step 4: Multi-stage annealing
        stages = [
            {"temperature": 10.0, "sweeps": 50},
            {"temperature": 1.0, "sweeps": 100},
            {"temperature": 0.1, "sweeps": 50},
        ]
        
        energy_history = []
        
        for stage in stages:
            temperature = stage["temperature"]
            n_sweeps = stage["sweeps"]
            
            for sweep in range(n_sweeps):
                for i in range(n_spins):
                    local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                    energy_change = 2 * spins[i] * local_field
                    
                    if (energy_change < 0 or 
                        np.random.random() < np.exp(-energy_change / temperature)):
                        spins[i] *= -1
                
                # Record energy every 10 sweeps
                if sweep % 10 == 0:
                    energy = (
                        -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                        - np.sum(external_field * spins)
                    )
                    energy_history.append(energy)
        
        # Step 5: Solution extraction
        assignment_matrix = spins.reshape(n_agents, n_tasks)
        assignment_matrix = (assignment_matrix + 1) // 2  # Convert to 0/1
        
        # Step 6: Solution validation
        assert assignment_matrix.shape == (n_agents, n_tasks)
        assert np.all((assignment_matrix == 0) | (assignment_matrix == 1))
        
        # Check basic constraints
        task_assignments = np.sum(assignment_matrix, axis=0)
        # Allow for some constraint violations in this mock test
        over_assigned_tasks = np.sum(task_assignments > 1)
        under_assigned_tasks = np.sum(task_assignments == 0)
        
        # Step 7: Save results
        results = {
            "problem_size": {"agents": n_agents, "tasks": n_tasks, "resources": n_resources},
            "solution_quality": {
                "final_energy": energy_history[-1] if energy_history else 0,
                "energy_improvement": energy_history[0] - energy_history[-1] if len(energy_history) > 1 else 0,
                "over_assigned_tasks": int(over_assigned_tasks),
                "under_assigned_tasks": int(under_assigned_tasks),
            },
            "assignment_matrix": assignment_matrix.tolist(),
            "energy_history": energy_history,
        }
        
        results_file = temp_dir / "scheduling_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Verify results file
        assert results_file.exists()
        assert results_file.stat().st_size > 0
        
        # Verify results content
        with open(results_file, "r") as f:
            loaded_results = json.load(f)
        
        assert loaded_results["problem_size"]["agents"] == n_agents
        assert loaded_results["problem_size"]["tasks"] == n_tasks
        assert "solution_quality" in loaded_results
        assert len(loaded_results["energy_history"]) > 0
    
    def test_benchmark_comparison_workflow(self, temp_dir):
        """Test benchmark comparison workflow."""
        # Step 1: Setup multiple problem instances
        problem_sizes = [10, 20, 30]
        algorithms = ["random", "greedy", "annealing"]
        
        results = {}
        
        for size in problem_sizes:
            results[f"size_{size}"] = {}
            
            # Create problem instance
            n_spins = size
            coupling_matrix = np.random.randn(n_spins, n_spins) * 0.1
            coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
            np.fill_diagonal(coupling_matrix, 0)
            external_field = np.random.randn(n_spins) * 0.5
            
            for algorithm in algorithms:
                # Step 2: Run different algorithms
                if algorithm == "random":
                    spins = np.random.choice([-1, 1], size=n_spins)
                    energy = (
                        -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                        - np.sum(external_field * spins)
                    )
                    runtime = 0.001  # Mock runtime
                
                elif algorithm == "greedy":
                    spins = np.ones(n_spins)  # Start all up
                    
                    # Greedy improvement
                    for i in range(n_spins):
                        # Try flipping each spin
                        local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                        energy_change = 2 * spins[i] * local_field
                        
                        if energy_change < 0:
                            spins[i] *= -1
                    
                    energy = (
                        -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                        - np.sum(external_field * spins)
                    )
                    runtime = 0.01  # Mock runtime
                
                elif algorithm == "annealing":
                    spins = np.random.choice([-1, 1], size=n_spins)
                    temperature = 1.0
                    
                    # Short annealing run
                    for _ in range(50):
                        for i in range(n_spins):
                            local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                            energy_change = 2 * spins[i] * local_field
                            
                            if (energy_change < 0 or 
                                np.random.random() < np.exp(-energy_change / temperature)):
                                spins[i] *= -1
                        
                        temperature *= 0.95
                    
                    energy = (
                        -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                        - np.sum(external_field * spins)
                    )
                    runtime = 0.1  # Mock runtime
                
                # Step 3: Store results
                results[f"size_{size}"][algorithm] = {
                    "energy": float(energy),
                    "runtime": runtime,
                    "configuration": spins.tolist(),
                }
        
        # Step 4: Generate comparison report
        report = {"benchmark_results": results, "summary": {}}
        
        # Calculate summary statistics
        for size in problem_sizes:
            size_key = f"size_{size}"
            energies = [results[size_key][alg]["energy"] for alg in algorithms]
            runtimes = [results[size_key][alg]["runtime"] for alg in algorithms]
            
            best_energy = min(energies)
            best_algorithm = algorithms[energies.index(best_energy)]
            
            report["summary"][size_key] = {
                "best_energy": best_energy,
                "best_algorithm": best_algorithm,
                "total_runtime": sum(runtimes),
            }
        
        # Step 5: Save report
        report_file = temp_dir / "benchmark_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Step 6: Verify report
        assert report_file.exists()
        
        with open(report_file, "r") as f:
            loaded_report = json.load(f)
        
        assert "benchmark_results" in loaded_report
        assert "summary" in loaded_report
        assert len(loaded_report["benchmark_results"]) == len(problem_sizes)
        
        # Verify all algorithms were tested
        for size in problem_sizes:
            size_key = f"size_{size}"
            assert size_key in loaded_report["benchmark_results"]
            for algorithm in algorithms:
                assert algorithm in loaded_report["benchmark_results"][size_key]
                assert "energy" in loaded_report["benchmark_results"][size_key][algorithm]
                assert "runtime" in loaded_report["benchmark_results"][size_key][algorithm]


@pytest.mark.e2e
class TestDataPersistence:
    """Test data persistence and loading workflows."""
    
    def test_experiment_data_lifecycle(self, temp_dir):
        """Test complete experiment data lifecycle."""
        # Step 1: Generate experiment data
        experiment_id = "test_exp_001"
        
        # Create experiment directory
        exp_dir = temp_dir / experiment_id
        exp_dir.mkdir()
        
        # Generate problem data
        problem_data = {
            "n_spins": 50,
            "coupling_matrix": np.random.randn(50, 50).tolist(),
            "external_field": np.random.randn(50).tolist(),
            "metadata": {
                "created_at": "2025-01-01T00:00:00Z",
                "problem_type": "random_ising",
                "difficulty": "medium",
            },
        }
        
        problem_file = exp_dir / "problem.json"
        with open(problem_file, "w") as f:
            json.dump(problem_data, f, indent=2)
        
        # Step 2: Run experiment and save results
        n_spins = problem_data["n_spins"]
        coupling_matrix = np.array(problem_data["coupling_matrix"])
        external_field = np.array(problem_data["external_field"])
        
        # Multiple runs
        n_runs = 5
        run_results = []
        
        for run in range(n_runs):
            # Initialize
            spins = np.random.choice([-1, 1], size=n_spins)
            
            # Short annealing
            temperature = 2.0
            energy_trajectory = []
            
            for step in range(100):
                # Annealing step
                for i in range(n_spins):
                    local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                    energy_change = 2 * spins[i] * local_field
                    
                    if (energy_change < 0 or 
                        np.random.random() < np.exp(-energy_change / temperature)):
                        spins[i] *= -1
                
                # Record energy
                if step % 10 == 0:
                    energy = (
                        -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                        - np.sum(external_field * spins)
                    )
                    energy_trajectory.append(energy)
                
                temperature *= 0.98
            
            # Save run results
            run_result = {
                "run_id": run,
                "final_configuration": spins.tolist(),
                "final_energy": energy_trajectory[-1] if energy_trajectory else 0,
                "energy_trajectory": energy_trajectory,
                "converged": len(energy_trajectory) > 5 and abs(energy_trajectory[-1] - energy_trajectory[-2]) < 0.01,
            }
            
            run_results.append(run_result)
            
            # Save individual run
            run_file = exp_dir / f"run_{run:03d}.json"
            with open(run_file, "w") as f:
                json.dump(run_result, f, indent=2)
        
        # Step 3: Aggregate results
        final_energies = [run["final_energy"] for run in run_results]
        
        aggregate_results = {
            "experiment_id": experiment_id,
            "n_runs": n_runs,
            "statistics": {
                "best_energy": float(np.min(final_energies)),
                "worst_energy": float(np.max(final_energies)),
                "mean_energy": float(np.mean(final_energies)),
                "std_energy": float(np.std(final_energies)),
            },
            "success_rate": sum(1 for run in run_results if run["converged"]) / n_runs,
            "run_files": [f"run_{run:03d}.json" for run in range(n_runs)],
        }
        
        aggregate_file = exp_dir / "aggregate_results.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate_results, f, indent=2)
        
        # Step 4: Verify data integrity
        assert problem_file.exists()
        assert aggregate_file.exists()
        
        # Check all run files exist
        for run in range(n_runs):
            run_file = exp_dir / f"run_{run:03d}.json"
            assert run_file.exists()
        
        # Step 5: Test data loading
        with open(aggregate_file, "r") as f:
            loaded_aggregate = json.load(f)
        
        assert loaded_aggregate["experiment_id"] == experiment_id
        assert loaded_aggregate["n_runs"] == n_runs
        assert "statistics" in loaded_aggregate
        assert 0 <= loaded_aggregate["success_rate"] <= 1
        
        # Load and verify individual runs
        for run in range(n_runs):
            run_file = exp_dir / f"run_{run:03d}.json"
            with open(run_file, "r") as f:
                loaded_run = json.load(f)
            
            assert loaded_run["run_id"] == run
            assert "final_configuration" in loaded_run
            assert "final_energy" in loaded_run
            assert len(loaded_run["final_configuration"]) == n_spins
        
        # Step 6: Cleanup verification
        total_files = len(list(exp_dir.glob("*.json")))
        expected_files = 1 + n_runs + 1  # problem + runs + aggregate
        assert total_files == expected_files


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceWorkflow:
    """Test performance-related end-to-end workflows."""
    
    def test_scaling_analysis_workflow(self, temp_dir):
        """Test complete scaling analysis workflow."""
        # Step 1: Define scaling experiment
        problem_sizes = [20, 40, 60, 80, 100]
        metrics = ["runtime", "energy", "convergence"]
        
        scaling_results = {}
        
        # Step 2: Run scaling experiments
        for size in problem_sizes:
            print(f"Testing size {size}...")
            
            # Create problem
            coupling_matrix = np.random.randn(size, size) * 0.1
            coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
            np.fill_diagonal(coupling_matrix, 0)
            external_field = np.random.randn(size) * 0.5
            
            # Initialize
            spins = np.random.choice([-1, 1], size=size)
            
            # Time the annealing
            import time
            start_time = time.perf_counter()
            
            temperature = 1.0
            n_sweeps = min(100, 2000 // size)  # Adaptive sweep count
            
            initial_energy = (
                -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                - np.sum(external_field * spins)
            )
            
            for sweep in range(n_sweeps):
                for i in range(size):
                    local_field = np.sum(coupling_matrix[i] * spins) + external_field[i]
                    energy_change = 2 * spins[i] * local_field
                    
                    if (energy_change < 0 or 
                        np.random.random() < np.exp(-energy_change / temperature)):
                        spins[i] *= -1
                
                temperature *= 0.99
            
            end_time = time.perf_counter()
            runtime = end_time - start_time
            
            final_energy = (
                -0.5 * np.sum(coupling_matrix * np.outer(spins, spins))
                - np.sum(external_field * spins)
            )
            
            # Step 3: Calculate metrics
            energy_improvement = initial_energy - final_energy
            convergence_metric = energy_improvement / abs(initial_energy) if initial_energy != 0 else 0
            
            scaling_results[size] = {
                "runtime": runtime,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_improvement": energy_improvement,
                "convergence_metric": convergence_metric,
                "sweeps_per_second": n_sweeps / runtime if runtime > 0 else 0,
                "spins_per_second": size * n_sweeps / runtime if runtime > 0 else 0,
            }
        
        # Step 4: Analyze scaling behavior
        sizes = list(scaling_results.keys())
        runtimes = [scaling_results[s]["runtime"] for s in sizes]
        
        # Simple linear regression for runtime scaling
        log_sizes = np.log(sizes)
        log_runtimes = np.log(runtimes)
        
        # Fit log(runtime) = a * log(size) + b
        A = np.vstack([log_sizes, np.ones(len(log_sizes))]).T
        scaling_exponent, log_intercept = np.linalg.lstsq(A, log_runtimes, rcond=None)[0]
        
        analysis = {
            "scaling_exponent": float(scaling_exponent),
            "theoretical_complexity": "O(n^2)" if 1.8 <= scaling_exponent <= 2.2 else "Other",
            "max_size_tested": max(sizes),
            "min_runtime": min(runtimes),
            "max_runtime": max(runtimes),
            "runtime_scaling_factor": max(runtimes) / min(runtimes),
        }
        
        # Step 5: Save scaling analysis
        full_report = {
            "scaling_results": scaling_results,
            "analysis": analysis,
            "metadata": {
                "problem_type": "random_ising",
                "algorithm": "simulated_annealing",
                "test_date": "2025-01-01",
            },
        }
        
        report_file = temp_dir / "scaling_analysis.json"
        with open(report_file, "w") as f:
            json.dump(full_report, f, indent=2)
        
        # Step 6: Verify analysis
        assert report_file.exists()
        
        with open(report_file, "r") as f:
            loaded_report = json.load(f)
        
        assert "scaling_results" in loaded_report
        assert "analysis" in loaded_report
        assert len(loaded_report["scaling_results"]) == len(problem_sizes)
        
        # Verify scaling makes sense
        assert loaded_report["analysis"]["scaling_exponent"] > 0
        assert loaded_report["analysis"]["max_size_tested"] == max(problem_sizes)
        
        # Runtime should generally increase with size
        size_runtime_pairs = [(s, scaling_results[s]["runtime"]) for s in sizes]
        size_runtime_pairs.sort()
        
        # At least the trend should be generally increasing
        first_half_avg = np.mean([rt for s, rt in size_runtime_pairs[:len(size_runtime_pairs)//2]])
        second_half_avg = np.mean([rt for s, rt in size_runtime_pairs[len(size_runtime_pairs)//2:]])
        
        assert second_half_avg >= first_half_avg * 0.5  # Allow some variance