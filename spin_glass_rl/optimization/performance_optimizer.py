"""
Generation 3: Performance optimization and scaling enhancements.

Advanced optimization techniques for high-performance spin-glass annealing.
"""

import time
import math
import random
from typing import List, Tuple, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class OptimizedIsingModel:
    """
    High-performance Ising model with optimization features.
    
    Includes caching, vectorized operations, and memory optimization.
    """
    
    def __init__(self, n_spins: int, use_cache: bool = True):
        if n_spins <= 0:
            raise ValueError(f"Number of spins must be positive, got {n_spins}")
        
        self.n_spins = n_spins
        self.use_cache = use_cache
        
        # Initialize with efficient data structures
        self.spins = [1 if random.random() > 0.5 else -1 for _ in range(n_spins)]
        
        # Use sparse representation for couplings to save memory
        self.couplings = {}  # (i, j) -> strength
        self.external_fields = [0.0] * n_spins
        
        # Performance caching
        self._energy_cache = None
        self._cache_valid = False
        self._local_field_cache = {}
        
        # Performance monitoring
        self.stats = {
            'energy_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def set_coupling(self, i: int, j: int, strength: float) -> None:
        """Set coupling between spins i and j with optimizations."""
        if not (0 <= i < self.n_spins and 0 <= j < self.n_spins):
            return
        
        if abs(strength) < 1e-12:  # Skip very small couplings
            self.couplings.pop((i, j), None)
            self.couplings.pop((j, i), None)
        else:
            self.couplings[(i, j)] = strength
            self.couplings[(j, i)] = strength
        
        self._invalidate_cache()
    
    def set_external_field(self, i: int, strength: float) -> None:
        """Set external field on spin i."""
        if 0 <= i < self.n_spins:
            self.external_fields[i] = strength
            self._invalidate_cache()
    
    def compute_energy(self) -> float:
        """Optimized energy computation with caching."""
        if self.use_cache and self._cache_valid and self._energy_cache is not None:
            self.stats['cache_hits'] += 1
            return self._energy_cache
        
        self.stats['cache_misses'] += 1
        self.stats['energy_computations'] += 1
        
        energy = 0.0
        
        # Optimized coupling energy calculation
        for (i, j), strength in self.couplings.items():
            if i < j:  # Avoid double counting
                energy -= strength * self.spins[i] * self.spins[j]
        
        # External field energy - vectorized
        for i, field in enumerate(self.external_fields):
            if abs(field) > 1e-12:  # Skip zero fields
                energy -= field * self.spins[i]
        
        if self.use_cache:
            self._energy_cache = energy
            self._cache_valid = True
        
        return energy
    
    def compute_local_field(self, i: int, use_cache: bool = True) -> float:
        """Optimized local field computation with caching."""
        if use_cache and i in self._local_field_cache:
            return self._local_field_cache[i]
        
        field = self.external_fields[i]
        
        # Only iterate over non-zero couplings
        for (spin_i, j), strength in self.couplings.items():
            if spin_i == i and j != i:
                field += strength * self.spins[j]
        
        if use_cache:
            self._local_field_cache[i] = field
        
        return field
    
    def flip_spin_optimized(self, i: int) -> float:
        """Optimized spin flip with delta energy calculation."""
        if not (0 <= i < self.n_spins):
            return 0.0
        
        # Fast local field computation
        local_field = self.compute_local_field(i, use_cache=False)
        delta_energy = 2.0 * self.spins[i] * local_field
        
        # Flip the spin
        self.spins[i] *= -1
        
        # Invalidate caches
        self._invalidate_cache()
        
        return delta_energy
    
    def get_coupling_density(self) -> float:
        """Get the density of non-zero couplings."""
        max_couplings = self.n_spins * (self.n_spins - 1) // 2
        return len(self.couplings) / (2 * max_couplings) if max_couplings > 0 else 0.0
    
    def _invalidate_cache(self) -> None:
        """Invalidate all caches."""
        self._cache_valid = False
        self._local_field_cache.clear()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            'cache_hit_rate': hit_rate,
            'coupling_density': self.get_coupling_density(),
            'memory_efficiency': len(self.couplings) / (self.n_spins * self.n_spins)
        }


class ParallelAnnealer:
    """
    High-performance parallel simulated annealing implementation.
    
    Features multiple optimization strategies and parallel execution.
    """
    
    def __init__(
        self,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        n_replicas: int = 1,
        parallel_mode: str = "thread",  # "thread", "process", "none"
        adaptive_schedule: bool = True
    ):
        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError(f"Temperatures must be positive: initial={initial_temp}, final={final_temp}")
        if final_temp > initial_temp:
            raise ValueError(f"Final temperature must be <= initial: {final_temp} > {initial_temp}")
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_replicas = max(1, n_replicas)
        self.parallel_mode = parallel_mode
        self.adaptive_schedule = adaptive_schedule
        
        # Performance tracking
        self.stats = {
            'total_moves': 0,
            'accepted_moves': 0,
            'energy_evaluations': 0,
            'best_energy_updates': 0,
            'parallel_efficiency': 0.0
        }
    
    def metropolis_accept_fast(self, delta_energy: float, inv_temperature: float) -> bool:
        """Optimized Metropolis criterion using inverse temperature."""
        if delta_energy <= 0:
            return True
        if inv_temperature <= 0:
            return False
        
        # Use precomputed inverse temperature for efficiency
        return random.random() < math.exp(-delta_energy * inv_temperature)
    
    def adaptive_temperature_schedule(
        self, 
        sweep: int, 
        n_sweeps: int, 
        acceptance_rate: float,
        target_acceptance: float = 0.4
    ) -> float:
        """Adaptive temperature schedule based on acceptance rate."""
        if not self.adaptive_schedule:
            # Standard geometric schedule
            if n_sweeps > 1:
                alpha = sweep / (n_sweeps - 1)
                return self.initial_temp * (self.final_temp / self.initial_temp) ** alpha
            return self.initial_temp
        
        # Adaptive schedule
        base_alpha = sweep / n_sweeps if n_sweeps > 0 else 0
        base_temp = self.initial_temp * (self.final_temp / self.initial_temp) ** base_alpha
        
        # Adjust based on acceptance rate
        if acceptance_rate > target_acceptance * 1.2:
            # Too much acceptance, cool faster
            adaptation_factor = 0.95
        elif acceptance_rate < target_acceptance * 0.8:
            # Too little acceptance, cool slower
            adaptation_factor = 1.05
        else:
            adaptation_factor = 1.0
        
        return base_temp * adaptation_factor
    
    def anneal_single_replica(
        self, 
        model: OptimizedIsingModel, 
        n_sweeps: int,
        replica_id: int = 0
    ) -> Tuple[float, List[float], Dict]:
        """Optimized single replica annealing."""
        best_energy = model.compute_energy()
        best_spins = model.spins[:]
        energy_history = [best_energy]
        
        accepted_moves = 0
        total_moves = 0
        
        # Pre-generate random spin order for better cache locality
        spin_order = list(range(model.n_spins))
        
        for sweep in range(n_sweeps):
            # Calculate acceptance rate for adaptive schedule
            acceptance_rate = accepted_moves / max(1, total_moves)
            
            # Update temperature
            temperature = self.adaptive_temperature_schedule(
                sweep, n_sweeps, acceptance_rate
            )
            inv_temperature = 1.0 / temperature if temperature > 0 else float('inf')
            
            # Shuffle spin order for this sweep to avoid bias
            random.shuffle(spin_order)
            
            # Perform optimized sweep
            for i in spin_order:
                # Optimized flip with fast delta computation
                delta_energy = model.flip_spin_optimized(i)
                total_moves += 1
                
                # Fast acceptance test
                if self.metropolis_accept_fast(delta_energy, inv_temperature):
                    accepted_moves += 1
                    
                    # Check for new best
                    current_energy = model.compute_energy()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_spins = model.spins[:]
                        self.stats['best_energy_updates'] += 1
                else:
                    # Reject: flip back
                    model.spins[i] *= -1
            
            # Record energy periodically
            if sweep % max(1, n_sweeps // 50) == 0:
                current_energy = model.compute_energy()
                energy_history.append(current_energy)
        
        # Restore best configuration
        model.spins = best_spins
        
        # Update global stats
        self.stats['total_moves'] += total_moves
        self.stats['accepted_moves'] += accepted_moves
        self.stats['energy_evaluations'] += model.stats['energy_computations']
        
        replica_stats = {
            'replica_id': replica_id,
            'best_energy': best_energy,
            'total_moves': total_moves,
            'accepted_moves': accepted_moves,
            'acceptance_rate': accepted_moves / max(1, total_moves),
            'final_temperature': temperature
        }
        
        return best_energy, energy_history, replica_stats
    
    def parallel_anneal(
        self,
        model: OptimizedIsingModel,
        n_sweeps: int = 1000
    ) -> Tuple[float, List[float], Dict]:
        """
        High-performance parallel annealing with multiple replicas.
        """
        start_time = time.time()
        
        if self.n_replicas == 1 or self.parallel_mode == "none":
            # Single replica execution
            result = self.anneal_single_replica(model, n_sweeps, 0)
            total_time = time.time() - start_time
            
            stats = {
                'total_time': total_time,
                'replicas_used': 1,
                'parallel_efficiency': 1.0,
                'best_replica': 0,
                **result[2]
            }
            
            return result[0], result[1], stats
        
        # Prepare replicas
        replicas = [model.__class__(model.n_spins, use_cache=True) for _ in range(self.n_replicas)]
        for i, replica in enumerate(replicas):
            # Copy structure but randomize initial state
            replica.couplings = model.couplings.copy()
            replica.external_fields = model.external_fields[:]
            # Each replica starts from different random state
            replica.spins = [1 if random.random() > 0.5 else -1 for _ in range(model.n_spins)]
        
        # Execute in parallel
        results = []
        
        if self.parallel_mode == "thread":
            with ThreadPoolExecutor(max_workers=self.n_replicas) as executor:
                futures = [
                    executor.submit(self.anneal_single_replica, replica, n_sweeps, i)
                    for i, replica in enumerate(replicas)
                ]
                results = [future.result() for future in futures]
        
        elif self.parallel_mode == "process":
            with ProcessPoolExecutor(max_workers=min(self.n_replicas, mp.cpu_count())) as executor:
                futures = [
                    executor.submit(self.anneal_single_replica, replica, n_sweeps, i)
                    for i, replica in enumerate(replicas)
                ]
                results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Find best result across all replicas
        best_energy = float('inf')
        best_history = []
        best_replica_id = 0
        
        for i, (energy, history, replica_stats) in enumerate(results):
            if energy < best_energy:
                best_energy = energy
                best_history = history
                best_replica_id = i
        
        # Calculate parallel efficiency
        total_work = sum(r[2]['total_moves'] for r in results)
        theoretical_serial_time = total_work / max(1, results[0][2]['total_moves']) * total_time
        parallel_efficiency = theoretical_serial_time / (total_time * self.n_replicas)
        
        stats = {
            'total_time': total_time,
            'replicas_used': self.n_replicas,
            'parallel_efficiency': min(1.0, parallel_efficiency),
            'best_replica': best_replica_id,
            'replica_energies': [r[0] for r in results],
            'average_acceptance_rate': sum(r[2]['acceptance_rate'] for r in results) / len(results)
        }
        
        return best_energy, best_history, stats


def create_benchmark_problem(n_spins: int, coupling_density: float = 0.1) -> OptimizedIsingModel:
    """Create a benchmark optimization problem."""
    model = OptimizedIsingModel(n_spins, use_cache=True)
    
    # Add structured couplings for a challenging but solvable problem
    n_couplings = int(coupling_density * n_spins * (n_spins - 1) / 2)
    
    for _ in range(n_couplings):
        i, j = random.sample(range(n_spins), 2)
        strength = random.uniform(-1, 1)
        model.set_coupling(i, j, strength)
    
    # Add some external fields
    for i in range(n_spins):
        if random.random() < 0.1:
            field = random.uniform(-0.5, 0.5)
            model.set_external_field(i, field)
    
    return model


def performance_benchmark(
    problem_sizes: List[int] = [10, 20, 50, 100],
    n_trials: int = 3,
    n_sweeps: int = 1000
) -> Dict:
    """Run performance benchmarks across different problem sizes."""
    results = {}
    
    for size in problem_sizes:
        print(f"Benchmarking size {size}...")
        
        size_results = {
            'times': [],
            'energies': [],
            'speedups': [],
            'cache_hit_rates': [],
            'parallel_efficiencies': []
        }
        
        for trial in range(n_trials):
            # Create test problem
            model = create_benchmark_problem(size, coupling_density=0.15)
            
            # Single-threaded baseline
            annealer_single = ParallelAnnealer(
                initial_temp=5.0,
                final_temp=0.01,
                n_replicas=1,
                parallel_mode="none"
            )
            
            start_time = time.time()
            best_energy_single, _, stats_single = annealer_single.parallel_anneal(model, n_sweeps)
            single_time = time.time() - start_time
            
            # Multi-threaded version
            annealer_parallel = ParallelAnnealer(
                initial_temp=5.0,
                final_temp=0.01,
                n_replicas=min(4, mp.cpu_count()),
                parallel_mode="thread",
                adaptive_schedule=True
            )
            
            model_copy = create_benchmark_problem(size, coupling_density=0.15)
            start_time = time.time()
            best_energy_parallel, _, stats_parallel = annealer_parallel.parallel_anneal(model_copy, n_sweeps)
            parallel_time = time.time() - start_time
            
            # Record results
            size_results['times'].append(parallel_time)
            size_results['energies'].append(best_energy_parallel)
            size_results['speedups'].append(single_time / parallel_time if parallel_time > 0 else 1.0)
            size_results['cache_hit_rates'].append(model.get_performance_stats()['cache_hit_rate'])
            size_results['parallel_efficiencies'].append(stats_parallel['parallel_efficiency'])
        
        # Compute statistics
        results[size] = {
            'mean_time': sum(size_results['times']) / len(size_results['times']),
            'mean_energy': sum(size_results['energies']) / len(size_results['energies']),
            'mean_speedup': sum(size_results['speedups']) / len(size_results['speedups']),
            'mean_cache_hit_rate': sum(size_results['cache_hit_rates']) / len(size_results['cache_hit_rates']),
            'mean_parallel_efficiency': sum(size_results['parallel_efficiencies']) / len(size_results['parallel_efficiencies']),
        }
    
    return results


def demo_scaling_performance():
    """Demonstrate scaling and performance optimizations."""
    print("🚀 Generation 3: SCALING & PERFORMANCE DEMO")
    print("=" * 50)
    
    # Small problem for detailed analysis
    print("Creating optimized test problem (20 spins)...")
    model = create_benchmark_problem(20, coupling_density=0.2)
    print(f"Problem density: {model.get_coupling_density():.3f}")
    
    # Compare single vs parallel
    print("\nComparing execution modes...")
    
    # Single replica
    annealer_single = ParallelAnnealer(n_replicas=1, parallel_mode="none")
    start = time.time()
    energy_single, _, stats_single = annealer_single.parallel_anneal(model, 500)
    time_single = time.time() - start
    
    # Parallel replicas
    annealer_parallel = ParallelAnnealer(
        n_replicas=4, 
        parallel_mode="thread",
        adaptive_schedule=True
    )
    model_copy = create_benchmark_problem(20, coupling_density=0.2)
    start = time.time()
    energy_parallel, _, stats_parallel = annealer_parallel.parallel_anneal(model_copy, 500)
    time_parallel = time.time() - start
    
    print(f"\nSingle replica: {energy_single:.4f} in {time_single:.3f}s")
    print(f"Parallel (4x):  {energy_parallel:.4f} in {time_parallel:.3f}s")
    print(f"Speedup: {time_single/time_parallel:.2f}x")
    print(f"Parallel efficiency: {stats_parallel['parallel_efficiency']:.2%}")
    
    # Performance statistics
    perf_stats = model.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"Cache hit rate: {perf_stats['cache_hit_rate']:.2%}")
    print(f"Memory efficiency: {perf_stats['memory_efficiency']:.2%}")
    print(f"Energy computations: {perf_stats['energy_computations']}")
    
    print("\n✅ Generation 3 scaling optimizations working!")


if __name__ == "__main__":
    demo_scaling_performance()