"""Performance benchmarking suite for annealing algorithms."""

import time
import torch
import numpy as np
import psutil
import platform
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.ising_model import IsingModel, IsingModelConfig
from ..annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from ..annealing.parallel_tempering import ParallelTempering, ParallelTemperingConfig
from ..annealing.temperature_scheduler import ScheduleType
from ..problems.routing import TSPProblem
from ..problems.scheduling import SchedulingProblem
from ..utils.exceptions import BenchmarkError


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    
    problem_sizes: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    algorithms: List[str] = field(default_factory=lambda: ["gpu_annealer", "parallel_tempering"])
    temperature_schedules: List[ScheduleType] = field(default_factory=lambda: [
        ScheduleType.GEOMETRIC, ScheduleType.LINEAR, ScheduleType.EXPONENTIAL
    ])
    n_trials: int = 5
    max_sweeps: int = 1000
    timeout_seconds: float = 300.0
    measure_memory: bool = True
    measure_energy_convergence: bool = True
    measure_scaling: bool = True
    save_plots: bool = True
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.problem_sizes:
            raise ValueError("Problem sizes cannot be empty")
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    algorithm: str
    problem_size: int
    problem_type: str
    temperature_schedule: str
    trial: int
    
    # Performance metrics
    execution_time: float
    energy_final: float
    energy_best: float
    convergence_sweep: int
    quality_score: float
    
    # Resource usage
    memory_peak: float
    memory_mean: float
    cpu_usage: float
    gpu_usage: float
    
    # Algorithm-specific metrics
    acceptance_rate: float
    temperature_final: float
    n_sweeps_completed: int
    
    # Solution quality
    is_feasible: bool
    constraint_violations: Dict[str, float]
    
    # Metadata
    device: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemProfiler:
    """Profile system capabilities and configuration."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
        }
        
        # GPU information
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor
                })
            info['gpu_info'] = gpu_info
        else:
            info['cuda_available'] = False
        
        # PyTorch information
        info['torch_version'] = torch.__version__
        info['torch_backends'] = {
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None,
            'mkl_enabled': torch.backends.mkl.is_available(),
            'openmp_enabled': torch.backends.openmp.is_available()
        }
        
        return info


class PerformanceMonitor:
    """Monitor performance metrics during benchmark execution."""
    
    def __init__(self, measure_memory: bool = True, sample_interval: float = 0.1):
        """Initialize performance monitor.
        
        Args:
            measure_memory: Whether to measure memory usage
            sample_interval: Sampling interval in seconds
        """
        self.measure_memory = measure_memory
        self.sample_interval = sample_interval
        self.samples = []
        self.monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.samples = []
        
        if self.measure_memory:
            import threading
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self.samples:
            return {
                'memory_peak': 0.0,
                'memory_mean': 0.0,
                'cpu_usage': 0.0,
                'gpu_usage': 0.0
            }
        
        memory_values = [s['memory'] for s in self.samples]
        cpu_values = [s['cpu'] for s in self.samples]
        gpu_values = [s['gpu'] for s in self.samples if s['gpu'] is not None]
        
        return {
            'memory_peak': max(memory_values) if memory_values else 0.0,
            'memory_mean': np.mean(memory_values) if memory_values else 0.0,
            'cpu_usage': np.mean(cpu_values) if cpu_values else 0.0,
            'gpu_usage': np.mean(gpu_values) if gpu_values else 0.0
        }
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # GPU usage (if available)
                gpu_usage = None
                if torch.cuda.is_available():
                    try:
                        gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                    except:
                        pass
                
                self.samples.append({
                    'timestamp': time.time(),
                    'memory': memory_mb,
                    'cpu': cpu_percent,
                    'gpu': gpu_usage
                })
                
                time.sleep(self.sample_interval)
                
            except Exception:
                # Continue monitoring even if some metrics fail
                continue


class PerformanceBenchmark:
    """Main performance benchmarking class."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize performance benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.system_info = SystemProfiler.get_system_info()
        self.results = []
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize problem generators
        self.problem_generators = {
            'tsp': self._generate_tsp_problem,
            'scheduling': self._generate_scheduling_problem,
            'random_ising': self._generate_random_ising
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        self.logger.info("Starting comprehensive performance benchmark")
        self.logger.info(f"System: {self.system_info['platform']}")
        self.logger.info(f"GPU Available: {self.system_info['cuda_available']}")
        
        benchmark_start = time.time()
        
        try:
            # Run benchmarks for different problem types
            problem_types = ['tsp', 'scheduling', 'random_ising']
            
            for problem_type in problem_types:
                self.logger.info(f"Benchmarking {problem_type} problems")
                self._benchmark_problem_type(problem_type)
            
            # Generate analysis and reports
            analysis = self._analyze_results()
            
            if self.config.save_plots:
                self._generate_plots()
            
            # Save results
            self._save_results()
            
            benchmark_time = time.time() - benchmark_start
            self.logger.info(f"Benchmark completed in {benchmark_time:.2f} seconds")
            
            return {
                'results': self.results,
                'analysis': analysis,
                'system_info': self.system_info,
                'benchmark_time': benchmark_time
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise BenchmarkError(f"Benchmark execution failed: {e}")
    
    def _benchmark_problem_type(self, problem_type: str):
        """Benchmark specific problem type."""
        for size in self.config.problem_sizes:
            self.logger.info(f"  Problem size: {size}")
            
            for algorithm in self.config.algorithms:
                for schedule in self.config.temperature_schedules:
                    self._benchmark_configuration(
                        problem_type, size, algorithm, schedule
                    )
    
    def _benchmark_configuration(
        self,
        problem_type: str,
        size: int,
        algorithm: str,
        schedule: ScheduleType
    ):
        """Benchmark specific configuration."""
        self.logger.debug(f"    Algorithm: {algorithm}, Schedule: {schedule.value}")
        
        for trial in range(self.config.n_trials):
            try:
                result = self._run_single_benchmark(
                    problem_type, size, algorithm, schedule, trial
                )
                self.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Trial {trial} failed: {e}")
                continue
    
    def _run_single_benchmark(
        self,
        problem_type: str,
        size: int,
        algorithm: str,
        schedule: ScheduleType,
        trial: int
    ) -> BenchmarkResult:
        """Run single benchmark trial."""
        # Generate problem
        problem = self.problem_generators[problem_type](size)
        
        # Create annealer
        annealer_config = GPUAnnealerConfig(
            n_sweeps=self.config.max_sweeps,
            schedule_type=schedule,
            random_seed=trial * 1000 + size  # Reproducible but varied seeds
        )
        
        if algorithm == "gpu_annealer":
            annealer = GPUAnnealer(annealer_config)
        elif algorithm == "parallel_tempering":
            pt_config = ParallelTemperingConfig(
                n_replicas=4,
                n_sweeps=self.config.max_sweeps // 4,  # Adjust for multiple replicas
                random_seed=trial * 1000 + size
            )
            annealer = ParallelTempering(pt_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Setup monitoring
        monitor = PerformanceMonitor(self.config.measure_memory)
        monitor.start_monitoring()
        
        # Run annealing
        start_time = time.time()
        
        try:
            if hasattr(problem, 'solve_with_annealer'):
                # Problem with built-in solver
                solution = problem.solve_with_annealer(annealer)
                annealing_result = solution  # Assume solution contains annealing result
            else:
                # Direct Ising model
                annealing_result = annealer.anneal(problem)
            
            execution_time = time.time() - start_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise BenchmarkError(f"Annealing failed: {e}")
        
        finally:
            resource_stats = monitor.stop_monitoring()
        
        # Extract metrics
        energy_final = annealing_result.best_energy
        energy_history = annealing_result.energy_history
        
        # Find convergence point
        convergence_sweep = len(energy_history)
        if len(energy_history) > 10:
            # Find when energy stops improving significantly
            energy_diffs = np.diff(energy_history)
            for i in range(len(energy_diffs) - 10):
                if np.std(energy_diffs[i:i+10]) < 0.01:
                    convergence_sweep = i
                    break
        
        # Calculate quality score (problem-dependent)
        quality_score = self._calculate_quality_score(problem, annealing_result)
        
        # Determine device
        device = "GPU" if torch.cuda.is_available() and hasattr(annealer, 'use_cuda') and annealer.use_cuda else "CPU"
        
        # Create result
        result = BenchmarkResult(
            algorithm=algorithm,
            problem_size=size,
            problem_type=problem_type,
            temperature_schedule=schedule.value,
            trial=trial,
            execution_time=execution_time,
            energy_final=energy_final,
            energy_best=min(energy_history) if energy_history else energy_final,
            convergence_sweep=convergence_sweep,
            quality_score=quality_score,
            memory_peak=resource_stats['memory_peak'],
            memory_mean=resource_stats['memory_mean'],
            cpu_usage=resource_stats['cpu_usage'],
            gpu_usage=resource_stats['gpu_usage'],
            acceptance_rate=np.mean(annealing_result.acceptance_rate_history) if annealing_result.acceptance_rate_history else 0.0,
            temperature_final=annealing_result.temperature_history[-1] if annealing_result.temperature_history else 0.0,
            n_sweeps_completed=annealing_result.n_sweeps,
            is_feasible=getattr(annealing_result, 'is_feasible', True),
            constraint_violations=getattr(annealing_result, 'constraint_violations', {}),
            device=device,
            timestamp=time.time(),
            metadata={
                'problem_instance': str(type(problem).__name__),
                'annealer_config': annealer_config.__dict__ if hasattr(annealer_config, '__dict__') else {}
            }
        )
        
        return result
    
    def _generate_tsp_problem(self, size: int) -> TSPProblem:
        """Generate TSP problem of given size."""
        tsp = TSPProblem()
        tsp.generate_random_instance(n_locations=size, area_size=100.0)
        return tsp
    
    def _generate_scheduling_problem(self, size: int) -> SchedulingProblem:
        """Generate scheduling problem of given size."""
        problem = SchedulingProblem()
        problem.generate_random_instance(
            n_tasks=size,
            n_agents=max(2, size // 5),
            time_horizon=100.0
        )
        return problem
    
    def _generate_random_ising(self, size: int) -> IsingModel:
        """Generate random Ising model of given size."""
        config = IsingModelConfig(n_spins=size, use_sparse=size > 100)
        model = IsingModel(config)
        
        # Add random couplings
        n_couplings = min(size * (size - 1) // 4, 1000)  # Limit for large models
        for _ in range(n_couplings):
            i, j = np.random.choice(size, 2, replace=False)
            coupling = np.random.normal(0, 1)
            model.set_coupling(i, j, coupling)
        
        # Add random external fields
        fields = np.random.normal(0, 0.5, size)
        model.set_external_fields(torch.tensor(fields, dtype=torch.float32))
        
        return model
    
    def _calculate_quality_score(self, problem: Any, result: Any) -> float:
        """Calculate problem-specific quality score."""
        # Generic quality score based on energy improvement
        if hasattr(result, 'energy_history') and result.energy_history:
            initial_energy = result.energy_history[0]
            final_energy = result.best_energy
            
            if initial_energy != 0:
                improvement = (initial_energy - final_energy) / abs(initial_energy)
                return max(0.0, min(1.0, improvement))
        
        return 0.5  # Default neutral score
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        if not self.results:
            return {}
        
        analysis = {
            'summary': self._analyze_summary(),
            'algorithm_comparison': self._analyze_algorithms(),
            'scaling_analysis': self._analyze_scaling(),
            'schedule_comparison': self._analyze_schedules(),
            'resource_usage': self._analyze_resources()
        }
        
        return analysis
    
    def _analyze_summary(self) -> Dict[str, Any]:
        """Generate summary analysis."""
        return {
            'total_runs': len(self.results),
            'problem_types': list(set(r.problem_type for r in self.results)),
            'algorithms': list(set(r.algorithm for r in self.results)),
            'problem_sizes': sorted(list(set(r.problem_size for r in self.results))),
            'avg_execution_time': np.mean([r.execution_time for r in self.results]),
            'avg_quality_score': np.mean([r.quality_score for r in self.results]),
            'feasibility_rate': np.mean([r.is_feasible for r in self.results])
        }
    
    def _analyze_algorithms(self) -> Dict[str, Any]:
        """Compare algorithm performance."""
        algorithm_stats = defaultdict(list)
        
        for result in self.results:
            algorithm_stats[result.algorithm].append(result)
        
        comparison = {}
        for algorithm, results in algorithm_stats.items():
            comparison[algorithm] = {
                'avg_execution_time': np.mean([r.execution_time for r in results]),
                'avg_quality_score': np.mean([r.quality_score for r in results]),
                'avg_energy': np.mean([r.energy_best for r in results]),
                'feasibility_rate': np.mean([r.is_feasible for r in results]),
                'avg_memory_usage': np.mean([r.memory_peak for r in results])
            }
        
        return comparison
    
    def _analyze_scaling(self) -> Dict[str, Any]:
        """Analyze scaling behavior."""
        scaling_data = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            key = f"{result.algorithm}_{result.problem_type}"
            scaling_data[key]['sizes'].append(result.problem_size)
            scaling_data[key]['times'].append(result.execution_time)
            scaling_data[key]['memory'].append(result.memory_peak)
        
        scaling_analysis = {}
        for key, data in scaling_data.items():
            if len(data['sizes']) > 1:
                # Fit polynomial to estimate scaling
                sizes = np.array(data['sizes'])
                times = np.array(data['times'])
                
                try:
                    coeffs = np.polyfit(np.log(sizes), np.log(times), 1)
                    scaling_exponent = coeffs[0]
                except:
                    scaling_exponent = 1.0
                
                scaling_analysis[key] = {
                    'scaling_exponent': scaling_exponent,
                    'size_range': [min(sizes), max(sizes)],
                    'time_range': [min(times), max(times)]
                }
        
        return scaling_analysis
    
    def _analyze_schedules(self) -> Dict[str, Any]:
        """Compare temperature schedules."""
        schedule_stats = defaultdict(list)
        
        for result in self.results:
            schedule_stats[result.temperature_schedule].append(result)
        
        comparison = {}
        for schedule, results in schedule_stats.items():
            comparison[schedule] = {
                'avg_execution_time': np.mean([r.execution_time for r in results]),
                'avg_quality_score': np.mean([r.quality_score for r in results]),
                'avg_convergence_sweep': np.mean([r.convergence_sweep for r in results]),
                'avg_acceptance_rate': np.mean([r.acceptance_rate for r in results])
            }
        
        return comparison
    
    def _analyze_resources(self) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        return {
            'memory_usage': {
                'peak_max': max(r.memory_peak for r in self.results),
                'peak_avg': np.mean([r.memory_peak for r in self.results]),
                'peak_std': np.std([r.memory_peak for r in self.results])
            },
            'cpu_usage': {
                'avg': np.mean([r.cpu_usage for r in self.results]),
                'std': np.std([r.cpu_usage for r in self.results])
            },
            'gpu_usage': {
                'avg': np.mean([r.gpu_usage for r in self.results if r.gpu_usage > 0]),
                'std': np.std([r.gpu_usage for r in self.results if r.gpu_usage > 0])
            } if any(r.gpu_usage > 0 for r in self.results) else None
        }
    
    def _generate_plots(self):
        """Generate performance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8')
            fig_size = (12, 8)
            
            # Performance vs problem size
            self._plot_performance_scaling(fig_size)
            
            # Algorithm comparison
            self._plot_algorithm_comparison(fig_size)
            
            # Resource usage
            self._plot_resource_usage(fig_size)
            
            # Quality scores
            self._plot_quality_scores(fig_size)
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")
    
    def _plot_performance_scaling(self, fig_size: Tuple[int, int]):
        """Plot performance scaling with problem size."""
        plt.figure(figsize=fig_size)
        
        # Group by algorithm and problem type
        for algorithm in set(r.algorithm for r in self.results):
            for problem_type in set(r.problem_type for r in self.results):
                subset = [r for r in self.results 
                         if r.algorithm == algorithm and r.problem_type == problem_type]
                
                if len(subset) > 1:
                    sizes = [r.problem_size for r in subset]
                    times = [r.execution_time for r in subset]
                    
                    plt.scatter(sizes, times, label=f"{algorithm}_{problem_type}", alpha=0.7)
        
        plt.xlabel('Problem Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Performance Scaling')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'performance_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_algorithm_comparison(self, fig_size: Tuple[int, int]):
        """Plot algorithm comparison."""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        
        # Execution time comparison
        algorithms = list(set(r.algorithm for r in self.results))
        exec_times = [[r.execution_time for r in self.results if r.algorithm == alg] 
                     for alg in algorithms]
        
        axes[0, 0].boxplot(exec_times, labels=algorithms)
        axes[0, 0].set_title('Execution Time by Algorithm')
        axes[0, 0].set_ylabel('Time (seconds)')
        
        # Quality score comparison
        quality_scores = [[r.quality_score for r in self.results if r.algorithm == alg] 
                         for alg in algorithms]
        
        axes[0, 1].boxplot(quality_scores, labels=algorithms)
        axes[0, 1].set_title('Quality Score by Algorithm')
        axes[0, 1].set_ylabel('Quality Score')
        
        # Memory usage comparison
        memory_usage = [[r.memory_peak for r in self.results if r.algorithm == alg] 
                       for alg in algorithms]
        
        axes[1, 0].boxplot(memory_usage, labels=algorithms)
        axes[1, 0].set_title('Memory Usage by Algorithm')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # Convergence comparison
        convergence = [[r.convergence_sweep for r in self.results if r.algorithm == alg] 
                      for alg in algorithms]
        
        axes[1, 1].boxplot(convergence, labels=algorithms)
        axes[1, 1].set_title('Convergence Speed by Algorithm')
        axes[1, 1].set_ylabel('Sweeps to Convergence')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_usage(self, fig_size: Tuple[int, int]):
        """Plot resource usage patterns."""
        plt.figure(figsize=fig_size)
        
        # Memory vs execution time
        memory_values = [r.memory_peak for r in self.results]
        time_values = [r.execution_time for r in self.results]
        problem_sizes = [r.problem_size for r in self.results]
        
        scatter = plt.scatter(memory_values, time_values, c=problem_sizes, 
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Problem Size')
        plt.xlabel('Peak Memory Usage (MB)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Resource Usage Patterns')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'resource_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_scores(self, fig_size: Tuple[int, int]):
        """Plot quality score distributions."""
        plt.figure(figsize=fig_size)
        
        # Quality scores by problem type
        problem_types = list(set(r.problem_type for r in self.results))
        quality_data = [[r.quality_score for r in self.results if r.problem_type == pt] 
                       for pt in problem_types]
        
        plt.boxplot(quality_data, labels=problem_types)
        plt.title('Solution Quality by Problem Type')
        plt.ylabel('Quality Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'quality_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save detailed results as JSON
        results_data = []
        for result in self.results:
            # Convert to dict, handling special types
            result_dict = {}
            for field in result.__dataclass_fields__:
                value = getattr(result, field)
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    value = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                result_dict[field] = value
            results_data.append(result_dict)
        
        # Save results
        with open(self.output_path / 'benchmark_results.json', 'w') as f:
            json.dump({
                'results': results_data,
                'system_info': self.system_info,
                'config': self.config.__dict__
            }, f, indent=2, default=str)
        
        # Save summary report
        analysis = self._analyze_results()
        with open(self.output_path / 'benchmark_summary.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save human-readable report
        self._save_text_report(analysis)
        
        self.logger.info(f"Results saved to {self.output_path}")
    
    def _save_text_report(self, analysis: Dict[str, Any]):
        """Save human-readable text report."""
        with open(self.output_path / 'benchmark_report.txt', 'w') as f:
            f.write("SPIN-GLASS-ANNEAL-RL PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # System information
            f.write("SYSTEM INFORMATION:\n")
            f.write(f"Platform: {self.system_info['platform']}\n")
            f.write(f"CPU: {self.system_info['processor']}\n")
            f.write(f"CPU Cores: {self.system_info['cpu_count']}\n")
            f.write(f"Memory: {self.system_info['memory_total'] / 1024**3:.1f} GB\n")
            f.write(f"CUDA Available: {self.system_info['cuda_available']}\n")
            if self.system_info['cuda_available']:
                f.write(f"GPU Count: {self.system_info['gpu_count']}\n")
                for i, gpu in enumerate(self.system_info['gpu_info']):
                    f.write(f"  GPU {i}: {gpu['name']} ({gpu['memory_total'] / 1024**3:.1f} GB)\n")
            f.write("\n")
            
            # Summary
            if 'summary' in analysis:
                summary = analysis['summary']
                f.write("BENCHMARK SUMMARY:\n")
                f.write(f"Total Runs: {summary['total_runs']}\n")
                f.write(f"Problem Types: {', '.join(summary['problem_types'])}\n")
                f.write(f"Algorithms: {', '.join(summary['algorithms'])}\n")
                f.write(f"Problem Sizes: {summary['problem_sizes']}\n")
                f.write(f"Average Execution Time: {summary['avg_execution_time']:.3f} seconds\n")
                f.write(f"Average Quality Score: {summary['avg_quality_score']:.3f}\n")
                f.write(f"Feasibility Rate: {summary['feasibility_rate']:.1%}\n\n")
            
            # Algorithm comparison
            if 'algorithm_comparison' in analysis:
                f.write("ALGORITHM COMPARISON:\n")
                for algorithm, stats in analysis['algorithm_comparison'].items():
                    f.write(f"  {algorithm}:\n")
                    f.write(f"    Avg Execution Time: {stats['avg_execution_time']:.3f}s\n")
                    f.write(f"    Avg Quality Score: {stats['avg_quality_score']:.3f}\n")
                    f.write(f"    Feasibility Rate: {stats['feasibility_rate']:.1%}\n")
                    f.write(f"    Avg Memory Usage: {stats['avg_memory_usage']:.1f} MB\n\n")


class BenchmarkSuite:
    """Complete benchmarking suite with multiple configurations."""
    
    def __init__(self, output_base_dir: str = "benchmark_suite"):
        """Initialize benchmark suite.
        
        Args:
            output_base_dir: Base directory for all benchmark outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def run_full_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite with multiple configurations."""
        suite_results = {}
        
        # Quick benchmark for development
        quick_config = BenchmarkConfig(
            problem_sizes=[10, 20, 50],
            n_trials=3,
            max_sweeps=500,
            output_dir=str(self.output_base_dir / "quick")
        )
        suite_results['quick'] = PerformanceBenchmark(quick_config).run_comprehensive_benchmark()
        
        # Comprehensive benchmark
        comprehensive_config = BenchmarkConfig(
            problem_sizes=[10, 25, 50, 100, 200, 500],
            n_trials=5,
            max_sweeps=2000,
            output_dir=str(self.output_base_dir / "comprehensive")
        )
        suite_results['comprehensive'] = PerformanceBenchmark(comprehensive_config).run_comprehensive_benchmark()
        
        # Scaling benchmark
        scaling_config = BenchmarkConfig(
            problem_sizes=[10, 20, 50, 100, 200, 500, 1000],
            algorithms=["gpu_annealer"],  # Focus on main algorithm
            temperature_schedules=[ScheduleType.GEOMETRIC],  # Single schedule
            n_trials=3,
            max_sweeps=1000,
            output_dir=str(self.output_base_dir / "scaling")
        )
        suite_results['scaling'] = PerformanceBenchmark(scaling_config).run_comprehensive_benchmark()
        
        # Save suite summary
        self._save_suite_summary(suite_results)
        
        return suite_results
    
    def _save_suite_summary(self, suite_results: Dict[str, Any]):
        """Save benchmark suite summary."""
        summary_path = self.output_base_dir / "suite_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump({
                'suite_results': {k: v.get('analysis', {}) for k, v in suite_results.items()},
                'timestamp': time.time(),
                'system_info': SystemProfiler.get_system_info()
            }, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark suite completed. Results saved to {self.output_base_dir}")