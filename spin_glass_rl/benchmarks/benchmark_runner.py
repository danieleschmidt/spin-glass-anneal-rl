"""Benchmark execution and coordination framework."""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from spin_glass_rl.utils.exceptions import ValidationError
from spin_glass_rl.utils.validation import validate_numeric, validate_string
from spin_glass_rl.utils.performance import profile


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str
    description: str = ""
    n_trials: int = 10
    timeout_seconds: float = 300.0
    random_seed: Optional[int] = None
    parallel_trials: bool = False
    max_workers: int = 4
    save_results: bool = True
    save_intermediate: bool = False
    log_level: str = "INFO"
    device: str = "cpu"
    
    def __post_init__(self):
        validate_string(self.name, "name", min_length=1)
        validate_numeric(self.n_trials, "n_trials", min_value=1, integer=True)
        validate_numeric(self.timeout_seconds, "timeout_seconds", min_value=0.1)


@dataclass 
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    metrics: Dict[str, Any]
    execution_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    trial_results: Optional[List[Dict]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle non-serializable types
        for key, value in result['metrics'].items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                result['metrics'][key] = value.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        config = BenchmarkConfig(**data['config'])
        return cls(
            config=config,
            metrics=data['metrics'],
            execution_time=data['execution_time'],
            timestamp=data['timestamp'],
            success=data['success'],
            error_message=data.get('error_message'),
            trial_results=data.get('trial_results'),
            metadata=data.get('metadata')
        )


class Benchmark:
    """Base class for benchmarks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Setup before benchmark execution."""
        pass
    
    def run_trial(self, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run a single benchmark trial."""
        raise NotImplementedError("Subclasses must implement run_trial")
    
    def teardown(self, config: BenchmarkConfig) -> None:
        """Cleanup after benchmark execution."""
        pass
    
    def aggregate_results(self, trial_results: List[Dict[str, Any]], config: BenchmarkConfig) -> Dict[str, Any]:
        """Aggregate results from multiple trials."""
        if not trial_results:
            return {}
        
        # Collect numeric metrics
        numeric_metrics = {}
        for result in trial_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute statistics
        aggregated = {}
        for key, values in numeric_metrics.items():
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
                aggregated[f"{key}_median"] = float(np.median(values))
                if len(values) > 1:
                    aggregated[f"{key}_p95"] = float(np.percentile(values, 95))
                    aggregated[f"{key}_p99"] = float(np.percentile(values, 99))
        
        # Add non-numeric aggregations
        aggregated.update({
            'n_trials': len(trial_results),
            'success_rate': sum(1 for r in trial_results if r.get('success', False)) / len(trial_results)
        })
        
        return aggregated


class BenchmarkRunner:
    """Main benchmark execution engine."""
    
    def __init__(self, results_dir: Optional[Union[str, Path]] = None):
        self.results_dir = Path(results_dir or "benchmark_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Track running benchmarks
        self._running_benchmarks = {}
    
    @profile("benchmark_execution")
    def run_benchmark(
        self,
        benchmark: Benchmark,
        config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Execute a benchmark with given configuration."""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        self.logger.info(f"Starting benchmark: {benchmark.name}")
        self.logger.info(f"Configuration: {config}")
        
        # Setup random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.random_seed)
        
        try:
            # Setup benchmark
            benchmark.setup(config)
            
            # Run trials
            if config.parallel_trials and config.n_trials > 1:
                trial_results = self._run_trials_parallel(benchmark, config)
            else:
                trial_results = self._run_trials_sequential(benchmark, config)
            
            # Aggregate results
            metrics = benchmark.aggregate_results(trial_results, config)
            
            # Cleanup
            benchmark.teardown(config)
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                config=config,
                metrics=metrics,
                execution_time=execution_time,
                timestamp=timestamp,
                success=True,
                trial_results=trial_results if config.save_intermediate else None
            )
            
            self.logger.info(f"Benchmark completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Benchmark failed: {e}")
            
            result = BenchmarkResult(
                config=config,
                metrics={},
                execution_time=execution_time,
                timestamp=timestamp,
                success=False,
                error_message=str(e)
            )
        
        # Save results
        if config.save_results:
            self._save_result(result)
        
        return result
    
    def _run_trials_sequential(self, benchmark: Benchmark, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Run trials sequentially."""
        trial_results = []
        
        for trial_id in range(config.n_trials):
            self.logger.debug(f"Running trial {trial_id + 1}/{config.n_trials}")
            
            trial_start = time.time()
            
            try:
                # Run trial with timeout
                result = self._run_single_trial(benchmark, trial_id, config)
                result['success'] = True
                result['trial_time'] = time.time() - trial_start
                
            except Exception as e:
                self.logger.warning(f"Trial {trial_id} failed: {e}")
                result = {
                    'trial_id': trial_id,
                    'success': False,
                    'error': str(e),
                    'trial_time': time.time() - trial_start
                }
            
            trial_results.append(result)
            
            # Check if we should continue after failures
            success_rate = sum(1 for r in trial_results if r.get('success', False)) / len(trial_results)
            if len(trial_results) >= 3 and success_rate < 0.5:
                self.logger.warning("High failure rate, stopping trials")
                break
        
        return trial_results
    
    def _run_trials_parallel(self, benchmark: Benchmark, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Run trials in parallel."""
        trial_results = []
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all trials
            future_to_trial = {
                executor.submit(self._run_single_trial, benchmark, i, config): i
                for i in range(config.n_trials)
            }
            
            # Collect results
            for future in as_completed(future_to_trial):
                trial_id = future_to_trial[future]
                
                try:
                    result = future.result(timeout=config.timeout_seconds)
                    result['success'] = True
                    
                except Exception as e:
                    self.logger.warning(f"Trial {trial_id} failed: {e}")
                    result = {
                        'trial_id': trial_id,
                        'success': False,
                        'error': str(e)
                    }
                
                trial_results.append(result)
        
        # Sort by trial_id
        trial_results.sort(key=lambda x: x.get('trial_id', 0))
        
        return trial_results
    
    def _run_single_trial(self, benchmark: Benchmark, trial_id: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run a single trial with timeout handling."""
        result = benchmark.run_trial(trial_id, config)
        result['trial_id'] = trial_id
        return result
    
    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file."""
        filename = f"{result.config.name}_{result.timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_results(self, pattern: str = "*.json") -> List[BenchmarkResult]:
        """Load benchmark results from files."""
        results = []
        
        for filepath in self.results_dir.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                result = BenchmarkResult.from_dict(data)
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {filepath}: {e}")
        
        return sorted(results, key=lambda x: x.timestamp)
    
    def get_summary(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        results = self.load_results()
        
        if benchmark_name:
            results = [r for r in results if r.config.name == benchmark_name]
        
        if not results:
            return {"message": "No results found"}
        
        summary = {
            "total_benchmarks": len(results),
            "success_rate": sum(1 for r in results if r.success) / len(results),
            "total_execution_time": sum(r.execution_time for r in results),
            "benchmark_names": list(set(r.config.name for r in results)),
            "date_range": {
                "earliest": min(r.timestamp for r in results),
                "latest": max(r.timestamp for r in results)
            }
        }
        
        # Performance metrics
        successful_results = [r for r in results if r.success]
        if successful_results:
            execution_times = [r.execution_time for r in successful_results]
            summary["performance"] = {
                "avg_execution_time": np.mean(execution_times),
                "min_execution_time": np.min(execution_times),
                "max_execution_time": np.max(execution_times)
            }
        
        return summary
    
    def cleanup_old_results(self, days_old: int = 30) -> None:
        """Remove old benchmark results."""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
        removed_count = 0
        
        for filepath in self.results_dir.glob("*.json"):
            if filepath.stat().st_mtime < cutoff_date:
                filepath.unlink()
                removed_count += 1
        
        self.logger.info(f"Removed {removed_count} old result files")


class BenchmarkSuite:
    """Collection of related benchmarks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: List[Benchmark] = []
        self.runner = BenchmarkRunner()
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)
        self.logger.info(f"Added benchmark: {benchmark.name}")
    
    def run_suite(
        self,
        base_config: Optional[BenchmarkConfig] = None,
        benchmark_configs: Optional[Dict[str, BenchmarkConfig]] = None
    ) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        if base_config is None:
            base_config = BenchmarkConfig(name=self.name)
        
        benchmark_configs = benchmark_configs or {}
        results = []
        
        self.logger.info(f"Starting benchmark suite: {self.name}")
        self.logger.info(f"Running {len(self.benchmarks)} benchmarks")
        
        for benchmark in self.benchmarks:
            # Use specific config if provided, otherwise use base config
            if benchmark.name in benchmark_configs:
                config = benchmark_configs[benchmark.name]
            else:
                # Create config for this benchmark
                config = BenchmarkConfig(
                    name=f"{base_config.name}_{benchmark.name}",
                    description=benchmark.description,
                    n_trials=base_config.n_trials,
                    timeout_seconds=base_config.timeout_seconds,
                    random_seed=base_config.random_seed,
                    parallel_trials=base_config.parallel_trials,
                    max_workers=base_config.max_workers,
                    save_results=base_config.save_results,
                    save_intermediate=base_config.save_intermediate,
                    log_level=base_config.log_level,
                    device=base_config.device
                )
            
            result = self.runner.run_benchmark(benchmark, config)
            results.append(result)
            
            # Stop suite if benchmark fails critically
            if not result.success and "critical" in benchmark.description.lower():
                self.logger.error(f"Critical benchmark failed, stopping suite")
                break
        
        self.logger.info(f"Suite completed with {sum(1 for r in results if r.success)}/{len(results)} successful benchmarks")
        
        return results
    
    def get_suite_summary(self) -> Dict[str, Any]:
        """Get summary of suite execution."""
        return self.runner.get_summary(self.name)