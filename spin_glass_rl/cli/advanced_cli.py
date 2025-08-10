"""Advanced CLI interface with comprehensive features."""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import signal

from spin_glass_rl.utils.robust_logging import setup_global_logging, get_logger, LoggingContext
from spin_glass_rl.utils.monitoring import get_system_health_report, start_global_monitoring, stop_global_monitoring
from spin_glass_rl.utils.validation import ValidationError
from spin_glass_rl.utils.security import audit_log
from spin_glass_rl.optimization.performance_cache import get_cache_performance_summary, clear_all_caches
from spin_glass_rl.optimization.adaptive_scaling import AdaptiveScaler, ScalingPolicy
from spin_glass_rl.distributed.load_balancer import LoadBalancer

logger = get_logger("cli")


class AdvancedCLI:
    """Advanced command-line interface with monitoring and scaling."""
    
    def __init__(self):
        self.logger = None
        self.load_balancer = None
        self.adaptive_scaler = None
        self.monitoring_started = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nReceived shutdown signal. Cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            if self.monitoring_started:
                stop_global_monitoring()
            
            if self.load_balancer:
                self.load_balancer.shutdown()
            
            # Clear caches
            clear_all_caches()
            
            if self.logger:
                self.logger.info("CLI shutdown complete")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="spin-glass-rl",
            description="Advanced Spin-Glass Reinforcement Learning Optimization Framework",
            epilog="For more help: spin-glass-rl <command> --help"
        )
        
        # Global options
        parser.add_argument("--log-level", default="INFO", 
                          choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
                          help="Set logging level")
        parser.add_argument("--log-dir", help="Directory for log files")
        parser.add_argument("--config", help="Configuration file path")
        parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
        parser.add_argument("--workers", type=int, help="Number of worker threads")
        parser.add_argument("--monitoring", action="store_true", help="Enable performance monitoring")
        parser.add_argument("--cache-dir", help="Cache directory path")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Optimization commands
        self._add_optimization_commands(subparsers)
        
        # Problem-specific commands
        self._add_problem_commands(subparsers)
        
        # System commands
        self._add_system_commands(subparsers)
        
        # Benchmarking commands
        self._add_benchmark_commands(subparsers)
        
        return parser
    
    def _add_optimization_commands(self, subparsers):
        """Add optimization-related commands."""
        # Optimize command
        opt_parser = subparsers.add_parser("optimize", help="Run optimization")
        opt_parser.add_argument("--problem", required=True, 
                               choices=["ising", "tsp", "scheduling", "custom"],
                               help="Problem type to optimize")
        opt_parser.add_argument("--size", type=int, default=100, help="Problem size")
        opt_parser.add_argument("--sweeps", type=int, default=1000, help="Number of sweeps")
        opt_parser.add_argument("--temp-initial", type=float, default=10.0, help="Initial temperature")
        opt_parser.add_argument("--temp-final", type=float, default=0.01, help="Final temperature")
        opt_parser.add_argument("--schedule", default="geometric", 
                               choices=["linear", "geometric", "exponential", "adaptive"],
                               help="Temperature schedule")
        opt_parser.add_argument("--output", help="Output file for results")
        opt_parser.add_argument("--adaptive-scaling", action="store_true", 
                               help="Enable adaptive parameter scaling")
        opt_parser.add_argument("--distributed", action="store_true", 
                               help="Enable distributed processing")
        
        # Parallel tempering command
        pt_parser = subparsers.add_parser("parallel-tempering", 
                                         help="Run parallel tempering optimization")
        pt_parser.add_argument("--replicas", type=int, default=8, help="Number of replicas")
        pt_parser.add_argument("--temp-min", type=float, default=0.1, help="Minimum temperature")
        pt_parser.add_argument("--temp-max", type=float, default=10.0, help="Maximum temperature")
        pt_parser.add_argument("--exchange-interval", type=int, default=10, 
                              help="Exchange attempt interval")
        pt_parser.add_argument("--problem", required=True, help="Problem configuration file")
        pt_parser.add_argument("--output", help="Output file for results")
    
    def _add_problem_commands(self, subparsers):
        """Add problem-specific commands."""
        # TSP command
        tsp_parser = subparsers.add_parser("tsp", help="Solve Traveling Salesman Problem")
        tsp_parser.add_argument("--cities", type=int, default=10, help="Number of cities")
        tsp_parser.add_argument("--coordinates", help="City coordinates file")
        tsp_parser.add_argument("--distance-matrix", help="Distance matrix file")
        tsp_parser.add_argument("--output", help="Output file for solution")
        tsp_parser.add_argument("--plot", action="store_true", help="Generate solution plot")
        
        # Scheduling command
        sched_parser = subparsers.add_parser("schedule", help="Solve scheduling problem")
        sched_parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
        sched_parser.add_argument("--agents", type=int, default=3, help="Number of agents")
        sched_parser.add_argument("--time-horizon", type=float, default=100.0, help="Time horizon")
        sched_parser.add_argument("--objective", default="makespan",
                                 choices=["makespan", "total_time", "weighted_completion"],
                                 help="Optimization objective")
        sched_parser.add_argument("--constraints", help="Constraints configuration file")
        sched_parser.add_argument("--output", help="Output file for schedule")
        
        # Custom problem command
        custom_parser = subparsers.add_parser("custom", help="Solve custom problem")
        custom_parser.add_argument("--problem-file", required=True, 
                                  help="Custom problem definition file")
        custom_parser.add_argument("--parameters", help="Problem parameters file")
        custom_parser.add_argument("--output", help="Output file for results")
    
    def _add_system_commands(self, subparsers):
        """Add system management commands."""
        # Health check command
        health_parser = subparsers.add_parser("health", help="System health check")
        health_parser.add_argument("--detailed", action="store_true", 
                                  help="Detailed health report")
        health_parser.add_argument("--output", help="Output file for health report")
        
        # Cache management commands
        cache_parser = subparsers.add_parser("cache", help="Cache management")
        cache_subparsers = cache_parser.add_subparsers(dest="cache_command")
        
        cache_subparsers.add_parser("clear", help="Clear all caches")
        cache_subparsers.add_parser("stats", help="Show cache statistics")
        cache_subparsers.add_parser("optimize", help="Optimize cache configuration")
        
        # Configuration commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_command")
        
        config_subparsers.add_parser("show", help="Show current configuration")
        config_subparsers.add_parser("validate", help="Validate configuration")
        
        validate_parser = config_subparsers.add_parser("set", help="Set configuration value")
        validate_parser.add_argument("key", help="Configuration key")
        validate_parser.add_argument("value", help="Configuration value")
    
    def _add_benchmark_commands(self, subparsers):
        """Add benchmarking commands."""
        bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
        bench_parser.add_argument("--suite", default="standard",
                                 choices=["standard", "performance", "scaling", "gpu"],
                                 help="Benchmark suite to run")
        bench_parser.add_argument("--problems", nargs="+", 
                                 choices=["tsp", "scheduling", "ising"],
                                 help="Specific problems to benchmark")
        bench_parser.add_argument("--sizes", nargs="+", type=int,
                                 default=[10, 50, 100, 500],
                                 help="Problem sizes to test")
        bench_parser.add_argument("--trials", type=int, default=5, 
                                 help="Number of trials per test")
        bench_parser.add_argument("--output", help="Benchmark results file")
        bench_parser.add_argument("--compare", help="Compare with previous results")
    
    def setup_logging_and_monitoring(self, args) -> None:
        """Setup logging and monitoring based on arguments."""
        # Setup logging
        self.logger = setup_global_logging(
            level=args.log_level,
            log_dir=args.log_dir,
            structured=True
        )
        
        # Start monitoring if requested
        if args.monitoring:
            start_global_monitoring()
            self.monitoring_started = True
            self.logger.info("Performance monitoring enabled")
        
        # Log command execution
        audit_log(
            action="cli_command_start",
            details={
                "command": args.command,
                "arguments": vars(args)
            },
            severity="INFO"
        )
    
    def setup_scaling_and_balancing(self, args) -> None:
        """Setup adaptive scaling and load balancing."""
        if hasattr(args, 'adaptive_scaling') and args.adaptive_scaling:
            policy = ScalingPolicy()
            self.adaptive_scaler = AdaptiveScaler(policy=policy)
            self.logger.info("Adaptive scaling enabled")
        
        if hasattr(args, 'distributed') and args.distributed:
            max_workers = args.workers if args.workers else None
            self.load_balancer = LoadBalancer(
                max_workers=max_workers,
                enable_gpu_workers=args.gpu
            )
            self.logger.info("Distributed processing enabled")
    
    def run_optimization(self, args) -> int:
        """Run optimization command."""
        with LoggingContext("optimization", self.logger) as ctx:
            try:
                if args.problem == "tsp":
                    return self._run_tsp_optimization(args)
                elif args.problem == "scheduling":
                    return self._run_scheduling_optimization(args)
                elif args.problem == "ising":
                    return self._run_ising_optimization(args)
                elif args.problem == "custom":
                    return self._run_custom_optimization(args)
                else:
                    self.logger.error(f"Unknown problem type: {args.problem}")
                    return 1
                
            except Exception as e:
                self.logger.error("Optimization failed", exception=e)
                return 1
    
    def _run_tsp_optimization(self, args) -> int:
        """Run TSP optimization."""
        from spin_glass_rl.problems.routing import TSPProblem
        from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
        from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
        
        # Create TSP problem
        tsp = TSPProblem()
        
        # Generate or load problem instance
        if hasattr(args, 'coordinates') and args.coordinates:
            # Load from coordinates file
            self.logger.info(f"Loading coordinates from {args.coordinates}")
            # Implementation for loading coordinates
        else:
            # Generate random instance
            instance_params = tsp.generate_random_instance(
                n_locations=args.size,
                area_size=100.0
            )
            self.logger.info(f"Generated random TSP with {args.size} cities")
        
        # Encode as Ising model
        ising_model = tsp.encode_to_ising()
        self.logger.info(f"Encoded as Ising model with {ising_model.n_spins} spins")
        
        # Configure annealer
        schedule_type = getattr(ScheduleType, args.schedule.upper())
        annealer_config = GPUAnnealerConfig(
            n_sweeps=args.sweeps,
            initial_temp=args.temp_initial,
            final_temp=args.temp_final,
            schedule_type=schedule_type,
            device="cuda" if args.gpu else "cpu"
        )
        
        annealer = GPUAnnealer(annealer_config)
        
        # Solve with adaptive scaling if enabled
        if self.adaptive_scaler:
            # Implement adaptive optimization loop
            self.logger.info("Running with adaptive scaling")
            # Implementation here
        
        # Standard solve
        solution = tsp.solve_with_annealer(annealer)
        
        # Output results
        self.logger.success(f"TSP solved: distance={solution.objective_value:.2f}, feasible={solution.is_feasible}")
        
        if args.output:
            self._save_results(solution, args.output)
        
        return 0 if solution.is_feasible else 1
    
    def _run_scheduling_optimization(self, args) -> int:
        """Run scheduling optimization."""
        # Implementation similar to TSP
        self.logger.info("Scheduling optimization not yet implemented")
        return 1
    
    def _run_ising_optimization(self, args) -> int:
        """Run basic Ising model optimization."""
        # Implementation for basic Ising optimization
        self.logger.info("Basic Ising optimization not yet implemented")
        return 1
    
    def _run_custom_optimization(self, args) -> int:
        """Run custom problem optimization."""
        # Implementation for custom problems
        self.logger.info("Custom optimization not yet implemented")
        return 1
    
    def run_health_check(self, args) -> int:
        """Run system health check."""
        with LoggingContext("health_check", self.logger):
            try:
                self.logger.info("Running system health check...")
                
                health_report = get_system_health_report()
                
                # Display results
                if args.detailed:
                    self._display_detailed_health_report(health_report)
                else:
                    self._display_basic_health_report(health_report)
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(health_report, f, indent=2, default=str)
                    self.logger.info(f"Health report saved to {args.output}")
                
                return 0
                
            except Exception as e:
                self.logger.error("Health check failed", exception=e)
                return 1
    
    def _display_basic_health_report(self, report: Dict[str, Any]) -> None:
        """Display basic health report."""
        print("\n=== System Health Report ===")
        
        # System requirements
        req = report.get("system_requirements", {})
        print(f"Python: {'✓' if req.get('python_version', {}).get('supported', False) else '✗'}")
        print(f"PyTorch: {'✓' if req.get('pytorch', {}).get('available', False) else '✗'}")
        print(f"Memory: {'✓' if req.get('memory', {}).get('sufficient', False) else '✗'}")
        print(f"Disk: {'✓' if req.get('disk_space', {}).get('sufficient', False) else '✗'}")
        
        # GPU health
        gpu = report.get("gpu_health", {})
        if gpu.get("available", False):
            print(f"GPU: ✓ ({gpu.get('device_count', 0)} devices)")
        else:
            print("GPU: ✗ (Not available)")
        
        print()
    
    def _display_detailed_health_report(self, report: Dict[str, Any]) -> None:
        """Display detailed health report."""
        print("\n=== Detailed System Health Report ===")
        print(json.dumps(report, indent=2, default=str))
    
    def run_cache_command(self, args) -> int:
        """Run cache management commands."""
        if args.cache_command == "clear":
            clear_all_caches()
            self.logger.info("All caches cleared")
            print("✓ All caches cleared")
            
        elif args.cache_command == "stats":
            stats = get_cache_performance_summary()
            print("\n=== Cache Performance Summary ===")
            print(json.dumps(stats, indent=2, default=str))
            
        elif args.cache_command == "optimize":
            # Implement cache optimization
            self.logger.info("Cache optimization not yet implemented")
            print("Cache optimization not yet implemented")
            
        return 0
    
    def run_benchmark(self, args) -> int:
        """Run benchmarks."""
        with LoggingContext("benchmark", self.logger):
            try:
                self.logger.info(f"Running {args.suite} benchmark suite")
                
                # Implementation for benchmarking
                self.logger.info("Benchmarking not yet fully implemented")
                
                return 0
                
            except Exception as e:
                self.logger.error("Benchmark failed", exception=e)
                return 1
    
    def _save_results(self, results: Any, filepath: str) -> None:
        """Save results to file."""
        try:
            if filepath.endswith('.json'):
                # Save as JSON
                if hasattr(results, 'to_dict'):
                    data = results.to_dict()
                elif hasattr(results, '__dict__'):
                    data = results.__dict__
                else:
                    data = {'results': str(results)}
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            else:
                # Save as text
                with open(filepath, 'w') as f:
                    f.write(str(results))
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {filepath}", exception=e)
    
    def run(self, args=None) -> int:
        """Main entry point."""
        if args is None:
            args = sys.argv[1:]
        
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            # Setup logging and monitoring
            self.setup_logging_and_monitoring(parsed_args)
            
            # Setup scaling and balancing
            self.setup_scaling_and_balancing(parsed_args)
            
            # Route to appropriate command handler
            if parsed_args.command == "optimize":
                return self.run_optimization(parsed_args)
            elif parsed_args.command == "health":
                return self.run_health_check(parsed_args)
            elif parsed_args.command == "cache":
                return self.run_cache_command(parsed_args)
            elif parsed_args.command == "benchmark":
                return self.run_benchmark(parsed_args)
            elif parsed_args.command in ["tsp", "schedule", "custom"]:
                # These are handled by optimize command
                parsed_args.problem = parsed_args.command
                return self.run_optimization(parsed_args)
            else:
                parser.print_help()
                return 1
        
        finally:
            self._cleanup()


def main() -> int:
    """Main entry point for the CLI."""
    cli = AdvancedCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())