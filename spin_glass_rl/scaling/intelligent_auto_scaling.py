"""
Intelligent Auto-Scaling System for Spin-Glass Optimization.

Implements adaptive resource allocation, predictive scaling, and 
high-performance computing orchestration for massive problem instances.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import threading
import queue
import time
import json
from enum import Enum
from collections import deque, defaultdict
import concurrent.futures
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    MACHINE_LEARNING = "ml"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class WorkloadPattern(Enum):
    """Workload patterns for predictive scaling."""
    STEADY_STATE = "steady"
    BURST = "burst"
    PERIODIC = "periodic"
    RANDOM = "random"


@dataclass
class ResourceRequirements:
    """Resource requirements specification."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 1.0
    network_bandwidth_mbps: float = 100.0
    priority: int = 1  # 1=low, 5=critical


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system."""
    strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    max_resources: ResourceRequirements = field(default_factory=lambda: ResourceRequirements(
        cpu_cores=64, memory_gb=128.0, gpu_count=8, gpu_memory_gb=64.0
    ))
    min_resources: ResourceRequirements = field(default_factory=lambda: ResourceRequirements(
        cpu_cores=1, memory_gb=1.0, gpu_count=0, gpu_memory_gb=0.0
    ))
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    prediction_window_minutes: int = 10
    cooldown_seconds: int = 300
    enable_preemption: bool = True


class WorkloadPredictor:
    """Predicts future workload for proactive scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.workload_history = deque(maxlen=1000)
        self.resource_history = deque(maxlen=1000)
        self.prediction_model = None
        self.last_prediction = None
        
    def record_workload(self, problem_size: int, execution_time: float, 
                       resources_used: ResourceRequirements):
        """Record workload observation for learning."""
        observation = {
            'timestamp': time.time(),
            'problem_size': problem_size,
            'execution_time': execution_time,
            'resources_used': resources_used,
            'throughput': problem_size / execution_time if execution_time > 0 else 0.0
        }
        
        self.workload_history.append(observation)
        self.resource_history.append(resources_used)
        
        # Update prediction model periodically
        if len(self.workload_history) >= 10 and len(self.workload_history) % 10 == 0:
            self._update_prediction_model()
    
    def predict_resource_needs(self, incoming_problems: List[Dict[str, Any]]) -> ResourceRequirements:
        """Predict resource requirements for incoming problems."""
        if not incoming_problems:
            return self.config.min_resources
        
        # Simple heuristic prediction
        total_problem_complexity = sum(self._estimate_complexity(p) for p in incoming_problems)
        
        # Base resource estimation
        base_cpu = max(1, int(total_problem_complexity / 1000))
        base_memory = max(1.0, total_problem_complexity / 500)
        base_gpu = 1 if total_problem_complexity > 5000 else 0
        
        # Apply historical learning if available
        if self.prediction_model:
            scaling_factor = self.prediction_model.get('scaling_factor', 1.0)
            base_cpu = int(base_cpu * scaling_factor)
            base_memory *= scaling_factor
        
        # Clamp to configured limits
        predicted_resources = ResourceRequirements(
            cpu_cores=min(base_cpu, self.config.max_resources.cpu_cores),
            memory_gb=min(base_memory, self.config.max_resources.memory_gb),
            gpu_count=min(base_gpu, self.config.max_resources.gpu_count),
            gpu_memory_gb=min(base_gpu * 8.0, self.config.max_resources.gpu_memory_gb)
        )
        
        return predicted_resources
    
    def predict_workload_pattern(self, time_horizon_minutes: int = 30) -> WorkloadPattern:
        """Predict workload pattern for given time horizon."""
        if len(self.workload_history) < 20:
            return WorkloadPattern.STEADY_STATE
        
        # Analyze recent workload trends
        recent_observations = list(self.workload_history)[-20:]
        timestamps = [obs['timestamp'] for obs in recent_observations]
        throughputs = [obs['throughput'] for obs in recent_observations]
        
        # Simple pattern recognition
        throughput_variance = np.var(throughputs) if throughputs else 0
        time_diffs = np.diff(timestamps) if len(timestamps) > 1 else [1]
        avg_interval = np.mean(time_diffs)
        
        if throughput_variance < 0.1:
            return WorkloadPattern.STEADY_STATE
        elif throughput_variance > 1.0:
            return WorkloadPattern.BURST
        elif avg_interval < 60:  # Less than 1 minute intervals
            return WorkloadPattern.PERIODIC
        else:
            return WorkloadPattern.RANDOM
    
    def _estimate_complexity(self, problem: Dict[str, Any]) -> float:
        """Estimate computational complexity of problem."""
        # Extract problem characteristics
        n_spins = problem.get('n_spins', 10)
        coupling_density = problem.get('coupling_density', 0.5)
        
        # Complexity heuristic (typically O(n^2) for spin glass)
        base_complexity = n_spins ** 2
        density_factor = 1.0 + coupling_density
        
        return base_complexity * density_factor
    
    def _update_prediction_model(self):
        """Update internal prediction model based on historical data."""
        if len(self.workload_history) < 10:
            return
        
        observations = list(self.workload_history)[-50:]  # Recent observations
        
        # Calculate performance metrics
        problem_sizes = [obs['problem_size'] for obs in observations]
        execution_times = [obs['execution_time'] for obs in observations]
        cpu_usage = [obs['resources_used'].cpu_cores for obs in observations]
        
        # Simple linear regression for scaling factor
        if len(problem_sizes) > 5:
            # Correlation between problem size and required resources
            size_cpu_correlation = np.corrcoef(problem_sizes, cpu_usage)[0, 1] if len(set(cpu_usage)) > 1 else 1.0
            
            # Adjust scaling factor based on recent performance
            recent_efficiency = np.mean([size / time for size, time in zip(problem_sizes, execution_times)])
            baseline_efficiency = 1000.0  # Problems per second baseline
            
            efficiency_ratio = recent_efficiency / baseline_efficiency
            scaling_factor = 1.0 / max(0.1, efficiency_ratio)  # Inverse relationship
            
            self.prediction_model = {
                'scaling_factor': scaling_factor,
                'size_cpu_correlation': size_cpu_correlation,
                'recent_efficiency': recent_efficiency,
                'update_time': time.time()
            }


class ResourceAllocator:
    """Allocates and manages computational resources."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.allocated_resources = ResourceRequirements()
        self.resource_pools = self._initialize_resource_pools()
        self.allocation_history = deque(maxlen=100)
        self.resource_lock = threading.Lock()
        
    def _initialize_resource_pools(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Initialize available resource pools."""
        # Detect actual system resources
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            cpu_count = mp.cpu_count()
            memory_gb = 8.0  # Default
        
        # GPU detection
        gpu_count = 0
        gpu_memory_gb = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory_gb = sum(gpu.memoryTotal / 1024 for gpu in gpus)  # Convert MB to GB
        except ImportError:
            pass
        
        return {
            ResourceType.CPU: {
                'total': cpu_count,
                'available': cpu_count,
                'allocated': 0
            },
            ResourceType.MEMORY: {
                'total': memory_gb,
                'available': memory_gb,
                'allocated': 0.0
            },
            ResourceType.GPU: {
                'total': gpu_count,
                'available': gpu_count,
                'allocated': 0
            }
        }
    
    def allocate_resources(self, requirements: ResourceRequirements, 
                         priority: int = 1) -> Optional[str]:
        """Allocate resources for optimization task."""
        with self.resource_lock:
            allocation_id = f"alloc_{time.time()}_{priority}"
            
            # Check availability
            if not self._can_allocate(requirements):
                if self.config.enable_preemption and priority >= 3:
                    # Try preemption for high-priority tasks
                    if self._attempt_preemption(requirements, priority):
                        return self._perform_allocation(allocation_id, requirements)
                return None
            
            return self._perform_allocation(allocation_id, requirements)
    
    def deallocate_resources(self, allocation_id: str):
        """Deallocate resources for completed task."""
        with self.resource_lock:
            # Find allocation in history
            allocation = None
            for alloc in self.allocation_history:
                if alloc['allocation_id'] == allocation_id:
                    allocation = alloc
                    break
            
            if allocation:
                requirements = allocation['requirements']
                
                # Return resources to pools
                self.resource_pools[ResourceType.CPU]['available'] += requirements.cpu_cores
                self.resource_pools[ResourceType.MEMORY]['available'] += requirements.memory_gb
                self.resource_pools[ResourceType.GPU]['available'] += requirements.gpu_count
                
                # Update allocated tracking
                self.allocated_resources.cpu_cores -= requirements.cpu_cores
                self.allocated_resources.memory_gb -= requirements.memory_gb
                self.allocated_resources.gpu_count -= requirements.gpu_count
                
                # Mark as deallocated
                allocation['deallocated'] = True
                allocation['deallocation_time'] = time.time()
                
                logger.info(f"Deallocated resources for {allocation_id}")
    
    def _can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if required resources are available."""
        cpu_available = self.resource_pools[ResourceType.CPU]['available'] >= requirements.cpu_cores
        memory_available = self.resource_pools[ResourceType.MEMORY]['available'] >= requirements.memory_gb
        gpu_available = self.resource_pools[ResourceType.GPU]['available'] >= requirements.gpu_count
        
        return cpu_available and memory_available and gpu_available
    
    def _attempt_preemption(self, requirements: ResourceRequirements, priority: int) -> bool:
        """Attempt to preempt lower-priority allocations."""
        # Find lower-priority allocations that could be preempted
        preemptable_allocations = [
            alloc for alloc in self.allocation_history
            if alloc.get('priority', 1) < priority and not alloc.get('deallocated', False)
        ]
        
        # Sort by priority (lowest first)
        preemptable_allocations.sort(key=lambda x: x.get('priority', 1))
        
        # Try to free enough resources
        cpu_needed = requirements.cpu_cores - self.resource_pools[ResourceType.CPU]['available']
        memory_needed = requirements.memory_gb - self.resource_pools[ResourceType.MEMORY]['available']
        gpu_needed = requirements.gpu_count - self.resource_pools[ResourceType.GPU]['available']
        
        cpu_freed = 0
        memory_freed = 0.0
        gpu_freed = 0
        
        for alloc in preemptable_allocations:
            if cpu_freed >= cpu_needed and memory_freed >= memory_needed and gpu_freed >= gpu_needed:
                break
            
            # Preempt this allocation
            req = alloc['requirements']
            cpu_freed += req.cpu_cores
            memory_freed += req.memory_gb
            gpu_freed += req.gpu_count
            
            # Mark as preempted
            alloc['preempted'] = True
            alloc['preemption_time'] = time.time()
            
            # Return resources to pool
            self.resource_pools[ResourceType.CPU]['available'] += req.cpu_cores
            self.resource_pools[ResourceType.MEMORY]['available'] += req.memory_gb
            self.resource_pools[ResourceType.GPU]['available'] += req.gpu_count
            
            logger.info(f"Preempted allocation {alloc['allocation_id']} for higher priority task")
        
        return cpu_freed >= cpu_needed and memory_freed >= memory_needed and gpu_freed >= gpu_needed
    
    def _perform_allocation(self, allocation_id: str, requirements: ResourceRequirements) -> str:
        """Perform the actual resource allocation."""
        # Allocate resources
        self.resource_pools[ResourceType.CPU]['available'] -= requirements.cpu_cores
        self.resource_pools[ResourceType.MEMORY]['available'] -= requirements.memory_gb
        self.resource_pools[ResourceType.GPU]['available'] -= requirements.gpu_count
        
        self.resource_pools[ResourceType.CPU]['allocated'] += requirements.cpu_cores
        self.resource_pools[ResourceType.MEMORY]['allocated'] += requirements.memory_gb
        self.resource_pools[ResourceType.GPU]['allocated'] += requirements.gpu_count
        
        # Update total allocated tracking
        self.allocated_resources.cpu_cores += requirements.cpu_cores
        self.allocated_resources.memory_gb += requirements.memory_gb
        self.allocated_resources.gpu_count += requirements.gpu_count
        
        # Record allocation
        allocation_record = {
            'allocation_id': allocation_id,
            'requirements': requirements,
            'priority': requirements.priority,
            'allocation_time': time.time(),
            'deallocated': False
        }
        self.allocation_history.append(allocation_record)
        
        logger.info(f"Allocated resources: {allocation_id}")
        return allocation_id
    
    def get_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization."""
        utilization = {}
        
        for resource_type, pool in self.resource_pools.items():
            total = pool['total']
            allocated = pool['allocated']
            if total > 0:
                utilization[resource_type] = allocated / total
            else:
                utilization[resource_type] = 0.0
        
        return utilization


class AutoScalingController:
    """Main controller for intelligent auto-scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.workload_predictor = WorkloadPredictor(config)
        self.resource_allocator = ResourceAllocator(config)
        
        self.scaling_history = deque(maxlen=100)
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        
        self.is_running = False
        self.controller_thread = None
        self.last_scaling_action = 0
        
    def start_controller(self):
        """Start the auto-scaling controller."""
        if self.is_running:
            logger.warning("Auto-scaling controller already running")
            return
        
        self.is_running = True
        self.controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
        self.controller_thread.start()
        logger.info("Auto-scaling controller started")
    
    def stop_controller(self):
        """Stop the auto-scaling controller."""
        self.is_running = False
        if self.controller_thread:
            self.controller_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("Auto-scaling controller stopped")
    
    def submit_optimization_task(self, optimization_func: Callable, 
                               problem: Dict[str, Any], priority: int = 1) -> str:
        """Submit optimization task for auto-scaled execution."""
        task_id = f"task_{time.time()}_{priority}"
        
        # Estimate resource requirements
        predicted_resources = self.workload_predictor.predict_resource_needs([problem])
        predicted_resources.priority = priority
        
        # Create task
        task = {
            'task_id': task_id,
            'optimization_func': optimization_func,
            'problem': problem,
            'requirements': predicted_resources,
            'priority': priority,
            'submission_time': time.time()
        }
        
        # Add to priority queue (lower priority value = higher priority)
        self.task_queue.put((priority, task))
        
        logger.info(f"Submitted task {task_id} with priority {priority}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of submitted task."""
        if task_id in self.active_tasks:
            return {
                'status': 'running',
                'task': self.active_tasks[task_id],
                'start_time': self.active_tasks[task_id].get('start_time')
            }
        
        # Check if completed (would need more sophisticated tracking)
        return {'status': 'unknown'}
    
    def _controller_loop(self):
        """Main auto-scaling controller loop."""
        while self.is_running:
            try:
                # Check for scaling decisions
                scaling_decision = self._make_scaling_decision()
                
                if scaling_decision['action'] != 'none':
                    self._execute_scaling_action(scaling_decision)
                
                # Process pending tasks
                self._process_task_queue()
                
                # Cleanup completed tasks
                self._cleanup_completed_tasks()
                
                time.sleep(1.0)  # Controller cycle interval
                
            except Exception as e:
                logger.error(f"Auto-scaling controller error: {e}")
                time.sleep(5.0)
    
    def _make_scaling_decision(self) -> Dict[str, Any]:
        """Make intelligent scaling decision."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.config.cooldown_seconds:
            return {'action': 'none', 'reason': 'cooldown_period'}
        
        # Get current utilization
        utilization = self.resource_allocator.get_utilization()
        
        # Calculate average utilization
        avg_utilization = np.mean(list(utilization.values()))
        
        # Predictive scaling
        if self.config.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.ADAPTIVE]:
            workload_pattern = self.workload_predictor.predict_workload_pattern()
            
            if workload_pattern == WorkloadPattern.BURST and avg_utilization > 0.5:
                return {
                    'action': 'scale_up',
                    'reason': 'predicted_burst',
                    'factor': 1.5,
                    'utilization': avg_utilization
                }
        
        # Reactive scaling
        if avg_utilization > self.config.scale_up_threshold:
            return {
                'action': 'scale_up',
                'reason': 'high_utilization',
                'factor': 1.3,
                'utilization': avg_utilization
            }
        elif avg_utilization < self.config.scale_down_threshold:
            return {
                'action': 'scale_down',
                'reason': 'low_utilization',
                'factor': 0.8,
                'utilization': avg_utilization
            }
        
        return {'action': 'none', 'reason': 'target_utilization', 'utilization': avg_utilization}
    
    def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute scaling action."""
        action = decision['action']
        factor = decision.get('factor', 1.0)
        
        if action == 'scale_up':
            # Increase resource limits (simulation)
            new_max_cpu = min(
                int(self.config.max_resources.cpu_cores * factor),
                128  # Hard limit
            )
            new_max_memory = min(
                self.config.max_resources.memory_gb * factor,
                256.0  # Hard limit
            )
            
            # Update configuration
            self.config.max_resources.cpu_cores = new_max_cpu
            self.config.max_resources.memory_gb = new_max_memory
            
            logger.info(f"Scaled up: CPU={new_max_cpu}, Memory={new_max_memory:.1f}GB")
            
        elif action == 'scale_down':
            # Decrease resource limits (simulation)
            new_max_cpu = max(
                int(self.config.max_resources.cpu_cores * factor),
                self.config.min_resources.cpu_cores
            )
            new_max_memory = max(
                self.config.max_resources.memory_gb * factor,
                self.config.min_resources.memory_gb
            )
            
            # Update configuration
            self.config.max_resources.cpu_cores = new_max_cpu
            self.config.max_resources.memory_gb = new_max_memory
            
            logger.info(f"Scaled down: CPU={new_max_cpu}, Memory={new_max_memory:.1f}GB")
        
        # Record scaling action
        scaling_record = {
            'timestamp': time.time(),
            'action': action,
            'decision': decision,
            'new_limits': {
                'cpu': self.config.max_resources.cpu_cores,
                'memory': self.config.max_resources.memory_gb
            }
        }
        self.scaling_history.append(scaling_record)
        self.last_scaling_action = time.time()
    
    def _process_task_queue(self):
        """Process pending tasks from queue."""
        max_concurrent = 5  # Limit concurrent tasks
        
        while (not self.task_queue.empty() and 
               len(self.active_tasks) < max_concurrent):
            
            try:
                priority, task = self.task_queue.get_nowait()
                
                # Try to allocate resources
                allocation_id = self.resource_allocator.allocate_resources(
                    task['requirements'], priority
                )
                
                if allocation_id:
                    # Start task execution
                    task['allocation_id'] = allocation_id
                    task['start_time'] = time.time()
                    
                    # Submit to executor
                    future = self.executor.submit(self._execute_task, task)
                    task['future'] = future
                    
                    self.active_tasks[task['task_id']] = task
                    logger.info(f"Started task {task['task_id']}")
                else:
                    # Resource allocation failed, put back in queue
                    self.task_queue.put((priority, task))
                    break  # Wait for resources to become available
                    
            except queue.Empty:
                break
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization task."""
        task_id = task['task_id']
        
        try:
            # Execute optimization
            start_time = time.time()
            result = task['optimization_func'](task['problem'])
            execution_time = time.time() - start_time
            
            # Record workload for learning
            problem_size = task['problem'].get('n_spins', 10)
            self.workload_predictor.record_workload(
                problem_size, execution_time, task['requirements']
            )
            
            # Task completed successfully
            task_result = {
                'task_id': task_id,
                'result': result,
                'execution_time': execution_time,
                'success': True,
                'completion_time': time.time()
            }
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            return task_result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return {
                'task_id': task_id,
                'error': str(e),
                'success': False,
                'completion_time': time.time()
            }
        finally:
            # Always deallocate resources
            if 'allocation_id' in task:
                self.resource_allocator.deallocate_resources(task['allocation_id'])
    
    def _cleanup_completed_tasks(self):
        """Clean up completed tasks."""
        completed_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if 'future' in task and task['future'].done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling report."""
        utilization = self.resource_allocator.get_utilization()
        
        # Calculate efficiency metrics
        total_allocations = len(self.resource_allocator.allocation_history)
        preempted_count = sum(1 for alloc in self.resource_allocator.allocation_history 
                            if alloc.get('preempted', False))
        
        preemption_rate = preempted_count / total_allocations if total_allocations > 0 else 0.0
        
        # Scaling actions summary
        scaling_actions = list(self.scaling_history)
        scale_up_count = sum(1 for action in scaling_actions if action['action'] == 'scale_up')
        scale_down_count = sum(1 for action in scaling_actions if action['action'] == 'scale_down')
        
        return {
            'current_utilization': utilization,
            'average_utilization': np.mean(list(utilization.values())),
            'total_allocations': total_allocations,
            'preemption_rate': preemption_rate,
            'active_tasks': len(self.active_tasks),
            'pending_tasks': self.task_queue.qsize(),
            'scaling_actions': {
                'scale_up': scale_up_count,
                'scale_down': scale_down_count,
                'total': len(scaling_actions)
            },
            'resource_limits': {
                'cpu': self.config.max_resources.cpu_cores,
                'memory': self.config.max_resources.memory_gb,
                'gpu': self.config.max_resources.gpu_count
            },
            'workload_prediction': {
                'model_available': self.workload_predictor.prediction_model is not None,
                'history_length': len(self.workload_predictor.workload_history)
            }
        }


# Demonstration and testing functions
def create_auto_scaling_demo():
    """Create demonstration of intelligent auto-scaling system."""
    print("Creating Intelligent Auto-Scaling System Demo...")
    
    # Configuration
    config = ScalingConfig(
        strategy=ScalingStrategy.ADAPTIVE,
        max_resources=ResourceRequirements(
            cpu_cores=16, memory_gb=32.0, gpu_count=2
        ),
        min_resources=ResourceRequirements(
            cpu_cores=2, memory_gb=4.0, gpu_count=0
        ),
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        cooldown_seconds=10  # Short for demo
    )
    
    # Create auto-scaling controller
    controller = AutoScalingController(config)
    controller.start_controller()
    
    # Mock optimization function with variable complexity
    def mock_optimization(problem: Dict[str, Any]) -> Dict[str, Any]:
        """Mock optimization function."""
        n_spins = problem.get('n_spins', 10)
        complexity = problem.get('complexity', 1.0)
        
        # Simulate computation time based on problem size
        computation_time = (n_spins / 100.0) * complexity
        time.sleep(min(computation_time, 2.0))  # Cap at 2 seconds for demo
        
        energy = np.random.uniform(-100, -10)
        
        return {
            'best_energy': energy,
            'best_configuration': torch.randint(0, 2, (n_spins,)) * 2 - 1,
            'energy_history': [0, energy/2, energy]
        }
    
    print("\nSubmitting optimization tasks with varying priorities and sizes...")
    
    # Submit tasks with different characteristics
    task_configs = [
        # Normal tasks
        {'n_spins': 20, 'complexity': 1.0, 'priority': 2},
        {'n_spins': 30, 'complexity': 1.5, 'priority': 1},
        {'n_spins': 25, 'complexity': 1.2, 'priority': 3},
        
        # High-priority burst
        {'n_spins': 50, 'complexity': 2.0, 'priority': 4},
        {'n_spins': 60, 'complexity': 2.5, 'priority': 5},
        {'n_spins': 40, 'complexity': 1.8, 'priority': 4},
        
        # More normal tasks
        {'n_spins': 15, 'complexity': 0.8, 'priority': 2},
        {'n_spins': 35, 'complexity': 1.3, 'priority': 1},
    ]
    
    submitted_tasks = []
    
    for i, task_config in enumerate(task_configs):
        task_id = controller.submit_optimization_task(
            mock_optimization, task_config, task_config['priority']
        )
        submitted_tasks.append(task_id)
        print(f"  Submitted task {i+1}: {task_config['n_spins']} spins, priority {task_config['priority']}")
        
        # Add some delay between submissions
        if i == 2:  # After first 3 tasks, create burst
            print("  Creating burst workload...")
        time.sleep(0.5)
    
    # Monitor scaling behavior
    print("\nMonitoring auto-scaling behavior...")
    
    start_time = time.time()
    reports = []
    
    while time.time() - start_time < 20:  # Monitor for 20 seconds
        time.sleep(2)
        
        report = controller.get_scaling_report()
        reports.append({
            'timestamp': time.time(),
            'report': report
        })
        
        print(f"  Active: {report['active_tasks']}, Pending: {report['pending_tasks']}, "
              f"Avg Util: {report['average_utilization']:.1%}")
    
    # Stop controller
    controller.stop_controller()
    
    # Wait for remaining tasks to complete
    print("\nWaiting for tasks to complete...")
    time.sleep(5)
    
    # Final report
    final_report = controller.get_scaling_report()
    
    # Display results
    print("\n" + "="*60)
    print("INTELLIGENT AUTO-SCALING SYSTEM RESULTS")
    print("="*60)
    
    print(f"\nTask Execution Summary:")
    print(f"  Total allocations: {final_report['total_allocations']}")
    print(f"  Preemption rate: {final_report['preemption_rate']:.1%}")
    print(f"  Final utilization: {final_report['average_utilization']:.1%}")
    
    print(f"\nScaling Actions:")
    scaling_actions = final_report['scaling_actions']
    print(f"  Scale-up events: {scaling_actions['scale_up']}")
    print(f"  Scale-down events: {scaling_actions['scale_down']}")
    print(f"  Total scaling actions: {scaling_actions['total']}")
    
    print(f"\nResource Limits (Final):")
    limits = final_report['resource_limits']
    print(f"  CPU cores: {limits['cpu']}")
    print(f"  Memory: {limits['memory']:.1f} GB")
    print(f"  GPUs: {limits['gpu']}")
    
    print(f"\nWorkload Prediction:")
    prediction = final_report['workload_prediction']
    print(f"  Model trained: {prediction['model_available']}")
    print(f"  History length: {prediction['history_length']}")
    
    # Utilization trend analysis
    if reports:
        utilizations = [r['report']['average_utilization'] for r in reports]
        print(f"\nUtilization Trends:")
        print(f"  Peak utilization: {max(utilizations):.1%}")
        print(f"  Min utilization: {min(utilizations):.1%}")
        print(f"  Utilization variance: {np.var(utilizations):.3f}")
    
    return controller, final_report, reports


def benchmark_scaling_efficiency():
    """Benchmark scaling efficiency under different workload patterns."""
    print("Benchmarking Auto-Scaling Efficiency...")
    
    workload_patterns = [
        ('steady', [10, 10, 10, 10, 10]),
        ('burst', [5, 5, 50, 50, 50]),
        ('ramp', [5, 15, 25, 35, 45]),
        ('spike', [10, 10, 80, 10, 10])
    ]
    
    benchmark_results = {}
    
    for pattern_name, problem_sizes in workload_patterns:
        print(f"\nTesting {pattern_name} workload pattern...")
        
        # Configure for fast benchmarking
        config = ScalingConfig(
            strategy=ScalingStrategy.ADAPTIVE,
            max_resources=ResourceRequirements(cpu_cores=8, memory_gb=16.0),
            cooldown_seconds=2  # Fast scaling for benchmark
        )
        
        controller = AutoScalingController(config)
        controller.start_controller()
        
        def fast_optimization(problem):
            time.sleep(0.1)  # Fixed short time
            return {'best_energy': -50, 'energy_history': [0, -25, -50]}
        
        # Submit tasks
        start_time = time.time()
        task_ids = []
        
        for size in problem_sizes:
            task_id = controller.submit_optimization_task(
                fast_optimization, {'n_spins': size}, priority=1
            )
            task_ids.append(task_id)
            time.sleep(0.2)  # Small delay between submissions
        
        # Monitor completion
        all_completed = False
        timeout = 15  # 15 seconds timeout
        
        while not all_completed and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            
            report = controller.get_scaling_report()
            if report['active_tasks'] == 0 and report['pending_tasks'] == 0:
                all_completed = True
        
        execution_time = time.time() - start_time
        final_report = controller.get_scaling_report()
        controller.stop_controller()
        
        # Calculate efficiency metrics
        total_work = sum(problem_sizes)
        throughput = total_work / execution_time
        scaling_overhead = final_report['scaling_actions']['total']
        
        benchmark_results[pattern_name] = {
            'execution_time': execution_time,
            'throughput': throughput,
            'total_work': total_work,
            'scaling_actions': scaling_overhead,
            'avg_utilization': final_report['average_utilization'],
            'preemption_rate': final_report['preemption_rate']
        }
        
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} work/sec")
        print(f"  Scaling actions: {scaling_overhead}")
    
    # Display benchmark summary
    print("\n" + "="*80)
    print("AUTO-SCALING EFFICIENCY BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Pattern':<10} {'Time(s)':<10} {'Throughput':<12} {'Scaling':<10} {'Util(%)':<8}")
    print("-" * 80)
    
    for pattern, results in benchmark_results.items():
        time_val = results['execution_time']
        throughput = results['throughput']
        scaling = results['scaling_actions']
        util = results['avg_utilization'] * 100
        
        print(f"{pattern:<10} {time_val:<10.2f} {throughput:<12.1f} {scaling:<10} {util:<8.1f}")
    
    # Find best performing pattern
    best_pattern = min(benchmark_results.keys(), 
                      key=lambda k: benchmark_results[k]['execution_time'])
    print(f"\nBest performing pattern: {best_pattern}")
    
    return benchmark_results


if __name__ == "__main__":
    # Run auto-scaling demonstrations
    print("Starting Intelligent Auto-Scaling System Demonstrations...\n")
    
    # Main auto-scaling demo
    controller, final_report, reports = create_auto_scaling_demo()
    
    print("\n" + "="*80)
    
    # Scaling efficiency benchmark
    benchmark_results = benchmark_scaling_efficiency()
    
    print("\nIntelligent auto-scaling system demonstration completed successfully!")