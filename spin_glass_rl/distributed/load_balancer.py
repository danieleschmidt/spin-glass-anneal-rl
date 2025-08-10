"""Intelligent load balancing and distributed processing system."""

import time
import threading
import queue
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import torch
import numpy as np

from spin_glass_rl.utils.robust_logging import get_logger, LoggingContext
from spin_glass_rl.utils.monitoring import PerformanceMonitor

logger = get_logger("load_balancer")


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    id: str
    node_type: str  # 'cpu', 'gpu', 'hybrid'
    max_capacity: int
    current_load: int = 0
    total_completed: int = 0
    total_failed: int = 0
    average_task_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = "idle"  # 'idle', 'busy', 'failed', 'maintenance'


@dataclass
class Task:
    """Represents a computational task."""
    id: str
    task_type: str
    priority: int  # Lower number = higher priority
    estimated_time: float
    required_memory_mb: float
    gpu_required: bool
    data: Any
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    worker_id: str
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    completed_at: float = field(default_factory=time.time)


class LoadBalancer:
    """Intelligent load balancer for distributing computational tasks."""
    
    def __init__(self, 
                 max_workers: int = None,
                 enable_gpu_workers: bool = True,
                 task_timeout: float = 300.0,
                 heartbeat_interval: float = 10.0):
        
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) * 2)
        self.enable_gpu_workers = enable_gpu_workers and torch.cuda.is_available()
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        
        # Task tracking
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: List[Task] = []
        
        # Performance metrics
        self.performance_metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_queue_time': 0.0,
            'worker_utilization': 0.0
        }
        
        # Threading
        self._shutdown = threading.Event()
        self._scheduler_thread = None
        self._heartbeat_thread = None
        self._result_processor_thread = None
        
        # Thread pools
        self._cpu_executor = ThreadPoolExecutor(max_workers=self.max_workers // 2)
        self._gpu_executor = ThreadPoolExecutor(max_workers=4) if self.enable_gpu_workers else None
        
        # Initialize workers
        self._initialize_workers()
        self._start_background_threads()
        
        logger.info(f"Load balancer initialized: {len(self.workers)} workers, GPU enabled: {self.enable_gpu_workers}")
    
    def _initialize_workers(self) -> None:
        """Initialize worker nodes."""
        cpu_count = multiprocessing.cpu_count()
        
        # CPU workers
        for i in range(cpu_count):
            worker = WorkerNode(
                id=f"cpu_{i}",
                node_type="cpu",
                max_capacity=2,  # Each CPU core can handle 2 tasks
                capabilities={'cpu_intensive': True, 'memory_intensive': True}
            )
            self.workers[worker.id] = worker
        
        # GPU workers (if available)
        if self.enable_gpu_workers:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                worker = WorkerNode(
                    id=f"gpu_{i}",
                    node_type="gpu",
                    max_capacity=4,  # GPUs can handle more parallel tasks
                    capabilities={
                        'gpu_compute': True,
                        'parallel_processing': True,
                        'memory_intensive': True,
                        'device_id': i
                    }
                )
                self.workers[worker.id] = worker
        
        logger.info(f"Initialized {len(self.workers)} workers")
    
    def _start_background_threads(self) -> None:
        """Start background processing threads."""
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._result_processor_thread = threading.Thread(target=self._result_processor_loop, daemon=True)
        
        self._scheduler_thread.start()
        self._heartbeat_thread.start()
        self._result_processor_thread.start()
        
        logger.info("Background threads started")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop for task distribution."""
        while not self._shutdown.is_set():
            try:
                # Get next task (blocks for up to 1 second)
                try:
                    _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Find best worker for task
                worker = self._find_best_worker(task)
                
                if worker is None:
                    # No available worker, put task back in queue
                    self.task_queue.put((task.priority, task))
                    time.sleep(0.1)  # Brief pause to avoid busy waiting
                    continue
                
                # Assign task to worker
                self._assign_task_to_worker(task, worker)
                
            except Exception as e:
                logger.error("Error in scheduler loop", exception=e)
                time.sleep(1.0)
    
    def _find_best_worker(self, task: Task) -> Optional[WorkerNode]:
        """Find the best available worker for a task."""
        available_workers = []
        
        for worker in self.workers.values():
            if (worker.status == "idle" and 
                worker.current_load < worker.max_capacity and
                self._worker_can_handle_task(worker, task)):
                available_workers.append(worker)
        
        if not available_workers:
            return None
        
        # Score workers based on multiple factors
        def score_worker(worker: WorkerNode) -> float:
            score = 0.0
            
            # Lower load is better
            load_factor = 1.0 - (worker.current_load / worker.max_capacity)
            score += load_factor * 0.4
            
            # Faster workers are better (lower average time)
            if worker.average_task_time > 0:
                speed_factor = 1.0 / (1.0 + worker.average_task_time)
                score += speed_factor * 0.3
            
            # Higher success rate is better
            total_tasks = worker.total_completed + worker.total_failed
            if total_tasks > 0:
                success_rate = worker.total_completed / total_tasks
                score += success_rate * 0.2
            
            # Capability match bonus
            if task.gpu_required and worker.node_type == "gpu":
                score += 0.1
            elif not task.gpu_required and worker.node_type == "cpu":
                score += 0.05
            
            return score
        
        # Select worker with highest score
        best_worker = max(available_workers, key=score_worker)
        return best_worker
    
    def _worker_can_handle_task(self, worker: WorkerNode, task: Task) -> bool:
        """Check if worker can handle the specific task."""
        # GPU requirement check
        if task.gpu_required and worker.node_type != "gpu":
            return False
        
        # Memory requirement check (basic estimation)
        if task.required_memory_mb > 8000 and worker.node_type == "cpu":
            return False  # CPU workers have memory limits
        
        # Worker-specific capability checks
        if task.task_type == "matrix_multiplication" and "gpu_compute" not in worker.capabilities:
            if task.required_memory_mb > 1000:  # Large matrix multiplication
                return False
        
        return True
    
    def _assign_task_to_worker(self, task: Task, worker: WorkerNode) -> None:
        """Assign task to a specific worker."""
        try:
            # Update worker state
            worker.current_load += 1
            worker.status = "busy"
            
            # Track active task
            self.active_tasks[task.id] = task
            
            # Submit to appropriate executor
            if worker.node_type == "gpu" and self._gpu_executor:
                future = self._gpu_executor.submit(self._execute_task, task, worker)
            else:
                future = self._cpu_executor.submit(self._execute_task, task, worker)
            
            # Add callback for completion
            future.add_done_callback(lambda f: self._handle_task_completion(f, task, worker))
            
            logger.debug(f"Assigned task {task.id} to worker {worker.id}")
            
        except Exception as e:
            logger.error(f"Failed to assign task {task.id} to worker {worker.id}", exception=e)
            worker.current_load = max(0, worker.current_load - 1)
            self._handle_task_failure(task, worker, str(e))
    
    def _execute_task(self, task: Task, worker: WorkerNode) -> TaskResult:
        """Execute a task on a worker."""
        start_time = time.time()
        
        try:
            # Set GPU device if needed
            if worker.node_type == "gpu" and "device_id" in worker.capabilities:
                device_id = worker.capabilities["device_id"]
                torch.cuda.set_device(device_id)
            
            # Execute the task
            if hasattr(task.data, '__call__'):
                # Task is a callable
                result = task.data()
            elif isinstance(task.data, dict) and 'function' in task.data:
                # Task is a function with arguments
                func = task.data['function']
                args = task.data.get('args', ())
                kwargs = task.data.get('kwargs', {})
                result = func(*args, **kwargs)
            else:
                # Generic task execution
                result = self._execute_generic_task(task, worker)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                worker_id=worker.id,
                result=result,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                worker_id=worker.id,
                result=None,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _execute_generic_task(self, task: Task, worker: WorkerNode) -> Any:
        """Execute generic computational tasks."""
        if task.task_type == "matrix_multiplication":
            return self._execute_matrix_multiplication(task, worker)
        elif task.task_type == "optimization_step":
            return self._execute_optimization_step(task, worker)
        elif task.task_type == "energy_computation":
            return self._execute_energy_computation(task, worker)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _execute_matrix_multiplication(self, task: Task, worker: WorkerNode) -> torch.Tensor:
        """Execute matrix multiplication task."""
        data = task.data
        A = data['matrix_a']
        B = data['matrix_b']
        
        # Move to appropriate device
        if worker.node_type == "gpu":
            device = f"cuda:{worker.capabilities.get('device_id', 0)}"
            A = A.to(device)
            B = B.to(device)
        
        result = torch.mm(A, B)
        return result.cpu()  # Return to CPU for consistency
    
    def _execute_optimization_step(self, task: Task, worker: WorkerNode) -> Dict[str, Any]:
        """Execute optimization step task."""
        data = task.data
        model = data['model']
        n_steps = data.get('n_steps', 1)
        
        # Perform optimization steps
        initial_energy = model.compute_energy()
        
        for _ in range(n_steps):
            # Simple Monte Carlo step (placeholder)
            model.flip_random_spin()
        
        final_energy = model.compute_energy()
        
        return {
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': final_energy - initial_energy,
            'steps_performed': n_steps
        }
    
    def _execute_energy_computation(self, task: Task, worker: WorkerNode) -> float:
        """Execute energy computation task."""
        data = task.data
        model = data['model']
        
        # Move to GPU if available and beneficial
        if worker.node_type == "gpu" and hasattr(model, 'to'):
            device = f"cuda:{worker.capabilities.get('device_id', 0)}"
            model = model.to(device)
        
        energy = model.compute_energy()
        return float(energy)
    
    def _handle_task_completion(self, future, task: Task, worker: WorkerNode) -> None:
        """Handle task completion callback."""
        try:
            result = future.result()
            
            # Update worker statistics
            worker.current_load = max(0, worker.current_load - 1)
            worker.total_completed += 1
            
            # Update average task time
            if worker.total_completed == 1:
                worker.average_task_time = result.execution_time
            else:
                # Exponential moving average
                alpha = 0.1
                worker.average_task_time = (alpha * result.execution_time + 
                                          (1 - alpha) * worker.average_task_time)
            
            # Update worker status
            if worker.current_load == 0:
                worker.status = "idle"
            
            # Store result
            self.completed_tasks[task.id] = result
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            # Update performance metrics
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['total_execution_time'] += result.execution_time
            
            # Call task callback if provided
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    logger.warning(f"Task callback failed for {task.id}", exception=e)
            
            logger.debug(f"Task {task.id} completed on worker {worker.id} in {result.execution_time:.3f}s")
            
        except Exception as e:
            self._handle_task_failure(task, worker, str(e))
    
    def _handle_task_failure(self, task: Task, worker: WorkerNode, error_message: str) -> None:
        """Handle task failure."""
        # Update worker statistics
        worker.current_load = max(0, worker.current_load - 1)
        worker.total_failed += 1
        
        if worker.current_load == 0:
            worker.status = "idle"
        
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Retry logic
        task.attempts += 1
        
        if task.attempts < task.max_attempts:
            # Requeue task for retry
            logger.warning(f"Task {task.id} failed (attempt {task.attempts}/{task.max_attempts}), retrying")
            self.task_queue.put((task.priority, task))
        else:
            # Max attempts reached
            logger.error(f"Task {task.id} failed permanently after {task.attempts} attempts: {error_message}")
            
            result = TaskResult(
                task_id=task.id,
                worker_id=worker.id,
                result=None,
                execution_time=0.0,
                success=False,
                error_message=error_message
            )
            
            self.completed_tasks[task.id] = result
            self.failed_tasks.append(task)
            self.performance_metrics['tasks_failed'] += 1
    
    def _heartbeat_loop(self) -> None:
        """Monitor worker health and update statistics."""
        while not self._shutdown.is_set():
            try:
                current_time = time.time()
                
                # Update worker heartbeats and detect failed workers
                for worker in self.workers.values():
                    worker.last_heartbeat = current_time
                    
                    # Simple health check (in a real system, this would be more sophisticated)
                    if worker.status == "busy" and worker.current_load == 0:
                        worker.status = "idle"
                
                # Update global performance metrics
                total_capacity = sum(w.max_capacity for w in self.workers.values())
                current_load = sum(w.current_load for w in self.workers.values())
                self.performance_metrics['worker_utilization'] = current_load / total_capacity if total_capacity > 0 else 0
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error("Error in heartbeat loop", exception=e)
                time.sleep(self.heartbeat_interval)
    
    def _result_processor_loop(self) -> None:
        """Process completed task results."""
        while not self._shutdown.is_set():
            try:
                # Check for results (non-blocking)
                try:
                    result = self.result_queue.get(timeout=1.0)
                    # Process result (placeholder for future extensions)
                    logger.debug(f"Processed result for task {result.task_id}")
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error("Error in result processor loop", exception=e)
                time.sleep(1.0)
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        self.performance_metrics['tasks_submitted'] += 1
        self.task_queue.put((task.priority, task))
        logger.debug(f"Submitted task {task.id} (priority {task.priority})")
        return task.id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            time.sleep(0.1)  # Brief pause
        
        return None  # Timeout reached
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks to complete."""
        start_time = time.time()
        results = {}
        
        while len(results) < len(task_ids):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            for task_id in task_ids:
                if task_id not in results and task_id in self.completed_tasks:
                    results[task_id] = self.completed_tasks[task_id]
            
            time.sleep(0.1)
        
        return [results.get(task_id) for task_id in task_ids]
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        stats = {
            'total_workers': len(self.workers),
            'worker_breakdown': defaultdict(int),
            'workers': {},
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        for worker in self.workers.values():
            stats['worker_breakdown'][worker.node_type] += 1
            stats['workers'][worker.id] = {
                'node_type': worker.node_type,
                'current_load': worker.current_load,
                'max_capacity': worker.max_capacity,
                'utilization': worker.current_load / worker.max_capacity,
                'total_completed': worker.total_completed,
                'total_failed': worker.total_failed,
                'success_rate': worker.total_completed / max(1, worker.total_completed + worker.total_failed),
                'average_task_time': worker.average_task_time,
                'status': worker.status
            }
        
        return stats
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """Graceful shutdown of the load balancer."""
        logger.info("Initiating load balancer shutdown")
        
        # Signal shutdown
        self._shutdown.set()
        
        # Wait for background threads
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
        if self._result_processor_thread:
            self._result_processor_thread.join(timeout=5.0)
        
        # Shutdown executors
        self._cpu_executor.shutdown(wait=True, timeout=timeout)
        if self._gpu_executor:
            self._gpu_executor.shutdown(wait=True, timeout=timeout)
        
        logger.info("Load balancer shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience functions for common use cases

def distribute_matrix_operations(matrices_a: List[torch.Tensor], 
                                matrices_b: List[torch.Tensor],
                                load_balancer: LoadBalancer) -> List[torch.Tensor]:
    """Distribute matrix multiplication operations across workers."""
    tasks = []
    
    for i, (a, b) in enumerate(zip(matrices_a, matrices_b)):
        task = Task(
            id=f"matmul_{i}",
            task_type="matrix_multiplication",
            priority=1,
            estimated_time=0.1,  # Estimate based on matrix size
            required_memory_mb=a.numel() * 4 / 1024 / 1024,  # Rough estimate for float32
            gpu_required=a.numel() > 1000000,  # Use GPU for large matrices
            data={'matrix_a': a, 'matrix_b': b}
        )
        
        task_id = load_balancer.submit_task(task)
        tasks.append(task_id)
    
    # Wait for all tasks to complete
    results = load_balancer.wait_for_completion(tasks)
    
    # Extract results
    matrices_result = []
    for result in results:
        if result and result.success:
            matrices_result.append(result.result)
        else:
            matrices_result.append(None)
    
    return matrices_result


def distribute_optimization_steps(models: List[Any],
                                 steps_per_model: int,
                                 load_balancer: LoadBalancer) -> List[Dict[str, Any]]:
    """Distribute optimization steps across multiple models."""
    tasks = []
    
    for i, model in enumerate(models):
        task = Task(
            id=f"opt_step_{i}",
            task_type="optimization_step",
            priority=1,
            estimated_time=steps_per_model * 0.01,
            required_memory_mb=100,  # Rough estimate
            gpu_required=False,
            data={'model': model, 'n_steps': steps_per_model}
        )
        
        task_id = load_balancer.submit_task(task)
        tasks.append(task_id)
    
    # Wait for completion
    results = load_balancer.wait_for_completion(tasks)
    
    # Extract optimization results
    opt_results = []
    for result in results:
        if result and result.success:
            opt_results.append(result.result)
        else:
            opt_results.append(None)
    
    return opt_results