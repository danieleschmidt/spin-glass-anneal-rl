"""Distributed coordinator for managing cluster of workers."""

import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import weakref

from spin_glass_rl.utils.exceptions import ResourceError, ValidationError
from spin_glass_rl.utils.logging import get_logger


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """Worker status."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class Task:
    """Distributed task definition."""
    id: str
    type: str
    data: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3
    retry_count: int = 0
    timeout: float = 300.0
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class WorkerInfo:
    """Worker information."""
    id: str
    host: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    current_tasks: int = 0
    status: WorkerStatus = WorkerStatus.IDLE
    last_heartbeat: float = 0.0
    total_completed: int = 0
    total_failed: int = 0
    load_score: float = 0.0
    
    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


class DistributedCoordinator:
    """Distributed coordinator managing task distribution and worker coordination."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        heartbeat_interval: float = 30.0,
        task_timeout: float = 300.0
    ):
        """Initialize distributed coordinator.
        
        Args:
            host: Coordinator host address
            port: Coordinator port
            heartbeat_interval: Worker heartbeat interval in seconds
            task_timeout: Default task timeout in seconds
        """
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        
        # State management
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # Task IDs ordered by priority
        self.result_callbacks: Dict[str, List[Callable]] = {}
        
        # Synchronization
        self.lock = threading.RLock()
        self.running = False
        
        # Background threads
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Logging
        self.logger = get_logger("spin_glass_rl.distributed.coordinator")
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_workers": 0,
            "active_workers": 0
        }
    
    def start(self):
        """Start the coordinator."""
        if self.running:
            return
        
        self.running = True
        
        # Start background threads
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        self.scheduler_thread = threading.Thread(
            target=self._task_scheduler,
            daemon=True
        )
        self.scheduler_thread.start()
        
        self.logger.info(f"Distributed coordinator started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the coordinator."""
        self.running = False
        
        # Wait for threads to finish
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        self.logger.info("Distributed coordinator stopped")
    
    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        capabilities: List[str],
        max_concurrent_tasks: int = 1
    ) -> bool:
        """Register a new worker.
        
        Args:
            worker_id: Unique worker identifier
            host: Worker host address
            port: Worker port
            capabilities: List of task types the worker can handle
            max_concurrent_tasks: Maximum concurrent tasks for this worker
            
        Returns:
            True if registration successful
        """
        with self.lock:
            if worker_id in self.workers:
                self.logger.warning(f"Worker {worker_id} already registered, updating info")
            
            worker_info = WorkerInfo(
                id=worker_id,
                host=host,
                port=port,
                capabilities=capabilities,
                max_concurrent_tasks=max_concurrent_tasks,
                status=WorkerStatus.IDLE,
                last_heartbeat=time.time()
            )
            
            self.workers[worker_id] = worker_info
            self.stats["total_workers"] = len(self.workers)
            
            self.logger.info(f"Worker {worker_id} registered from {host}:{port}")
            return True
    
    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker.
        
        Args:
            worker_id: Worker identifier to unregister
            
        Returns:
            True if unregistration successful
        """
        with self.lock:
            if worker_id not in self.workers:
                return False
            
            # Cancel any tasks assigned to this worker
            for task_id, task in self.tasks.items():
                if task.worker_id == worker_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    self.task_queue.append(task_id)
                    self.logger.warning(f"Task {task_id} reassigned due to worker {worker_id} unregistration")
            
            del self.workers[worker_id]
            self.stats["total_workers"] = len(self.workers)
            
            self.logger.info(f"Worker {worker_id} unregistered")
            return True
    
    def submit_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        priority: int = 1,
        timeout: float = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a task for execution.
        
        Args:
            task_type: Type of task to execute
            task_data: Task data and parameters
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            callback: Optional callback for task completion
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            data=task_data,
            priority=priority,
            timeout=timeout or self.task_timeout
        )
        
        with self.lock:
            self.tasks[task_id] = task
            
            # Insert into queue by priority
            self._insert_task_by_priority(task_id)
            
            # Register callback
            if callback:
                if task_id not in self.result_callbacks:
                    self.result_callbacks[task_id] = []
                self.result_callbacks[task_id].append(callback)
            
            self.stats["tasks_submitted"] += 1
        
        self.logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    def _insert_task_by_priority(self, task_id: str):
        """Insert task into queue maintaining priority order."""
        task = self.tasks[task_id]
        
        # Find insertion point
        insert_index = len(self.task_queue)
        for i, queued_task_id in enumerate(self.task_queue):
            queued_task = self.tasks[queued_task_id]
            if task.priority > queued_task.priority:
                insert_index = i
                break
        
        self.task_queue.insert(insert_index, task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary or None if not found
        """
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            return {
                "id": task.id,
                "type": task.type,
                "status": task.status.value,
                "priority": task.priority,
                "worker_id": task.worker_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "retry_count": task.retry_count,
                "error": task.error
            }
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result if completed.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task result or None if not completed/found
        """
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancellation successful
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False  # Already finished
            
            # Remove from queue if pending
            if task.status == TaskStatus.PENDING and task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            self.logger.info(f"Task {task_id} cancelled")
            return True
    
    def heartbeat(self, worker_id: str, status: Dict[str, Any]) -> Dict[str, Any]:
        """Process worker heartbeat.
        
        Args:
            worker_id: Worker identifier
            status: Worker status information
            
        Returns:
            Response with any pending commands
        """
        with self.lock:
            if worker_id not in self.workers:
                return {"error": "Worker not registered"}
            
            worker = self.workers[worker_id]
            worker.last_heartbeat = time.time()
            worker.current_tasks = status.get("current_tasks", 0)
            worker.load_score = status.get("load_score", 0.0)
            
            # Update worker status
            if worker.current_tasks >= worker.max_concurrent_tasks:
                worker.status = WorkerStatus.BUSY
            else:
                worker.status = WorkerStatus.IDLE
            
            return {"status": "ok", "commands": []}
    
    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: Any = None,
        error: Optional[str] = None
    ):
        """Mark task as completed by worker.
        
        Args:
            task_id: Task identifier
            worker_id: Worker that completed the task
            result: Task result (if successful)
            error: Error message (if failed)
        """
        with self.lock:
            if task_id not in self.tasks:
                self.logger.warning(f"Unknown task completion: {task_id}")
                return
            
            task = self.tasks[task_id]
            
            if task.worker_id != worker_id:
                self.logger.warning(f"Task {task_id} completed by wrong worker: {worker_id} vs {task.worker_id}")
                return
            
            task.completed_at = time.time()
            
            if error:
                task.status = TaskStatus.FAILED
                task.error = error
                self.stats["tasks_failed"] += 1
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    task.started_at = None
                    task.completed_at = None
                    task.error = None
                    self.task_queue.append(task_id)
                    self.logger.info(f"Task {task_id} scheduled for retry ({task.retry_count}/{task.max_retries})")
                else:
                    self.logger.error(f"Task {task_id} failed permanently: {error}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.stats["tasks_completed"] += 1
                self.logger.info(f"Task {task_id} completed successfully")
            
            # Update worker stats
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_tasks = max(0, worker.current_tasks - 1)
                
                if error:
                    worker.total_failed += 1
                else:
                    worker.total_completed += 1
            
            # Execute callbacks
            if task_id in self.result_callbacks:
                for callback in self.result_callbacks[task_id]:
                    try:
                        callback(task)
                    except Exception as e:
                        self.logger.error(f"Task callback failed: {e}")
                
                del self.result_callbacks[task_id]
    
    def _task_scheduler(self):
        """Background task scheduler."""
        while self.running:
            try:
                self._schedule_pending_tasks()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")
    
    def _schedule_pending_tasks(self):
        """Schedule pending tasks to available workers."""
        with self.lock:
            if not self.task_queue:
                return
            
            # Get available workers
            available_workers = [
                worker for worker in self.workers.values()
                if (worker.status == WorkerStatus.IDLE and 
                    worker.current_tasks < worker.max_concurrent_tasks)
            ]
            
            if not available_workers:
                return
            
            # Schedule tasks
            tasks_scheduled = []
            
            for task_id in self.task_queue[:]:
                if not available_workers:
                    break
                
                task = self.tasks[task_id]
                
                # Find suitable worker
                suitable_workers = [
                    w for w in available_workers
                    if task.type in w.capabilities
                ]
                
                if not suitable_workers:
                    continue
                
                # Select worker with lowest load
                worker = min(suitable_workers, key=lambda w: w.load_score)
                
                # Assign task
                task.status = TaskStatus.ASSIGNED
                task.worker_id = worker.id
                task.started_at = time.time()
                
                worker.current_tasks += 1
                if worker.current_tasks >= worker.max_concurrent_tasks:
                    worker.status = WorkerStatus.BUSY
                    available_workers.remove(worker)
                
                tasks_scheduled.append(task_id)
                self.logger.info(f"Task {task_id} assigned to worker {worker.id}")
            
            # Remove scheduled tasks from queue
            for task_id in tasks_scheduled:
                self.task_queue.remove(task_id)
    
    def _heartbeat_monitor(self):
        """Monitor worker heartbeats and detect failures."""
        while self.running:
            try:
                current_time = time.time()
                timeout_threshold = current_time - (self.heartbeat_interval * 2)
                
                with self.lock:
                    offline_workers = []
                    
                    for worker_id, worker in self.workers.items():
                        if worker.last_heartbeat < timeout_threshold:
                            if worker.status != WorkerStatus.OFFLINE:
                                self.logger.warning(f"Worker {worker_id} heartbeat timeout")
                                worker.status = WorkerStatus.OFFLINE
                                offline_workers.append(worker_id)
                    
                    # Reassign tasks from offline workers
                    for worker_id in offline_workers:
                        for task_id, task in self.tasks.items():
                            if (task.worker_id == worker_id and 
                                task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]):
                                
                                task.status = TaskStatus.PENDING
                                task.worker_id = None
                                task.started_at = None
                                self.task_queue.append(task_id)
                                
                                self.logger.warning(f"Task {task_id} reassigned due to worker timeout")
                    
                    # Update active worker count
                    self.stats["active_workers"] = sum(
                        1 for w in self.workers.values()
                        if w.status != WorkerStatus.OFFLINE
                    )
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status.
        
        Returns:
            Cluster status dictionary
        """
        with self.lock:
            # Task statistics by status
            task_stats = {}
            for status in TaskStatus:
                task_stats[status.value] = sum(
                    1 for task in self.tasks.values()
                    if task.status == status
                )
            
            # Worker statistics by status
            worker_stats = {}
            for status in WorkerStatus:
                worker_stats[status.value] = sum(
                    1 for worker in self.workers.values()
                    if worker.status == status
                )
            
            return {
                "coordinator": {
                    "host": self.host,
                    "port": self.port,
                    "running": self.running,
                    "uptime": time.time() - getattr(self, '_start_time', time.time())
                },
                "workers": {
                    "total": len(self.workers),
                    "by_status": worker_stats,
                    "details": [asdict(worker) for worker in self.workers.values()]
                },
                "tasks": {
                    "total": len(self.tasks),
                    "queued": len(self.task_queue),
                    "by_status": task_stats
                },
                "statistics": self.stats.copy()
            }
    
    def cleanup_completed_tasks(self, max_age_hours: float = 24.0):
        """Clean up old completed tasks.
        
        Args:
            max_age_hours: Maximum age of completed tasks to keep
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.lock:
            completed_task_ids = [
                task_id for task_id, task in self.tasks.items()
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and task.completed_at < cutoff_time)
            ]
            
            for task_id in completed_task_ids:
                del self.tasks[task_id]
                if task_id in self.result_callbacks:
                    del self.result_callbacks[task_id]
            
            if completed_task_ids:
                self.logger.info(f"Cleaned up {len(completed_task_ids)} old completed tasks")