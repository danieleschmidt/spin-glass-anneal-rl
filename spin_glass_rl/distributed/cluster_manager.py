"""Distributed computing cluster manager for large-scale optimization."""

import time
import json
import threading
import socket
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


class NodeStatus(Enum):
    """Node status in the cluster."""
    ACTIVE = "active"
    BUSY = "busy"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class ClusterNode:
    """Cluster node representation."""
    node_id: str
    hostname: str
    port: int
    status: NodeStatus = NodeStatus.ACTIVE
    cpu_cores: int = 4
    memory_gb: float = 8.0
    current_load: float = 0.0
    last_heartbeat: float = 0.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["cpu", "optimization"]


@dataclass
class DistributedTask:
    """Task for distributed execution."""
    task_id: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 0
    estimated_duration: float = 60.0
    resource_requirements: Dict[str, float] = None
    dependencies: List[str] = None
    created_time: float = 0.0
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {"cpu": 1.0, "memory": 1.0}
        if self.dependencies is None:
            self.dependencies = []
        if self.created_time == 0.0:
            self.created_time = time.time()


@dataclass
class TaskResult:
    """Result of distributed task execution."""
    task_id: str
    node_id: str
    success: bool
    result: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    completed_time: float = 0.0


class LoadBalancer:
    """Intelligent load balancer for distributed tasks."""
    
    def __init__(self):
        self.node_scores = {}
        self.task_history = {}
    
    def select_node(self, task: DistributedTask, available_nodes: List[ClusterNode]) -> Optional[ClusterNode]:
        """Select optimal node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes that meet resource requirements
        suitable_nodes = []
        for node in available_nodes:
            if self._can_handle_task(node, task):
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Score nodes based on multiple factors
        scored_nodes = []
        for node in suitable_nodes:
            score = self._calculate_node_score(node, task)
            scored_nodes.append((score, node))
        
        # Select node with best score
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return scored_nodes[0][1]
    
    def _can_handle_task(self, node: ClusterNode, task: DistributedTask) -> bool:
        """Check if node can handle the task."""
        # Check status
        if node.status not in [NodeStatus.ACTIVE]:
            return False
        
        # Check resource requirements
        cpu_required = task.resource_requirements.get("cpu", 1.0)
        memory_required = task.resource_requirements.get("memory", 1.0)
        
        available_cpu = node.cpu_cores * (1.0 - node.current_load)
        available_memory = node.memory_gb * (1.0 - node.current_load)
        
        return (available_cpu >= cpu_required and 
                available_memory >= memory_required)
    
    def _calculate_node_score(self, node: ClusterNode, task: DistributedTask) -> float:
        """Calculate node score for task assignment."""
        # Base score: availability
        load_score = 1.0 - node.current_load
        
        # Performance history score
        node_key = f"{node.node_id}_{task.function_name}"
        if node_key in self.task_history:
            history = self.task_history[node_key]
            avg_time = sum(h["duration"] for h in history) / len(history)
            performance_score = 1.0 / max(avg_time, 0.1)  # Inverse of execution time
        else:
            performance_score = 0.5  # Neutral for unknown performance
        
        # Resource efficiency score
        cpu_ratio = task.resource_requirements.get("cpu", 1.0) / node.cpu_cores
        memory_ratio = task.resource_requirements.get("memory", 1.0) / node.memory_gb
        resource_score = 1.0 - max(cpu_ratio, memory_ratio)
        
        # Capability match score
        capability_score = 1.0  # Default, could be enhanced with task-specific capabilities
        
        # Combine scores
        total_score = (
            load_score * 0.4 +
            performance_score * 0.3 +
            resource_score * 0.2 +
            capability_score * 0.1
        )
        
        return total_score
    
    def record_task_completion(self, node_id: str, task: DistributedTask, duration: float):
        """Record task completion for performance tracking."""
        key = f"{node_id}_{task.function_name}"
        if key not in self.task_history:
            self.task_history[key] = []
        
        self.task_history[key].append({
            "duration": duration,
            "timestamp": time.time(),
            "task_size": task.estimated_duration
        })
        
        # Keep only recent history
        if len(self.task_history[key]) > 20:
            self.task_history[key] = self.task_history[key][-10:]


class ClusterManager:
    """Manages a cluster of compute nodes for distributed optimization."""
    
    def __init__(self, manager_port: int = 8888):
        self.manager_port = manager_port
        self.nodes = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.load_balancer = LoadBalancer()
        
        # Cluster state
        self.cluster_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_nodes": 0
        }
        
        # Threading
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.heartbeat_thread = None
        self.task_dispatch_thread = None
    
    def add_node(self, node: ClusterNode) -> bool:
        """Add node to cluster."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        self.cluster_stats["active_nodes"] = len(self._get_active_nodes())
        print(f"Added node {node.node_id} to cluster")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.cluster_stats["active_nodes"] = len(self._get_active_nodes())
            print(f"Removed node {node_id} from cluster")
            return True
        return False
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution."""
        # Generate task ID if not provided
        if not task.task_id:
            task.task_id = self._generate_task_id(task)
        
        # Add to queue with priority
        priority = -task.priority  # Negative for max-heap behavior
        self.task_queue.put((priority, task.created_time, task))
        
        self.cluster_stats["total_tasks"] += 1
        print(f"Submitted task {task.task_id} to cluster")
        return task.task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """Get result of completed task."""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def start_cluster(self) -> bool:
        """Start cluster management."""
        self.running = True
        
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        # Start task dispatcher
        self.task_dispatch_thread = threading.Thread(target=self._task_dispatcher)
        self.task_dispatch_thread.daemon = True
        self.task_dispatch_thread.start()
        
        print(f"Cluster manager started on port {self.manager_port}")
        return True
    
    def stop_cluster(self):
        """Stop cluster management."""
        self.running = False
        print("Cluster manager stopped")
    
    def _get_active_nodes(self) -> List[ClusterNode]:
        """Get list of active nodes."""
        current_time = time.time()
        active_nodes = []
        
        for node in self.nodes.values():
            # Check if node is responsive (heartbeat within last 30 seconds)
            if (current_time - node.last_heartbeat) < 30.0:
                if node.status == NodeStatus.ACTIVE:
                    active_nodes.append(node)
            else:
                # Mark unresponsive nodes as failed
                node.status = NodeStatus.FAILED
        
        return active_nodes
    
    def _heartbeat_monitor(self):
        """Monitor node heartbeats."""
        while self.running:
            try:
                current_time = time.time()
                
                for node in self.nodes.values():
                    # Simulate heartbeat check
                    if (current_time - node.last_heartbeat) > 60.0:
                        if node.status != NodeStatus.FAILED:
                            print(f"Node {node.node_id} heartbeat timeout")
                            node.status = NodeStatus.FAILED
                
                # Update cluster stats
                self.cluster_stats["active_nodes"] = len(self._get_active_nodes())
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Heartbeat monitor error: {e}")
                time.sleep(5)
    
    def _task_dispatcher(self):
        """Dispatch tasks to available nodes."""
        while self.running:
            try:
                # Get task from queue (blocking with timeout)
                try:
                    priority, created_time, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Find available node
                active_nodes = self._get_active_nodes()
                selected_node = self.load_balancer.select_node(task, active_nodes)
                
                if selected_node:
                    # Dispatch task
                    future = self.executor.submit(self._execute_task, selected_node, task)
                    print(f"Dispatched task {task.task_id} to node {selected_node.node_id}")
                else:
                    # No available nodes, put task back in queue
                    self.task_queue.put((priority, created_time, task))
                    time.sleep(1)
                
            except Exception as e:
                print(f"Task dispatcher error: {e}")
                time.sleep(1)
    
    def _execute_task(self, node: ClusterNode, task: DistributedTask) -> TaskResult:
        """Execute task on specific node."""
        start_time = time.time()
        node.status = NodeStatus.BUSY
        
        try:
            # Simulate task execution
            # In real implementation, this would send task to node via network
            result = self._simulate_task_execution(task)
            
            execution_time = time.time() - start_time
            
            # Create successful result
            task_result = TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                success=True,
                result=result,
                execution_time=execution_time,
                completed_time=time.time()
            )
            
            self.completed_tasks[task.task_id] = task_result
            self.cluster_stats["completed_tasks"] += 1
            
            # Record performance
            self.load_balancer.record_task_completion(node.node_id, task, execution_time)
            
            print(f"Task {task.task_id} completed on {node.node_id} in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create failed result
            task_result = TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                completed_time=time.time()
            )
            
            self.failed_tasks[task.task_id] = task_result
            self.cluster_stats["failed_tasks"] += 1
            
            print(f"Task {task.task_id} failed on {node.node_id}: {e}")
        
        finally:
            node.status = NodeStatus.ACTIVE
        
        return task_result
    
    def _simulate_task_execution(self, task: DistributedTask) -> Any:
        """Simulate task execution (replace with actual execution)."""
        # Simulate computation time
        time.sleep(min(task.estimated_duration, 2.0))  # Cap simulation time
        
        # Simulate function execution based on function name
        if task.function_name == "optimize_ising":
            return {"energy": -42.5, "spins": [1, -1, 1, -1], "iterations": 1000}
        elif task.function_name == "compute_energy":
            return {"energy": sum(task.args) if task.args else 0}
        else:
            return {"result": "computed", "args": task.args}
    
    def _generate_task_id(self, task: DistributedTask) -> str:
        """Generate unique task ID."""
        content = f"{task.function_name}_{task.args}_{task.kwargs}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_nodes = self._get_active_nodes()
        
        node_summary = {}
        for node in self.nodes.values():
            node_summary[node.node_id] = {
                "status": node.status.value,
                "hostname": node.hostname,
                "load": node.current_load,
                "last_heartbeat": node.last_heartbeat,
                "capabilities": node.capabilities
            }
        
        return {
            "cluster_stats": self.cluster_stats.copy(),
            "active_nodes": len(active_nodes),
            "total_nodes": len(self.nodes),
            "queue_size": self.task_queue.qsize(),
            "node_details": node_summary,
            "load_balancer_history": len(self.load_balancer.task_history)
        }


class DistributedOptimizer:
    """High-level interface for distributed optimization."""
    
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster = cluster_manager
        self.task_futures = {}
    
    def distributed_anneal(
        self, 
        ising_models: List[Any], 
        annealing_params: Dict[str, Any],
        parallel_runs: int = 1
    ) -> List[Dict[str, Any]]:
        """Run distributed annealing on multiple Ising models."""
        tasks = []
        
        # Create tasks for each model and parallel run
        for model_idx, model in enumerate(ising_models):
            for run in range(parallel_runs):
                task = DistributedTask(
                    task_id="",  # Will be generated
                    function_name="optimize_ising",
                    args=(model_idx, run),
                    kwargs=annealing_params,
                    estimated_duration=annealing_params.get("estimated_time", 60.0),
                    resource_requirements={"cpu": 1.0, "memory": 0.5}
                )
                tasks.append(task)
        
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = self.cluster.submit_task(task)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.cluster.get_task_result(task_id, timeout=300.0)  # 5 minute timeout
            if result and result.success:
                results.append(result.result)
            else:
                error_msg = result.error_message if result else "Timeout"
                results.append({"error": error_msg})
        
        return results
    
    def parallel_benchmark(
        self, 
        problem_sizes: List[int], 
        benchmark_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run parallel benchmarking across cluster."""
        tasks = []
        
        for size in problem_sizes:
            task = DistributedTask(
                task_id="",
                function_name="benchmark_problem",
                args=(size,),
                kwargs=benchmark_params,
                estimated_duration=size * 0.01,  # Estimate based on size
                resource_requirements={"cpu": 1.0, "memory": size * 0.001}
            )
            tasks.append(task)
        
        # Submit and collect results
        task_ids = [self.cluster.submit_task(task) for task in tasks]
        
        benchmark_results = {}
        for i, task_id in enumerate(task_ids):
            result = self.cluster.get_task_result(task_id, timeout=600.0)
            size = problem_sizes[i]
            
            if result and result.success:
                benchmark_results[size] = {
                    "result": result.result,
                    "execution_time": result.execution_time,
                    "node_id": result.node_id
                }
            else:
                benchmark_results[size] = {
                    "error": result.error_message if result else "Timeout"
                }
        
        return benchmark_results


def test_distributed_cluster():
    """Test distributed cluster management."""
    print("🌐 Testing Distributed Cluster Manager...")
    
    # Create cluster manager
    cluster = ClusterManager(manager_port=8888)
    
    # Add some nodes
    nodes = [
        ClusterNode("node1", "localhost", 9001, cpu_cores=4, memory_gb=8.0),
        ClusterNode("node2", "localhost", 9002, cpu_cores=2, memory_gb=4.0),
        ClusterNode("node3", "localhost", 9003, cpu_cores=8, memory_gb=16.0)
    ]
    
    for node in nodes:
        cluster.add_node(node)
    
    # Start cluster
    cluster.start_cluster()
    
    print(f"✅ Started cluster with {len(nodes)} nodes")
    
    # Submit some tasks
    tasks = []
    for i in range(5):
        task = DistributedTask(
            task_id=f"test_task_{i}",
            function_name="compute_energy",
            args=(i, i*2),
            kwargs={"multiplier": 3},
            estimated_duration=1.0
        )
        task_id = cluster.submit_task(task)
        tasks.append(task_id)
    
    print(f"✅ Submitted {len(tasks)} tasks")
    
    # Wait for results
    results = []
    for task_id in tasks:
        result = cluster.get_task_result(task_id, timeout=10.0)
        if result:
            results.append(result.success)
        else:
            results.append(False)
    
    success_rate = sum(results) / len(results)
    print(f"✅ Task completion rate: {success_rate:.1%}")
    
    # Test distributed optimizer
    optimizer = DistributedOptimizer(cluster)
    
    # Simulate distributed annealing
    mock_models = [{"spins": 100}, {"spins": 200}]
    anneal_params = {"n_sweeps": 1000, "estimated_time": 1.0}
    
    anneal_results = optimizer.distributed_anneal(mock_models, anneal_params, parallel_runs=2)
    print(f"✅ Distributed annealing: {len(anneal_results)} results")
    
    # Get cluster status
    status = cluster.get_cluster_status()
    print(f"✅ Cluster status: {status['active_nodes']}/{status['total_nodes']} nodes active")
    print(f"   Completed tasks: {status['cluster_stats']['completed_tasks']}")
    print(f"   Queue size: {status['queue_size']}")
    
    # Stop cluster
    cluster.stop_cluster()
    
    print("🌐 Distributed Cluster Manager test completed!")


if __name__ == "__main__":
    test_distributed_cluster()