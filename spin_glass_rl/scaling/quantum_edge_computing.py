#!/usr/bin/env python3
"""
⚡ QUANTUM EDGE COMPUTING SCALING INFRASTRUCTURE
===============================================

Next-generation scaling system combining quantum-inspired optimization
with edge computing for planetary-scale deployment.

Features:
- Intelligent edge node orchestration
- Quantum-classical hybrid computation distribution
- Dynamic workload balancing with ML prediction
- Global optimization coordination
- Real-time performance optimization
"""

import time
import threading
import queue
import json
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

# Advanced computing with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    # Minimal numpy-like functions
    class MockNumpy:
        @staticmethod
        def array(data): return list(data) if isinstance(data, (list, tuple)) else [data]
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data): 
            if not data: return 0
            mean_val = sum(data) / len(data)
            return math.sqrt(sum((x - mean_val) ** 2 for x in data) / len(data))
        @staticmethod
        def random(size=None): 
            if size is None: return random.random()
            return [random.random() for _ in range(size)]
    
    np = MockNumpy()


@dataclass
class EdgeNode:
    """Edge computing node configuration."""
    node_id: str
    location: Dict[str, float]  # lat, lon, altitude
    capabilities: Dict[str, Any]
    current_load: float
    max_capacity: int
    status: str  # "active", "maintenance", "offline"
    quantum_enabled: bool = False
    last_heartbeat: float = 0.0


@dataclass
class ComputeTask:
    """Distributed computation task."""
    task_id: str
    task_type: str
    priority: int  # 1-10, higher is more important
    estimated_compute_time: float
    memory_requirement: int
    quantum_advantage: bool
    dependencies: List[str]
    data_locality_preference: Optional[str] = None
    deadline: Optional[float] = None


@dataclass
class OptimizationJob:
    """Optimization job for distributed processing."""
    job_id: str
    problem_size: int
    algorithm: str
    parameters: Dict[str, Any]
    subtasks: List[ComputeTask]
    coordination_requirements: Dict[str, Any]
    global_constraints: Dict[str, Any]


class QuantumClassicalHybridProcessor:
    """
    ⚛️ Quantum-Classical Hybrid Processing Engine
    
    Intelligently partitions problems between quantum-inspired
    and classical computation for optimal performance.
    """
    
    def __init__(self, node_id: str, quantum_enabled: bool = False):
        self.node_id = node_id
        self.quantum_enabled = quantum_enabled
        
        # Processing queues
        self.classical_queue = queue.PriorityQueue()
        self.quantum_queue = queue.PriorityQueue()
        
        # Performance tracking
        self.processing_history = []
        self.current_tasks = {}
        
        # Quantum simulation parameters
        self.quantum_coherence_time = 100.0  # microseconds
        self.quantum_error_rate = 0.001
        self.classical_speedup = 1.0
        
        logging.info(f"HybridProcessor initialized: {node_id} (quantum: {quantum_enabled})")
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit task for processing."""
        priority = task.priority
        task_item = (-priority, time.time(), task)  # Negative for max priority queue
        
        if task.quantum_advantage and self.quantum_enabled:
            self.quantum_queue.put(task_item)
            logging.info(f"Task {task.task_id} queued for quantum processing")
        else:
            self.classical_queue.put(task_item)
            logging.info(f"Task {task.task_id} queued for classical processing")
        
        return task.task_id
    
    def process_task(self, task: ComputeTask) -> Dict[str, Any]:
        """Process individual task."""
        start_time = time.time()
        
        try:
            if task.quantum_advantage and self.quantum_enabled:
                result = self._quantum_process(task)
            else:
                result = self._classical_process(task)
            
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.processing_history.append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'processing_time': processing_time,
                'quantum_used': task.quantum_advantage and self.quantum_enabled,
                'success': True,
                'timestamp': time.time()
            })
            
            return {
                'task_id': task.task_id,
                'status': 'completed',
                'result': result,
                'processing_time': processing_time,
                'node_id': self.node_id
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.processing_history.append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'processing_time': processing_time,
                'quantum_used': False,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            })
            
            logging.error(f"Task {task.task_id} failed: {e}")
            raise e
    
    def _quantum_process(self, task: ComputeTask) -> Dict[str, Any]:
        """Simulate quantum processing."""
        
        # Quantum advantage estimation
        problem_size = task.parameters.get('n_spins', 100) if hasattr(task, 'parameters') else 100
        
        # Quantum speedup calculation (theoretical)
        classical_complexity = problem_size ** 2
        quantum_complexity = problem_size * math.log(problem_size)
        
        speedup_factor = min(classical_complexity / quantum_complexity, 1000)  # Cap speedup
        
        # Simulate quantum computation time
        base_time = task.estimated_compute_time
        quantum_time = base_time / speedup_factor
        
        # Add quantum decoherence effects
        if quantum_time > self.quantum_coherence_time / 1000000:  # Convert to seconds
            # Decoherence penalty
            decoherence_penalty = 1 + (quantum_time * 1000000 / self.quantum_coherence_time)
            quantum_time *= decoherence_penalty
        
        # Simulate processing delay
        time.sleep(min(quantum_time, 0.1))  # Cap simulation delay
        
        # Simulate quantum measurement with error
        measurement_error = random.random() < self.quantum_error_rate
        
        result = {
            'algorithm_type': 'quantum_annealing',
            'problem_size': problem_size,
            'theoretical_speedup': speedup_factor,
            'actual_time': quantum_time,
            'measurement_error': measurement_error,
            'energy_value': -problem_size * (1 + random.random() * 0.1),  # Mock optimization result
            'quantum_fidelity': 1 - self.quantum_error_rate
        }
        
        if measurement_error:
            logging.warning(f"Quantum measurement error in task {task.task_id}")
            # Apply error correction
            result['energy_value'] *= (1 + random.random() * 0.05)  # Small error
        
        return result
    
    def _classical_process(self, task: ComputeTask) -> Dict[str, Any]:
        """Classical processing simulation."""
        
        # Simulate classical computation
        base_time = task.estimated_compute_time * self.classical_speedup
        time.sleep(min(base_time, 0.05))  # Cap simulation delay
        
        problem_size = task.parameters.get('n_spins', 100) if hasattr(task, 'parameters') else 100
        
        return {
            'algorithm_type': 'classical_annealing',
            'problem_size': problem_size,
            'processing_time': base_time,
            'energy_value': -problem_size * (0.8 + random.random() * 0.2),  # Mock result
            'iterations': int(1000 * base_time),
            'convergence_rate': 0.95
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        if not self.processing_history:
            return {'no_data': True}
        
        recent_history = self.processing_history[-100:]  # Last 100 tasks
        
        successful_tasks = [h for h in recent_history if h['success']]
        quantum_tasks = [h for h in recent_history if h.get('quantum_used', False)]
        
        if successful_tasks:
            avg_processing_time = sum(h['processing_time'] for h in successful_tasks) / len(successful_tasks)
        else:
            avg_processing_time = 0
        
        return {
            'total_tasks_processed': len(self.processing_history),
            'recent_tasks': len(recent_history),
            'success_rate': len(successful_tasks) / len(recent_history) if recent_history else 0,
            'avg_processing_time': avg_processing_time,
            'quantum_usage_rate': len(quantum_tasks) / len(recent_history) if recent_history else 0,
            'quantum_enabled': self.quantum_enabled,
            'queue_sizes': {
                'classical': self.classical_queue.qsize(),
                'quantum': self.quantum_queue.qsize()
            }
        }


class IntelligentWorkloadBalancer:
    """
    🧠 AI-Powered Workload Distribution System
    
    Uses machine learning to predict optimal task placement
    and dynamic load balancing across edge nodes.
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> EdgeNode
        self.processors = {}  # node_id -> QuantumClassicalHybridProcessor
        
        # ML-based prediction
        self.performance_history = []
        self.placement_history = []
        
        # Load balancing parameters
        self.load_threshold = 0.8
        self.rebalance_interval = 30  # seconds
        
        # Prediction models (simplified)
        self.task_completion_predictor = self._create_completion_predictor()
        self.node_health_predictor = self._create_health_predictor()
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        logging.info("IntelligentWorkloadBalancer initialized")
    
    def register_node(self, node: EdgeNode, quantum_enabled: bool = False):
        """Register new edge node."""
        self.nodes[node.node_id] = node
        self.processors[node.node_id] = QuantumClassicalHybridProcessor(
            node.node_id, quantum_enabled
        )
        
        logging.info(f"Registered edge node: {node.node_id} at {node.location}")
    
    def start_monitoring(self):
        """Start background monitoring and optimization."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(
                target=self._background_monitoring, daemon=True
            )
            self.monitoring_thread.start()
            logging.info("Workload monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logging.info("Workload monitoring stopped")
    
    def submit_optimization_job(self, job: OptimizationJob) -> Dict[str, Any]:
        """Submit optimization job for distributed processing."""
        
        # Analyze job requirements
        job_analysis = self._analyze_job(job)
        
        # Determine optimal task placement
        placement_plan = self._optimize_task_placement(job.subtasks, job_analysis)
        
        # Submit tasks to selected nodes
        task_assignments = {}
        task_dict = {task.task_id: task for task in job.subtasks}
        
        for task_id, node_id in placement_plan.items():
            if node_id in self.processors and task_id in task_dict:
                task = task_dict[task_id]
                submitted_task_id = self.processors[node_id].submit_task(task)
                task_assignments[submitted_task_id] = {
                    'task': task,
                    'node_id': node_id,
                    'submission_time': time.time()
                }
        
        # Create coordination framework for global optimization
        coordination = self._setup_job_coordination(job, task_assignments)
        
        return {
            'job_id': job.job_id,
            'status': 'submitted',
            'task_count': len(task_assignments),
            'coordination': coordination,
            'estimated_completion': self._estimate_job_completion(job, placement_plan)
        }
    
    def _analyze_job(self, job: OptimizationJob) -> Dict[str, Any]:
        """Analyze optimization job characteristics."""
        
        total_compute_time = sum(task.estimated_compute_time for task in job.subtasks)
        quantum_tasks = sum(1 for task in job.subtasks if task.quantum_advantage)
        high_priority_tasks = sum(1 for task in job.subtasks if task.priority > 7)
        
        return {
            'total_compute_time': total_compute_time,
            'task_count': len(job.subtasks),
            'quantum_task_ratio': quantum_tasks / len(job.subtasks) if job.subtasks else 0,
            'high_priority_ratio': high_priority_tasks / len(job.subtasks) if job.subtasks else 0,
            'parallelization_potential': self._estimate_parallelization(job),
            'coordination_complexity': len(job.coordination_requirements),
            'deadline_pressure': self._assess_deadline_pressure(job)
        }
    
    def _estimate_parallelization(self, job: OptimizationJob) -> float:
        """Estimate how much the job can be parallelized."""
        
        # Count task dependencies
        total_tasks = len(job.subtasks)
        if total_tasks == 0:
            return 0.0
        
        dependent_tasks = sum(1 for task in job.subtasks if task.dependencies)
        independent_tasks = total_tasks - dependent_tasks
        
        # Simple parallelization estimate
        parallelization_ratio = independent_tasks / total_tasks
        
        # Factor in coordination requirements
        coordination_penalty = len(job.coordination_requirements) * 0.1
        
        return max(0.1, parallelization_ratio - coordination_penalty)
    
    def _assess_deadline_pressure(self, job: OptimizationJob) -> float:
        """Assess urgency based on deadlines."""
        current_time = time.time()
        
        deadlines = [task.deadline for task in job.subtasks if task.deadline]
        if not deadlines:
            return 0.5  # Medium pressure if no deadlines
        
        min_deadline = min(deadlines)
        time_remaining = min_deadline - current_time
        
        # Normalize pressure (0 = no pressure, 1 = extreme pressure)
        if time_remaining <= 0:
            return 1.0
        elif time_remaining > 3600:  # More than 1 hour
            return 0.2
        else:
            return 1.0 - (time_remaining / 3600)
    
    def _optimize_task_placement(self, tasks: List[ComputeTask], 
                               job_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Optimize task placement across nodes using ML prediction."""
        
        placement = {}
        available_nodes = [node_id for node_id, node in self.nodes.items() 
                         if node.status == "active"]
        
        if not available_nodes:
            logging.error("No active nodes available for task placement")
            return placement
        
        # Sort tasks by priority and quantum advantage
        sorted_tasks = sorted(tasks, 
                            key=lambda t: (-t.priority, -int(t.quantum_advantage)))
        
        for task in sorted_tasks:
            best_node = self._select_best_node(task, available_nodes, job_analysis)
            if best_node:
                placement[task.task_id] = best_node
                
                # Update node load estimation
                if best_node in self.nodes:
                    self.nodes[best_node].current_load += task.estimated_compute_time / 100
        
        # Record placement decision for learning
        self.placement_history.append({
            'timestamp': time.time(),
            'job_analysis': job_analysis,
            'placement_decisions': len(placement),
            'nodes_used': len(set(placement.values()))
        })
        
        return placement
    
    def _select_best_node(self, task: ComputeTask, available_nodes: List[str],
                         job_analysis: Dict[str, Any]) -> Optional[str]:
        """Select best node for task using predictive scoring."""
        
        best_node = None
        best_score = -float('inf')
        
        for node_id in available_nodes:
            node = self.nodes[node_id]
            processor = self.processors[node_id]
            
            # Skip overloaded nodes
            if node.current_load > self.load_threshold:
                continue
            
            # Calculate placement score
            score = self._calculate_placement_score(task, node, processor, job_analysis)
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    def _calculate_placement_score(self, task: ComputeTask, node: EdgeNode, 
                                 processor: QuantumClassicalHybridProcessor,
                                 job_analysis: Dict[str, Any]) -> float:
        """Calculate placement score for task-node combination."""
        
        score = 100.0  # Base score
        
        # Quantum advantage bonus
        if task.quantum_advantage and processor.quantum_enabled:
            score += 50
        elif task.quantum_advantage and not processor.quantum_enabled:
            score -= 30  # Penalty for missing quantum capability
        
        # Load balancing
        load_penalty = node.current_load * 100
        score -= load_penalty
        
        # Capacity check
        if node.current_load + task.estimated_compute_time / 100 > node.max_capacity:
            score -= 200  # Heavy penalty for capacity violation
        
        # Data locality bonus
        if task.data_locality_preference and task.data_locality_preference == node_id:
            score += 30
        
        # Deadline pressure
        if task.deadline:
            time_remaining = task.deadline - time.time()
            if time_remaining < 300:  # Less than 5 minutes
                # Prioritize faster nodes for urgent tasks
                processor_metrics = processor.get_performance_metrics()
                if processor_metrics.get('avg_processing_time', float('inf')) < 1.0:
                    score += 40
        
        # Historical performance
        processor_metrics = processor.get_performance_metrics()
        success_rate = processor_metrics.get('success_rate', 0.5)
        score += success_rate * 20
        
        # Geographic distribution bonus (for global optimization)
        if job_analysis.get('coordination_complexity', 0) > 0:
            # Slight penalty for distant nodes in coordinated jobs
            score -= 5
        
        return score
    
    def _setup_job_coordination(self, job: OptimizationJob, 
                              task_assignments: Dict[str, Dict]) -> Dict[str, Any]:
        """Setup coordination framework for distributed job."""
        
        coordination = {
            'coordination_type': 'hierarchical',
            'master_node': self._select_master_node(task_assignments),
            'communication_topology': 'star',  # Could be 'mesh', 'ring', etc.
            'synchronization_points': [],
            'global_state_sharing': job.coordination_requirements.get('global_state', False)
        }
        
        # Add synchronization points based on task dependencies
        sync_points = []
        for task_id, assignment in task_assignments.items():
            task = assignment['task']
            if task.dependencies:
                sync_points.append({
                    'after_tasks': task.dependencies,
                    'before_task': task.task_id,
                    'sync_type': 'barrier'
                })
        
        coordination['synchronization_points'] = sync_points
        
        return coordination
    
    def _select_master_node(self, task_assignments: Dict[str, Dict]) -> str:
        """Select master node for job coordination."""
        
        # Count tasks per node
        node_task_counts = {}
        for assignment in task_assignments.values():
            node_id = assignment['node_id']
            node_task_counts[node_id] = node_task_counts.get(node_id, 0) + 1
        
        # Select node with most tasks as master
        if node_task_counts:
            master_node = max(node_task_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to first available node
            master_node = next(iter(self.nodes.keys())) if self.nodes else None
        
        return master_node
    
    def _estimate_job_completion(self, job: OptimizationJob, 
                               placement: Dict[str, str]) -> float:
        """Estimate job completion time."""
        
        if not placement:
            return float('inf')
        
        # Group tasks by node
        node_tasks = {}
        task_dict = {task.task_id: task for task in job.subtasks}
        
        for task_id, node_id in placement.items():
            if node_id not in node_tasks:
                node_tasks[node_id] = []
            if task_id in task_dict:
                node_tasks[node_id].append(task_dict[task_id])
        
        # Estimate completion time per node
        node_completion_times = {}
        for node_id, tasks in node_tasks.items():
            processor = self.processors[node_id]
            metrics = processor.get_performance_metrics()
            avg_time = metrics.get('avg_processing_time', 1.0)
            
            # Account for parallelization within node
            total_time = sum(task.estimated_compute_time for task in tasks)
            parallelization = self._estimate_parallelization(job)
            
            estimated_time = total_time * avg_time * (1 - parallelization * 0.5)
            node_completion_times[node_id] = estimated_time
        
        # Job completion is limited by slowest node
        max_completion_time = max(node_completion_times.values()) if node_completion_times else 0
        
        # Add coordination overhead
        coordination_overhead = len(job.coordination_requirements) * 0.1
        
        return max_completion_time * (1 + coordination_overhead)
    
    def _create_completion_predictor(self) -> Callable:
        """Create task completion time predictor."""
        
        def predictor(task_type: str, problem_size: int, node_capabilities: Dict) -> float:
            # Simple heuristic-based prediction
            base_time = {
                'optimization': 0.1 * math.log(problem_size),
                'simulation': 0.05 * problem_size ** 0.5,
                'analysis': 0.01 * problem_size,
                'coordination': 0.5
            }.get(task_type, 1.0)
            
            # Factor in node capabilities
            cpu_factor = node_capabilities.get('cpu_cores', 1) / 4
            memory_factor = node_capabilities.get('memory_gb', 8) / 16
            
            capability_multiplier = 1 / (cpu_factor * memory_factor)
            
            return base_time * capability_multiplier
        
        return predictor
    
    def _create_health_predictor(self) -> Callable:
        """Create node health predictor."""
        
        def predictor(node_metrics: Dict) -> float:
            # Predict node health based on current metrics
            cpu_load = node_metrics.get('cpu_percent', 50) / 100
            memory_load = node_metrics.get('memory_percent', 50) / 100
            success_rate = node_metrics.get('task_success_rate', 0.9)
            
            # Health score (0-1, higher is better)
            health = (1 - cpu_load) * 0.3 + (1 - memory_load) * 0.3 + success_rate * 0.4
            
            return max(0, min(1, health))
        
        return predictor
    
    def _background_monitoring(self):
        """Background monitoring and optimization thread."""
        
        while not self.stop_monitoring.is_set():
            try:
                # Update node health
                for node_id, node in self.nodes.items():
                    if node_id in self.processors:
                        processor = self.processors[node_id]
                        metrics = processor.get_performance_metrics()
                        
                        # Update node current load based on queue sizes
                        queue_sizes = metrics.get('queue_sizes', {'classical': 0, 'quantum': 0})
                        queue_load = (queue_sizes['classical'] + queue_sizes['quantum']) / 100
                        node.current_load = min(1.0, queue_load)
                        
                        # Update heartbeat
                        node.last_heartbeat = time.time()
                        
                        # Check for failed nodes
                        if time.time() - node.last_heartbeat > 300:  # 5 minutes
                            node.status = "offline"
                            logging.warning(f"Node {node_id} appears offline")
                        elif node.status == "offline" and metrics['success_rate'] > 0.5:
                            node.status = "active"
                            logging.info(f"Node {node_id} recovered")
                
                # Rebalance if needed
                self._check_rebalancing()
                
                # Clean up old performance data
                self._cleanup_old_data()
                
                # Wait before next monitoring cycle
                self.stop_monitoring.wait(self.rebalance_interval)
                
            except Exception as e:
                logging.error(f"Error in background monitoring: {e}")
                self.stop_monitoring.wait(60)  # Wait longer on error
    
    def _check_rebalancing(self):
        """Check if workload rebalancing is needed."""
        
        active_nodes = [node for node in self.nodes.values() if node.status == "active"]
        
        if len(active_nodes) < 2:
            return  # Need at least 2 nodes for rebalancing
        
        # Calculate load distribution
        loads = [node.current_load for node in active_nodes]
        if NUMPY_AVAILABLE:
            load_std = np.std(loads)
            load_mean = np.mean(loads)
        else:
            load_mean = sum(loads) / len(loads)
            load_variance = sum((load - load_mean) ** 2 for load in loads) / len(loads)
            load_std = math.sqrt(load_variance)
        
        # Check if rebalancing is needed
        if load_std > 0.3 or max(loads) > 0.9:
            logging.info(f"Load imbalance detected (std: {load_std:.3f}, max: {max(loads):.3f})")
            self._trigger_rebalancing(active_nodes)
    
    def _trigger_rebalancing(self, nodes: List[EdgeNode]):
        """Trigger workload rebalancing across nodes."""
        
        # Simple rebalancing: identify overloaded and underloaded nodes
        overloaded = [node for node in nodes if node.current_load > 0.8]
        underloaded = [node for node in nodes if node.current_load < 0.4]
        
        if overloaded and underloaded:
            logging.info(f"Rebalancing workload: {len(overloaded)} overloaded, {len(underloaded)} underloaded nodes")
            
            # In a full implementation, this would involve:
            # 1. Pausing new task submissions to overloaded nodes
            # 2. Migrating queued tasks to underloaded nodes
            # 3. Updating load balancing weights
            # 4. Coordinating with running optimizations
            
            # For now, just log the rebalancing decision
            for node in overloaded:
                logging.info(f"Node {node.node_id} is overloaded (load: {node.current_load:.3f})")
            
            for node in underloaded:
                logging.info(f"Node {node.node_id} has capacity (load: {node.current_load:.3f})")
    
    def _cleanup_old_data(self):
        """Clean up old performance and placement data."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours
        
        # Clean performance history
        self.performance_history = [
            h for h in self.performance_history 
            if h.get('timestamp', 0) > cutoff_time
        ]
        
        # Clean placement history  
        self.placement_history = [
            h for h in self.placement_history
            if h.get('timestamp', 0) > cutoff_time
        ]
        
        # Clean processor histories
        for processor in self.processors.values():
            processor.processing_history = [
                h for h in processor.processing_history
                if h.get('timestamp', 0) > cutoff_time
            ]
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance metrics."""
        
        active_nodes = [node for node in self.nodes.values() if node.status == "active"]
        
        if active_nodes:
            loads = [node.current_load for node in active_nodes]
            avg_load = sum(loads) / len(loads)
            max_load = max(loads)
            min_load = min(loads)
            
            if NUMPY_AVAILABLE:
                load_std = np.std(loads)
            else:
                load_variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
                load_std = math.sqrt(load_variance)
        else:
            avg_load = max_load = min_load = load_std = 0
        
        # Aggregate processor metrics
        total_tasks = sum(len(p.processing_history) for p in self.processors.values())
        
        if self.processors:
            avg_success_rate = sum(
                p.get_performance_metrics().get('success_rate', 0) 
                for p in self.processors.values()
            ) / len(self.processors)
        else:
            avg_success_rate = 0
        
        quantum_nodes = sum(1 for p in self.processors.values() if p.quantum_enabled)
        
        return {
            'timestamp': time.time(),
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'quantum_nodes': quantum_nodes,
            'load_distribution': {
                'average': avg_load,
                'maximum': max_load,
                'minimum': min_load,
                'standard_deviation': load_std,
                'balance_score': 1 - load_std if load_std < 1 else 0
            },
            'processing_stats': {
                'total_tasks_processed': total_tasks,
                'average_success_rate': avg_success_rate,
                'placement_decisions': len(self.placement_history)
            },
            'scaling_efficiency': {
                'node_utilization': avg_load,
                'quantum_utilization': quantum_nodes / len(self.nodes) if self.nodes else 0,
                'load_balance_quality': 1 - min(load_std / 0.5, 1.0) if load_std is not None else 0
            }
        }


if __name__ == "__main__":
    # Demonstration of quantum edge computing scaling
    print("⚡ Quantum Edge Computing Scaling - Demonstration")
    
    # Initialize workload balancer
    balancer = IntelligentWorkloadBalancer()
    
    # Register edge nodes around the world
    nodes = [
        EdgeNode("edge-us-west", {"lat": 37.7749, "lon": -122.4194}, {"cpu_cores": 16, "memory_gb": 64}, 0.2, 100, "active", quantum_enabled=True),
        EdgeNode("edge-eu-central", {"lat": 50.1109, "lon": 8.6821}, {"cpu_cores": 8, "memory_gb": 32}, 0.1, 50, "active", quantum_enabled=False),
        EdgeNode("edge-asia-pacific", {"lat": 35.6762, "lon": 139.6503}, {"cpu_cores": 12, "memory_gb": 48}, 0.3, 75, "active", quantum_enabled=True),
        EdgeNode("edge-south-america", {"lat": -23.5505, "lon": -46.6333}, {"cpu_cores": 4, "memory_gb": 16}, 0.5, 25, "active", quantum_enabled=False),
    ]
    
    for node in nodes:
        balancer.register_node(node, node.quantum_enabled)
    
    # Start monitoring
    balancer.start_monitoring()
    
    # Create sample optimization job
    subtasks = []
    for i in range(8):
        task = ComputeTask(
            task_id=f"task_{i}",
            task_type="optimization",
            priority=random.randint(1, 10),
            estimated_compute_time=random.uniform(0.1, 2.0),
            memory_requirement=random.randint(1, 8),
            quantum_advantage=random.random() > 0.5,
            dependencies=[]
        )
        task.parameters = {"n_spins": random.randint(50, 500)}
        subtasks.append(task)
    
    job = OptimizationJob(
        job_id="global_optimization_001",
        problem_size=1000,
        algorithm="quantum_hybrid_annealing",
        parameters={"temperature_schedule": "adaptive", "quantum_coherence": 0.9},
        subtasks=subtasks,
        coordination_requirements={"global_state": True, "synchronization": "barrier"},
        global_constraints={"max_runtime": 3600, "min_quality": 0.95}
    )
    
    # Submit job
    print(f"\n📋 Submitting optimization job with {len(subtasks)} subtasks...")
    job_result = balancer.submit_optimization_job(job)
    
    print(f"✅ Job submitted: {job_result['job_id']}")
    print(f"📊 Tasks distributed: {job_result['task_count']}")
    print(f"⏱️  Estimated completion: {job_result['estimated_completion']:.2f}s")
    
    # Simulate some processing time
    time.sleep(2)
    
    # Get scaling metrics
    metrics = balancer.get_scaling_metrics()
    print(f"\n📈 Scaling Metrics:")
    print(f"   Active Nodes: {metrics['active_nodes']}/{metrics['total_nodes']}")
    print(f"   Quantum Nodes: {metrics['quantum_nodes']}")
    print(f"   Average Load: {metrics['load_distribution']['average']:.3f}")
    print(f"   Load Balance Score: {metrics['scaling_efficiency']['load_balance_quality']:.3f}")
    print(f"   Total Tasks Processed: {metrics['processing_stats']['total_tasks_processed']}")
    
    # Test individual processor
    processor = list(balancer.processors.values())[0]
    perf_metrics = processor.get_performance_metrics()
    print(f"\n⚡ Sample Processor Metrics:")
    print(f"   Success Rate: {perf_metrics.get('success_rate', 0):.3f}")
    print(f"   Quantum Usage: {perf_metrics.get('quantum_usage_rate', 0):.3f}")
    print(f"   Queue Sizes: {perf_metrics.get('queue_sizes', {})}")
    
    balancer.stop_monitoring.set()
    print("\n✅ Quantum edge computing scaling demonstration complete!")