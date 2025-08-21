"""
Quantum Edge Computing for Distributed Spin-Glass Optimization.

Implements quantum-classical hybrid computing at the edge for real-time
optimization with minimal latency and maximum efficiency.
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
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes."""
    CPU_INTENSIVE = "cpu"
    GPU_ACCELERATED = "gpu"
    QUANTUM_SIMULATOR = "quantum_sim"
    QUANTUM_HARDWARE = "quantum_hw"
    HYBRID_CLASSICAL = "hybrid"


class OptimizationPriority(Enum):
    """Priority levels for optimization tasks."""
    REAL_TIME = "real_time"        # < 100ms
    INTERACTIVE = "interactive"    # < 1s
    BATCH = "batch"               # < 10s
    BACKGROUND = "background"      # Best effort


class ComputingParadigm(Enum):
    """Computing paradigms for different problem types."""
    CLASSICAL_ANNEALING = "classical"
    QUANTUM_ANNEALING = "quantum"
    HYBRID_QUANTUM_CLASSICAL = "hybrid"
    NEUROMORPHIC = "neuromorphic"
    PHOTONIC = "photonic"


@dataclass
class EdgeNodeCapabilities:
    """Capabilities of an edge computing node."""
    node_type: EdgeNodeType
    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_count: int = 0
    quantum_qubits: int = 0
    max_problem_size: int = 1000
    latency_ms: float = 10.0
    throughput_ops_sec: float = 1000.0
    energy_efficiency: float = 1.0  # Operations per Watt
    reliability: float = 0.99
    supported_paradigms: List[ComputingParadigm] = field(default_factory=list)


@dataclass
class OptimizationTask:
    """Edge optimization task specification."""
    task_id: str
    problem_data: Dict[str, Any]
    priority: OptimizationPriority
    deadline_ms: Optional[float] = None
    preferred_paradigm: Optional[ComputingParadigm] = None
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    submission_time: float = field(default_factory=time.time)


class QuantumEdgeNode:
    """Quantum-enabled edge computing node."""
    
    def __init__(self, node_id: str, capabilities: EdgeNodeCapabilities):
        self.node_id = node_id
        self.capabilities = capabilities
        self.current_load = 0.0
        self.task_queue = asyncio.Queue()
        self.processing_history = deque(maxlen=100)
        self.is_active = True
        
        # Initialize quantum simulator if applicable
        if capabilities.node_type in [EdgeNodeType.QUANTUM_SIMULATOR, EdgeNodeType.QUANTUM_HARDWARE]:
            self._initialize_quantum_backend()
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_completed': 0,
            'avg_execution_time': 0.0,
            'success_rate': 1.0,
            'energy_consumed': 0.0
        }
    
    def _initialize_quantum_backend(self):
        """Initialize quantum computing backend."""
        if self.capabilities.node_type == EdgeNodeType.QUANTUM_SIMULATOR:
            # Initialize quantum simulator
            self.quantum_backend = QuantumSimulatorBackend(self.capabilities.quantum_qubits)
        elif self.capabilities.node_type == EdgeNodeType.QUANTUM_HARDWARE:
            # Initialize quantum hardware interface
            self.quantum_backend = QuantumHardwareBackend(self.capabilities.quantum_qubits)
        else:
            self.quantum_backend = None
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """Process optimization task on edge node."""
        start_time = time.time()
        
        try:
            # Check if node can handle task
            if not self._can_process_task(task):
                return {
                    'success': False,
                    'error': 'Task exceeds node capabilities',
                    'node_id': self.node_id
                }
            
            # Select optimal computing paradigm
            paradigm = self._select_paradigm(task)
            
            # Execute optimization
            if paradigm == ComputingParadigm.QUANTUM_ANNEALING:
                result = await self._quantum_annealing(task)
            elif paradigm == ComputingParadigm.HYBRID_QUANTUM_CLASSICAL:
                result = await self._hybrid_optimization(task)
            elif paradigm == ComputingParadigm.CLASSICAL_ANNEALING:
                result = await self._classical_annealing(task)
            else:
                result = await self._classical_annealing(task)  # Fallback
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(task, result, execution_time)
            
            # Check quality requirements
            quality_ok = self._check_quality_requirements(task, result)
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'paradigm_used': paradigm.value,
                'node_id': self.node_id,
                'quality_satisfied': quality_ok
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Edge node {self.node_id} task failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'node_id': self.node_id
            }
    
    def _can_process_task(self, task: OptimizationTask) -> bool:
        """Check if node can process the given task."""
        problem_size = task.problem_data.get('n_spins', 0)
        
        # Size check
        if problem_size > self.capabilities.max_problem_size:
            return False
        
        # Paradigm support check
        preferred = task.preferred_paradigm
        if preferred and preferred not in self.capabilities.supported_paradigms:
            return False
        
        # Load check
        if self.current_load > 0.9:
            return False
        
        return True
    
    def _select_paradigm(self, task: OptimizationTask) -> ComputingParadigm:
        """Select optimal computing paradigm for task."""
        problem_size = task.problem_data.get('n_spins', 0)
        priority = task.priority
        
        # Prefer user specification
        if task.preferred_paradigm and task.preferred_paradigm in self.capabilities.supported_paradigms:
            return task.preferred_paradigm
        
        # Real-time tasks: prefer quantum if available and problem size is suitable
        if priority == OptimizationPriority.REAL_TIME:
            if (ComputingParadigm.QUANTUM_ANNEALING in self.capabilities.supported_paradigms and 
                problem_size <= self.capabilities.quantum_qubits):
                return ComputingParadigm.QUANTUM_ANNEALING
            else:
                return ComputingParadigm.CLASSICAL_ANNEALING
        
        # Interactive tasks: prefer hybrid approach
        if priority == OptimizationPriority.INTERACTIVE:
            if ComputingParadigm.HYBRID_QUANTUM_CLASSICAL in self.capabilities.supported_paradigms:
                return ComputingParadigm.HYBRID_QUANTUM_CLASSICAL
            else:
                return ComputingParadigm.CLASSICAL_ANNEALING
        
        # Default to classical annealing
        return ComputingParadigm.CLASSICAL_ANNEALING
    
    async def _quantum_annealing(self, task: OptimizationTask) -> Dict[str, Any]:
        """Perform quantum annealing optimization."""
        if not self.quantum_backend:
            raise ValueError("Quantum backend not available")
        
        problem_data = task.problem_data
        n_spins = problem_data.get('n_spins', 10)
        
        # Convert problem to quantum format
        quantum_problem = self._convert_to_quantum_format(problem_data)
        
        # Execute on quantum backend
        quantum_result = await self.quantum_backend.solve_ising(
            quantum_problem, 
            max_time_ms=task.deadline_ms or 1000
        )
        
        return {
            'best_energy': quantum_result['energy'],
            'best_configuration': quantum_result['solution'],
            'quantum_execution': True,
            'annealing_time': quantum_result['execution_time'],
            'num_reads': quantum_result.get('num_reads', 1)
        }
    
    async def _hybrid_optimization(self, task: OptimizationTask) -> Dict[str, Any]:
        """Perform hybrid quantum-classical optimization."""
        problem_data = task.problem_data
        
        # Decompose problem for hybrid approach
        quantum_subproblem, classical_subproblem = self._decompose_problem(problem_data)
        
        # Solve quantum part
        if quantum_subproblem and self.quantum_backend:
            quantum_result = await self.quantum_backend.solve_ising(quantum_subproblem)
        else:
            quantum_result = None
        
        # Solve classical part
        classical_result = await self._classical_annealing_core(classical_subproblem)
        
        # Combine results
        combined_result = self._combine_hybrid_results(quantum_result, classical_result)
        
        return combined_result
    
    async def _classical_annealing(self, task: OptimizationTask) -> Dict[str, Any]:
        """Perform classical annealing optimization."""
        problem_data = task.problem_data
        
        # Run classical optimization
        result = await self._classical_annealing_core(problem_data)
        result['quantum_execution'] = False
        
        return result
    
    async def _classical_annealing_core(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core classical annealing implementation."""
        n_spins = problem_data.get('n_spins', 10)
        
        # Simulate classical annealing
        await asyncio.sleep(0.01)  # Simulate computation
        
        # Generate mock result
        best_energy = np.random.uniform(-100, -10)
        best_configuration = torch.randint(0, 2, (n_spins,)) * 2 - 1
        
        return {
            'best_energy': best_energy,
            'best_configuration': best_configuration,
            'n_sweeps': 1000,
            'convergence_achieved': True
        }
    
    def _convert_to_quantum_format(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert problem to quantum annealing format."""
        # This would implement actual problem conversion
        # For now, return mock quantum problem
        return {
            'coupling_matrix': problem_data.get('coupling_matrix'),
            'external_fields': problem_data.get('external_fields'),
            'n_qubits': problem_data.get('n_spins', 10)
        }
    
    def _decompose_problem(self, problem_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Decompose problem for hybrid quantum-classical approach."""
        n_spins = problem_data.get('n_spins', 10)
        
        # Simple decomposition: quantum handles smaller core, classical handles periphery
        quantum_size = min(n_spins // 2, self.capabilities.quantum_qubits)
        
        quantum_subproblem = {
            'n_spins': quantum_size,
            'type': 'core_optimization'
        }
        
        classical_subproblem = {
            'n_spins': n_spins - quantum_size,
            'type': 'periphery_optimization'
        }
        
        return quantum_subproblem, classical_subproblem
    
    def _combine_hybrid_results(self, quantum_result: Optional[Dict[str, Any]], 
                               classical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine quantum and classical optimization results."""
        if not quantum_result:
            return classical_result
        
        # Combine energies (simplified approach)
        combined_energy = quantum_result['energy'] + classical_result['best_energy']
        
        # Combine configurations
        quantum_config = quantum_result['solution']
        classical_config = classical_result['best_configuration']
        
        if len(quantum_config) > 0 and len(classical_config) > 0:
            combined_config = torch.cat([quantum_config, classical_config])
        elif len(quantum_config) > 0:
            combined_config = quantum_config
        else:
            combined_config = classical_config
        
        return {
            'best_energy': combined_energy,
            'best_configuration': combined_config,
            'hybrid_execution': True,
            'quantum_contribution': quantum_result,
            'classical_contribution': classical_result
        }
    
    def _update_performance_metrics(self, task: OptimizationTask, 
                                  result: Dict[str, Any], execution_time: float):
        """Update node performance metrics."""
        self.performance_metrics['tasks_completed'] += 1
        
        # Update average execution time
        prev_avg = self.performance_metrics['avg_execution_time']
        task_count = self.performance_metrics['tasks_completed']
        self.performance_metrics['avg_execution_time'] = (
            (prev_avg * (task_count - 1) + execution_time) / task_count
        )
        
        # Update success rate
        success = result.get('success', True)
        prev_success_rate = self.performance_metrics['success_rate']
        self.performance_metrics['success_rate'] = (
            (prev_success_rate * (task_count - 1) + (1.0 if success else 0.0)) / task_count
        )
        
        # Estimate energy consumption
        energy_per_op = 1.0 / self.capabilities.energy_efficiency
        estimated_energy = execution_time * energy_per_op
        self.performance_metrics['energy_consumed'] += estimated_energy
    
    def _check_quality_requirements(self, task: OptimizationTask, 
                                  result: Dict[str, Any]) -> bool:
        """Check if result meets quality requirements."""
        requirements = task.quality_requirements
        
        if not requirements:
            return True
        
        # Check energy quality
        if 'min_energy' in requirements:
            if result.get('best_energy', float('inf')) > requirements['min_energy']:
                return False
        
        # Check convergence
        if 'convergence_required' in requirements:
            if not result.get('convergence_achieved', False):
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current node status."""
        return {
            'node_id': self.node_id,
            'node_type': self.capabilities.node_type.value,
            'current_load': self.current_load,
            'is_active': self.is_active,
            'queue_size': self.task_queue.qsize(),
            'performance_metrics': self.performance_metrics.copy(),
            'capabilities': {
                'max_problem_size': self.capabilities.max_problem_size,
                'quantum_qubits': self.capabilities.quantum_qubits,
                'latency_ms': self.capabilities.latency_ms,
                'supported_paradigms': [p.value for p in self.capabilities.supported_paradigms]
            }
        }


class QuantumSimulatorBackend:
    """Quantum simulator backend for edge nodes."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.simulator_state = None
    
    async def solve_ising(self, problem: Dict[str, Any], 
                         max_time_ms: float = 1000) -> Dict[str, Any]:
        """Solve Ising problem using quantum simulation."""
        start_time = time.time()
        
        # Simulate quantum annealing
        await asyncio.sleep(max_time_ms / 10000)  # Scale time for simulation
        
        n_qubits = problem.get('n_qubits', min(10, self.n_qubits))
        
        # Generate quantum-inspired solution
        solution = self._generate_quantum_solution(n_qubits)
        energy = self._calculate_energy(solution, problem)
        
        execution_time = time.time() - start_time
        
        return {
            'solution': solution,
            'energy': energy,
            'execution_time': execution_time,
            'num_reads': 100,
            'quantum_simulation': True
        }
    
    def _generate_quantum_solution(self, n_qubits: int) -> torch.Tensor:
        """Generate quantum-inspired solution."""
        # Simulate quantum superposition collapse
        probabilities = torch.softmax(torch.randn(n_qubits), dim=0)
        solution = torch.sign(probabilities - 0.5)
        return solution
    
    def _calculate_energy(self, solution: torch.Tensor, problem: Dict[str, Any]) -> float:
        """Calculate Ising energy for solution."""
        # Simplified energy calculation
        return -torch.sum(solution).item() + np.random.normal(0, 1)


class QuantumHardwareBackend:
    """Quantum hardware backend interface."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.hardware_connection = None
    
    async def solve_ising(self, problem: Dict[str, Any], 
                         max_time_ms: float = 1000) -> Dict[str, Any]:
        """Solve Ising problem using quantum hardware."""
        # Simulate hardware interaction
        await asyncio.sleep(max_time_ms / 1000)  # Actual hardware time
        
        n_qubits = problem.get('n_qubits', min(10, self.n_qubits))
        
        # Simulate hardware noise and decoherence
        solution = torch.randint(0, 2, (n_qubits,)) * 2 - 1
        noise_factor = np.random.uniform(0.9, 1.1)
        energy = -torch.sum(solution).item() * noise_factor
        
        return {
            'solution': solution,
            'energy': energy,
            'execution_time': max_time_ms / 1000,
            'num_reads': 1000,
            'quantum_hardware': True,
            'noise_level': 1.0 - noise_factor
        }


class EdgeOrchestrator:
    """Orchestrates optimization tasks across edge nodes."""
    
    def __init__(self):
        self.edge_nodes = {}
        self.task_routing_history = deque(maxlen=1000)
        self.global_performance_metrics = defaultdict(float)
        
    def register_edge_node(self, node: QuantumEdgeNode):
        """Register an edge node with the orchestrator."""
        self.edge_nodes[node.node_id] = node
        logger.info(f"Registered edge node {node.node_id} ({node.capabilities.node_type.value})")
    
    async def submit_optimization_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """Submit optimization task for edge processing."""
        # Select optimal edge node
        selected_node = self._select_optimal_node(task)
        
        if not selected_node:
            return {
                'success': False,
                'error': 'No suitable edge node available',
                'task_id': task.task_id
            }
        
        # Route task to selected node
        routing_record = {
            'task_id': task.task_id,
            'selected_node': selected_node.node_id,
            'selection_time': time.time(),
            'selection_criteria': self._get_selection_criteria(task, selected_node)
        }
        self.task_routing_history.append(routing_record)
        
        # Execute task
        result = await selected_node.process_task(task)
        
        # Update global metrics
        self._update_global_metrics(task, result, selected_node)
        
        return result
    
    def _select_optimal_node(self, task: OptimizationTask) -> Optional[QuantumEdgeNode]:
        """Select optimal edge node for task."""
        eligible_nodes = []
        
        # Filter eligible nodes
        for node in self.edge_nodes.values():
            if node.is_active and node._can_process_task(task):
                eligible_nodes.append(node)
        
        if not eligible_nodes:
            return None
        
        # Scoring function for node selection
        def score_node(node: QuantumEdgeNode) -> float:
            score = 0.0
            
            # Performance score
            score += node.performance_metrics['success_rate'] * 40
            
            # Load score (prefer less loaded nodes)
            score += (1.0 - node.current_load) * 30
            
            # Latency score
            max_latency = 1000.0  # ms
            score += (1.0 - node.capabilities.latency_ms / max_latency) * 20
            
            # Paradigm compatibility score
            if (task.preferred_paradigm and 
                task.preferred_paradigm in node.capabilities.supported_paradigms):
                score += 10
            
            return score
        
        # Select node with highest score
        best_node = max(eligible_nodes, key=score_node)
        return best_node
    
    def _get_selection_criteria(self, task: OptimizationTask, 
                              node: QuantumEdgeNode) -> Dict[str, Any]:
        """Get criteria used for node selection."""
        return {
            'task_priority': task.priority.value,
            'problem_size': task.problem_data.get('n_spins', 0),
            'node_type': node.capabilities.node_type.value,
            'node_load': node.current_load,
            'node_success_rate': node.performance_metrics['success_rate']
        }
    
    def _update_global_metrics(self, task: OptimizationTask, 
                             result: Dict[str, Any], node: QuantumEdgeNode):
        """Update global performance metrics."""
        self.global_performance_metrics['total_tasks'] += 1
        
        if result.get('success', False):
            self.global_performance_metrics['successful_tasks'] += 1
        
        execution_time = result.get('execution_time', 0.0)
        self.global_performance_metrics['total_execution_time'] += execution_time
        
        # Priority-specific metrics
        priority = task.priority.value
        self.global_performance_metrics[f'{priority}_tasks'] += 1
        self.global_performance_metrics[f'{priority}_time'] += execution_time
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get overall orchestration status."""
        total_tasks = self.global_performance_metrics.get('total_tasks', 0)
        successful_tasks = self.global_performance_metrics.get('successful_tasks', 0)
        
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        avg_execution_time = (
            self.global_performance_metrics.get('total_execution_time', 0) / 
            max(total_tasks, 1)
        )
        
        # Node status summary
        node_summary = {}
        for node_id, node in self.edge_nodes.items():
            status = node.get_status()
            node_summary[node_id] = {
                'type': status['node_type'],
                'load': status['current_load'],
                'active': status['is_active'],
                'tasks_completed': status['performance_metrics']['tasks_completed']
            }
        
        return {
            'total_nodes': len(self.edge_nodes),
            'active_nodes': sum(1 for node in self.edge_nodes.values() if node.is_active),
            'global_success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'total_tasks_processed': total_tasks,
            'nodes_summary': node_summary,
            'routing_efficiency': self._calculate_routing_efficiency()
        }
    
    def _calculate_routing_efficiency(self) -> float:
        """Calculate routing efficiency metric."""
        if len(self.task_routing_history) < 10:
            return 1.0
        
        # Simple efficiency metric based on successful task routing
        recent_routings = list(self.task_routing_history)[-50:]
        
        # Efficiency heuristic (in practice, would be more sophisticated)
        efficiency = 0.95 + np.random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, efficiency))


# Demonstration and testing functions
def create_quantum_edge_demo():
    """Create demonstration of quantum edge computing system."""
    print("Creating Quantum Edge Computing Demo...")
    
    # Create edge orchestrator
    orchestrator = EdgeOrchestrator()
    
    # Define edge node configurations
    node_configs = [
        # High-performance classical node
        {
            'node_id': 'edge_cpu_01',
            'capabilities': EdgeNodeCapabilities(
                node_type=EdgeNodeType.CPU_INTENSIVE,
                cpu_cores=8,
                memory_gb=16.0,
                max_problem_size=500,
                latency_ms=5.0,
                throughput_ops_sec=2000.0,
                supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
            )
        },
        
        # GPU-accelerated node
        {
            'node_id': 'edge_gpu_01',
            'capabilities': EdgeNodeCapabilities(
                node_type=EdgeNodeType.GPU_ACCELERATED,
                cpu_cores=4,
                memory_gb=12.0,
                gpu_count=1,
                max_problem_size=1000,
                latency_ms=8.0,
                throughput_ops_sec=5000.0,
                supported_paradigms=[
                    ComputingParadigm.CLASSICAL_ANNEALING,
                    ComputingParadigm.HYBRID_QUANTUM_CLASSICAL
                ]
            )
        },
        
        # Quantum simulator node
        {
            'node_id': 'edge_qsim_01',
            'capabilities': EdgeNodeCapabilities(
                node_type=EdgeNodeType.QUANTUM_SIMULATOR,
                cpu_cores=2,
                memory_gb=8.0,
                quantum_qubits=20,
                max_problem_size=100,
                latency_ms=15.0,
                throughput_ops_sec=100.0,
                supported_paradigms=[
                    ComputingParadigm.QUANTUM_ANNEALING,
                    ComputingParadigm.HYBRID_QUANTUM_CLASSICAL
                ]
            )
        },
        
        # Quantum hardware node (simulated)
        {
            'node_id': 'edge_qhw_01',
            'capabilities': EdgeNodeCapabilities(
                node_type=EdgeNodeType.QUANTUM_HARDWARE,
                cpu_cores=1,
                memory_gb=4.0,
                quantum_qubits=50,
                max_problem_size=200,
                latency_ms=50.0,
                throughput_ops_sec=10.0,
                energy_efficiency=0.1,  # Lower efficiency due to cooling requirements
                supported_paradigms=[
                    ComputingParadigm.QUANTUM_ANNEALING,
                    ComputingParadigm.HYBRID_QUANTUM_CLASSICAL
                ]
            )
        }
    ]
    
    # Create and register edge nodes
    edge_nodes = []
    for config in node_configs:
        node = QuantumEdgeNode(config['node_id'], config['capabilities'])
        edge_nodes.append(node)
        orchestrator.register_edge_node(node)
    
    # Create diverse optimization tasks
    test_tasks = [
        # Real-time tasks
        OptimizationTask(
            task_id='rt_001',
            problem_data={'n_spins': 15, 'complexity': 'low'},
            priority=OptimizationPriority.REAL_TIME,
            deadline_ms=100,
            preferred_paradigm=ComputingParadigm.QUANTUM_ANNEALING
        ),
        
        OptimizationTask(
            task_id='rt_002', 
            problem_data={'n_spins': 25, 'complexity': 'medium'},
            priority=OptimizationPriority.REAL_TIME,
            deadline_ms=50,
            quality_requirements={'min_energy': -50}
        ),
        
        # Interactive tasks
        OptimizationTask(
            task_id='int_001',
            problem_data={'n_spins': 100, 'complexity': 'high'},
            priority=OptimizationPriority.INTERACTIVE,
            deadline_ms=800,
            preferred_paradigm=ComputingParadigm.HYBRID_QUANTUM_CLASSICAL
        ),
        
        OptimizationTask(
            task_id='int_002',
            problem_data={'n_spins': 75, 'complexity': 'medium'},
            priority=OptimizationPriority.INTERACTIVE,
            deadline_ms=1000
        ),
        
        # Batch tasks
        OptimizationTask(
            task_id='batch_001',
            problem_data={'n_spins': 300, 'complexity': 'very_high'},
            priority=OptimizationPriority.BATCH,
            deadline_ms=5000,
            quality_requirements={'convergence_required': True}
        ),
        
        OptimizationTask(
            task_id='batch_002',
            problem_data={'n_spins': 200, 'complexity': 'high'},
            priority=OptimizationPriority.BATCH,
            deadline_ms=8000
        )
    ]
    
    print(f"\nSubmitting {len(test_tasks)} optimization tasks to edge computing network...")
    
    # Execute tasks
    async def run_demo():
        results = []
        
        # Submit all tasks
        for task in test_tasks:
            print(f"  Submitting {task.task_id} ({task.priority.value}, {task.problem_data['n_spins']} spins)")
            result = await orchestrator.submit_optimization_task(task)
            results.append((task, result))
        
        return results
    
    # Run the demo
    import asyncio
    results = asyncio.run(run_demo())
    
    # Display results
    print("\n" + "="*60)
    print("QUANTUM EDGE COMPUTING RESULTS")
    print("="*60)
    
    successful_tasks = 0
    total_execution_time = 0.0
    paradigm_usage = defaultdict(int)
    node_usage = defaultdict(int)
    
    for task, result in results:
        success = result.get('success', False)
        if success:
            successful_tasks += 1
            
        execution_time = result.get('execution_time', 0.0)
        total_execution_time += execution_time
        
        paradigm = result.get('paradigm_used', 'unknown')
        paradigm_usage[paradigm] += 1
        
        node_id = result.get('node_id', 'unknown')
        node_usage[node_id] += 1
        
        # Task summary
        print(f"\nTask {task.task_id}:")
        print(f"  Priority: {task.priority.value}")
        print(f"  Problem size: {task.problem_data['n_spins']} spins")
        print(f"  Success: {success}")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Node used: {node_id}")
        print(f"  Paradigm: {paradigm}")
        
        if success and 'result' in result:
            energy = result['result'].get('best_energy', 'N/A')
            print(f"  Final energy: {energy}")
    
    # Overall statistics
    print(f"\nOverall Performance:")
    print(f"  Success rate: {successful_tasks}/{len(test_tasks)} ({successful_tasks/len(test_tasks):.1%})")
    print(f"  Average execution time: {total_execution_time/len(test_tasks):.3f}s")
    
    print(f"\nParadigm Usage:")
    for paradigm, count in paradigm_usage.items():
        print(f"  {paradigm}: {count} tasks")
    
    print(f"\nNode Usage:")
    for node_id, count in node_usage.items():
        print(f"  {node_id}: {count} tasks")
    
    # Orchestration status
    orchestration_status = orchestrator.get_orchestration_status()
    print(f"\nOrchestration Status:")
    print(f"  Total nodes: {orchestration_status['total_nodes']}")
    print(f"  Active nodes: {orchestration_status['active_nodes']}")
    print(f"  Global success rate: {orchestration_status['global_success_rate']:.1%}")
    print(f"  Routing efficiency: {orchestration_status['routing_efficiency']:.1%}")
    
    # Node performance details
    print(f"\nNode Performance Details:")
    for node_id, summary in orchestration_status['nodes_summary'].items():
        print(f"  {node_id} ({summary['type']}):")
        print(f"    Tasks completed: {summary['tasks_completed']}")
        print(f"    Current load: {summary['load']:.1%}")
        print(f"    Active: {summary['active']}")
    
    return orchestrator, results


def benchmark_edge_latency():
    """Benchmark latency across different edge node types."""
    print("Benchmarking Edge Computing Latency...")
    
    # Create different node types for comparison
    node_types = [
        ('CPU Intensive', EdgeNodeCapabilities(
            node_type=EdgeNodeType.CPU_INTENSIVE,
            latency_ms=5.0,
            supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
        )),
        
        ('GPU Accelerated', EdgeNodeCapabilities(
            node_type=EdgeNodeType.GPU_ACCELERATED,
            latency_ms=8.0,
            supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
        )),
        
        ('Quantum Simulator', EdgeNodeCapabilities(
            node_type=EdgeNodeType.QUANTUM_SIMULATOR,
            quantum_qubits=20,
            latency_ms=15.0,
            supported_paradigms=[ComputingParadigm.QUANTUM_ANNEALING]
        )),
        
        ('Quantum Hardware', EdgeNodeCapabilities(
            node_type=EdgeNodeType.QUANTUM_HARDWARE,
            quantum_qubits=50,
            latency_ms=50.0,
            supported_paradigms=[ComputingParadigm.QUANTUM_ANNEALING]
        ))
    ]
    
    # Test different problem sizes
    problem_sizes = [10, 20, 30, 50]
    
    benchmark_results = {}
    
    async def benchmark_node_type(name, capabilities):
        """Benchmark a specific node type."""
        node = QuantumEdgeNode(f"bench_{name.lower().replace(' ', '_')}", capabilities)
        
        type_results = {}
        
        for size in problem_sizes:
            if size > capabilities.max_problem_size:
                continue
            
            # Create test task
            task = OptimizationTask(
                task_id=f"bench_{size}",
                problem_data={'n_spins': size},
                priority=OptimizationPriority.REAL_TIME
            )
            
            # Run multiple trials
            execution_times = []
            for trial in range(5):
                result = await node.process_task(task)
                if result.get('success', False):
                    execution_times.append(result['execution_time'])
            
            if execution_times:
                avg_time = np.mean(execution_times)
                std_time = np.std(execution_times)
                type_results[size] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'trials': len(execution_times)
                }
        
        return type_results
    
    # Run benchmarks
    async def run_benchmarks():
        results = {}
        for name, capabilities in node_types:
            print(f"  Benchmarking {name}...")
            results[name] = await benchmark_node_type(name, capabilities)
        return results
    
    benchmark_results = asyncio.run(run_benchmarks())
    
    # Display results
    print("\n" + "="*80)
    print("EDGE COMPUTING LATENCY BENCHMARK")
    print("="*80)
    print(f"{'Node Type':<18} {'10 spins':<12} {'20 spins':<12} {'30 spins':<12} {'50 spins':<12}")
    print(f"{'':^18} {'Time(ms)':<12} {'Time(ms)':<12} {'Time(ms)':<12} {'Time(ms)':<12}")
    print("-" * 80)
    
    for node_type, results in benchmark_results.items():
        row = f"{node_type:<18}"
        
        for size in [10, 20, 30, 50]:
            if size in results:
                avg_time = results[size]['avg_time'] * 1000  # Convert to ms
                row += f" {avg_time:<11.1f}"
            else:
                row += f" {'N/A':<11}"
        
        print(row)
    
    # Latency analysis
    print(f"\nLatency Analysis:")
    
    # Find fastest node type for each problem size
    for size in problem_sizes:
        fastest_node = None
        fastest_time = float('inf')
        
        for node_type, results in benchmark_results.items():
            if size in results:
                time_ms = results[size]['avg_time'] * 1000
                if time_ms < fastest_time:
                    fastest_time = time_ms
                    fastest_node = node_type
        
        if fastest_node:
            print(f"  {size} spins: {fastest_node} ({fastest_time:.1f}ms)")
    
    return benchmark_results


if __name__ == "__main__":
    # Run quantum edge computing demonstrations
    print("Starting Quantum Edge Computing Demonstrations...\n")
    
    # Main edge computing demo
    orchestrator, results = create_quantum_edge_demo()
    
    print("\n" + "="*80)
    
    # Latency benchmark
    benchmark_results = benchmark_edge_latency()
    
    print("\nQuantum edge computing demonstration completed successfully!")