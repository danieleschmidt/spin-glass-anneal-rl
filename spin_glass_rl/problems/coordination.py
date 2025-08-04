"""Multi-agent coordination problem implementations."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum

from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution
from spin_glass_rl.core.ising_model import IsingModel


class AgentType(Enum):
    """Types of agents in coordination problems."""
    ROBOT = "robot"
    DRONE = "drone"
    VEHICLE = "vehicle"
    WORKER = "worker"
    SENSOR = "sensor"


@dataclass
class CoordinationAgent:
    """Agent in coordination problem."""
    id: int
    name: str
    agent_type: AgentType
    position: Tuple[float, float]
    capabilities: List[str]
    capacity: float = 1.0
    speed: float = 1.0
    communication_range: float = 10.0
    energy: float = 100.0


@dataclass
class CoordinationTask:
    """Task requiring agent coordination."""
    id: int
    name: str
    location: Tuple[float, float]
    required_capabilities: List[str]
    required_agents: int = 1
    duration: float = 1.0
    priority: float = 1.0
    deadline: Optional[float] = None
    dependencies: List[int] = None  # Task IDs that must complete first
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class CoordinationProblem(ProblemTemplate):
    """
    Multi-agent coordination problem.
    
    Coordinates multiple agents to complete tasks while
    considering communication constraints, dependencies,
    and spatial relationships.
    """
    
    def __init__(self, name: str = "Multi-Agent Coordination"):
        super().__init__(name)
        self.agents: List[CoordinationAgent] = []
        self.tasks: List[CoordinationTask] = []
        self.communication_graph: Optional[np.ndarray] = None
        self.time_horizon: float = 100.0
        self.time_discretization: int = 50
        self._variable_mapping = {}
    
    def add_agent(self, agent: CoordinationAgent) -> None:
        """Add agent to coordination problem."""
        self.agents.append(agent)
        self._invalidate_communication_graph()
    
    def add_task(self, task: CoordinationTask) -> None:
        """Add task to coordination problem."""
        self.tasks.append(task)
    
    def compute_communication_graph(self) -> np.ndarray:
        """Compute communication graph based on agent positions and ranges."""
        n_agents = len(self.agents)
        graph = np.zeros((n_agents, n_agents))
        
        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if i != j:
                    # Calculate distance
                    dx = agent_i.position[0] - agent_j.position[0]
                    dy = agent_i.position[1] - agent_j.position[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Check if within communication range
                    max_range = min(agent_i.communication_range, agent_j.communication_range)
                    if distance <= max_range:
                        graph[i, j] = 1.0
        
        self.communication_graph = graph
        return graph
    
    def _invalidate_communication_graph(self) -> None:
        """Invalidate cached communication graph."""
        self.communication_graph = None
    
    def can_communicate(self, agent_i: int, agent_j: int) -> bool:
        """Check if two agents can communicate."""
        if self.communication_graph is None:
            self.compute_communication_graph()
        return self.communication_graph[agent_i, agent_j] > 0
    
    def get_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def agent_can_perform_task(self, agent: CoordinationAgent, task: CoordinationTask) -> bool:
        """Check if agent has required capabilities for task."""
        return all(cap in agent.capabilities for cap in task.required_capabilities)
    
    def encode_to_ising(
        self,
        objective: str = "minimize_completion_time",
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """
        Encode coordination problem as Ising model.
        
        Variables: x_{i,j,t} = 1 if agent i performs task j starting at time t
        """
        if not self.agents or not self.tasks:
            raise ValueError("Must add agents and tasks before encoding")
        
        if penalty_weights is None:
            penalty_weights = {
                "task_assignment": 100.0,
                "agent_capacity": 75.0,
                "communication": 50.0,
                "capability": 200.0,
                "dependencies": 150.0
            }
        
        n_agents = len(self.agents)
        n_tasks = len(self.tasks)
        n_time_slots = self.time_discretization
        
        # Binary variables: x_{i,j,t} for agent i, task j, time t
        n_spins = n_agents * n_tasks * n_time_slots
        
        self.ising_model = self.create_ising_model(n_spins)
        
        # Create variable mapping
        self._create_variable_mapping(n_agents, n_tasks, n_time_slots)
        
        # Add objective function
        self._add_coordination_objective(objective)
        
        # Add constraints
        self._add_task_assignment_constraints(penalty_weights["task_assignment"])
        self._add_agent_capacity_constraints(penalty_weights["agent_capacity"])
        self._add_capability_constraints(penalty_weights["capability"])
        self._add_dependency_constraints(penalty_weights["dependencies"])
        
        if penalty_weights.get("communication", 0) > 0:
            self._add_communication_constraints(penalty_weights["communication"])
        
        return self.ising_model
    
    def _create_variable_mapping(self, n_agents: int, n_tasks: int, n_time_slots: int) -> None:
        """Create mapping from (agent, task, time) to spin index."""
        self._variable_mapping = {}
        spin_idx = 0
        
        for agent_id in range(n_agents):
            for task_id in range(n_tasks):
                for time_slot in range(n_time_slots):
                    key = (agent_id, task_id, time_slot)
                    self._variable_mapping[key] = spin_idx
                    spin_idx += 1
    
    def _get_spin_index(self, agent_id: int, task_id: int, time_slot: int) -> int:
        """Get spin index for agent-task-time assignment."""
        return self._variable_mapping[(agent_id, task_id, time_slot)]
    
    def _add_coordination_objective(self, objective: str) -> None:
        """Add coordination objective to Ising model."""
        if objective == "minimize_completion_time":
            self._add_completion_time_objective()
        elif objective == "minimize_travel_cost":
            self._add_travel_cost_objective()
        elif objective == "maximize_task_priority":
            self._add_priority_objective()
        elif objective == "balance_workload":
            self._add_workload_balancing_objective()
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _add_completion_time_objective(self) -> None:
        """Minimize total completion time."""
        time_step = self.time_horizon / self.time_discretization
        
        for agent_id in range(len(self.agents)):
            for task_id, task in enumerate(self.tasks):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                    
                    # Completion time for this assignment
                    completion_time = (time_slot + 1) * time_step + task.duration
                    
                    # Weight by task priority
                    weight = completion_time * task.priority
                    
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field + weight)
    
    def _add_travel_cost_objective(self) -> None:
        """Minimize agent travel costs."""
        for agent_id, agent in enumerate(self.agents):
            for task_id, task in enumerate(self.tasks):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                    
                    # Travel cost from agent position to task location
                    travel_distance = self.get_distance(agent.position, task.location)
                    travel_cost = travel_distance / agent.speed
                    
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field + travel_cost)
    
    def _add_priority_objective(self) -> None:
        """Maximize task priority (minimize negative priority)."""
        for agent_id in range(len(self.agents)):
            for task_id, task in enumerate(self.tasks):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                    
                    # Negative priority for maximization
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field - task.priority)
    
    def _add_workload_balancing_objective(self) -> None:
        """Balance workload across agents."""
        # Add quadratic terms to penalize uneven workload distribution
        for agent_i in range(len(self.agents)):
            for agent_j in range(agent_i + 1, len(self.agents)):
                for task_i in range(len(self.tasks)):
                    for task_j in range(len(self.tasks)):
                        for time_i in range(self.time_discretization):
                            for time_j in range(self.time_discretization):
                                if task_i != task_j:  # Different tasks
                                    spin_i = self._get_spin_index(agent_i, task_i, time_i)
                                    spin_j = self._get_spin_index(agent_j, task_j, time_j)
                                    
                                    # Encourage balanced assignment
                                    balance_weight = 0.1
                                    current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                                    self.ising_model.set_coupling(spin_i, spin_j, 
                                                                current_coupling - balance_weight)
    
    def _add_task_assignment_constraints(self, penalty_weight: float) -> None:
        """Each task assigned to required number of agents."""
        for task_id, task in enumerate(self.tasks):
            # Get all possible agent-time assignments for this task
            task_spins = []
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                    task_spins.append(spin_idx)
            
            # Task requires exactly 'required_agents' assignments
            self.constraint_encoder.add_cardinality_constraint(
                task_spins,
                k=task.required_agents,
                penalty_weight=penalty_weight,
                description=f"Task {task_id} assignment"
            )
    
    def _add_agent_capacity_constraints(self, penalty_weight: float) -> None:
        """Each agent can perform at most one task at a time."""
        for agent_id in range(len(self.agents)):
            for time_slot in range(self.time_discretization):
                # Get all tasks this agent could be doing at this time
                agent_time_spins = []
                
                for task_id, task in enumerate(self.tasks):
                    # Check if task could overlap with this time slot
                    task_duration_slots = int(np.ceil(task.duration * self.time_discretization / self.time_horizon))
                    
                    for start_slot in range(max(0, time_slot - task_duration_slots + 1),
                                          min(self.time_discretization, time_slot + 1)):
                        if start_slot + task_duration_slots > time_slot:
                            spin_idx = self._get_spin_index(agent_id, task_id, start_slot)
                            agent_time_spins.append(spin_idx)
                
                # Agent can do at most one task at this time
                if len(agent_time_spins) > 1:
                    self.constraint_encoder.add_cardinality_constraint(
                        agent_time_spins,
                        k=1,
                        penalty_weight=penalty_weight,
                        description=f"Agent {agent_id} capacity at time {time_slot}"
                    )
    
    def _add_capability_constraints(self, penalty_weight: float) -> None:
        """Agents can only perform tasks they have capabilities for."""
        for agent_id, agent in enumerate(self.agents):
            for task_id, task in enumerate(self.tasks):
                if not self.agent_can_perform_task(agent, task):
                    # Add large penalty for impossible assignments
                    for time_slot in range(self.time_discretization):
                        spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                        
                        current_field = self.ising_model.external_fields[spin_idx].item()
                        self.ising_model.set_external_field(spin_idx, 
                                                          current_field + penalty_weight)
    
    def _add_dependency_constraints(self, penalty_weight: float) -> None:
        """Task dependencies: some tasks must complete before others."""
        time_step = self.time_horizon / self.time_discretization
        
        for task_id, task in enumerate(self.tasks):
            for dep_task_id in task.dependencies:
                if dep_task_id < len(self.tasks):
                    dep_task = self.tasks[dep_task_id]
                    
                    # Dependency task must complete before current task starts
                    for agent_i in range(len(self.agents)):
                        for agent_j in range(len(self.agents)):
                            for time_i in range(self.time_discretization):
                                for time_j in range(self.time_discretization):
                                    # Check if timing violates dependency
                                    dep_completion_time = (time_i + 1) * time_step + dep_task.duration
                                    task_start_time = time_j * time_step
                                    
                                    if task_start_time < dep_completion_time:
                                        # Violation: task starts before dependency completes
                                        spin_i = self._get_spin_index(agent_i, dep_task_id, time_i)
                                        spin_j = self._get_spin_index(agent_j, task_id, time_j)
                                        
                                        # Add penalty coupling
                                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                                    current_coupling + penalty_weight)
    
    def _add_communication_constraints(self, penalty_weight: float) -> None:
        """Communication constraints for coordinated tasks."""
        if self.communication_graph is None:
            self.compute_communication_graph()
        
        for task_id, task in enumerate(self.tasks):
            if task.required_agents > 1:
                # Multi-agent tasks require communication between assigned agents
                for agent_i in range(len(self.agents)):
                    for agent_j in range(agent_i + 1, len(self.agents)):
                        if not self.can_communicate(agent_i, agent_j):
                            # Penalize simultaneous assignment to non-communicating agents
                            for time_i in range(self.time_discretization):
                                for time_j in range(self.time_discretization):
                                    if abs(time_i - time_j) <= 1:  # Simultaneous or adjacent
                                        spin_i = self._get_spin_index(agent_i, task_id, time_i)
                                        spin_j = self._get_spin_index(agent_j, task_id, time_j)
                                        
                                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                                    current_coupling + penalty_weight)
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode coordination solution from spins."""
        binary_spins = (spins + 1) // 2
        
        # Extract agent-task assignments
        assignments = {}  # agent_id -> [(task_id, start_time, end_time)]
        task_assignments = {}  # task_id -> [(agent_id, start_time, end_time)]
        
        time_step = self.time_horizon / self.time_discretization
        
        for agent_id in range(len(self.agents)):
            assignments[agent_id] = []
        
        for task_id in range(len(self.tasks)):
            task_assignments[task_id] = []
        
        for agent_id in range(len(self.agents)):
            for task_id in range(len(self.tasks)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(agent_id, task_id, time_slot)
                    
                    if binary_spins[spin_idx].item() == 1:
                        start_time = time_slot * time_step
                        end_time = start_time + self.tasks[task_id].duration
                        
                        assignments[agent_id].append((task_id, start_time, end_time))
                        task_assignments[task_id].append((agent_id, start_time, end_time))
        
        # Calculate objective value
        objective_value = self._calculate_coordination_objective(assignments, task_assignments)
        
        # Check constraints
        constraint_violations = self._check_coordination_constraints(assignments, task_assignments)
        is_feasible = all(v == 0 for v in constraint_violations.values())
        
        # Calculate coordination metrics
        completed_tasks = sum(1 for assigns in task_assignments.values() if len(assigns) > 0)
        makespan = self._calculate_makespan(assignments)
        
        return ProblemSolution(
            variables={
                "agent_assignments": assignments,
                "task_assignments": task_assignments
            },
            objective_value=objective_value,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "completed_tasks": completed_tasks,
                "completion_rate": completed_tasks / len(self.tasks),
                "makespan": makespan,
                "total_travel_time": self._calculate_total_travel_time(assignments),
                "communication_violations": constraint_violations.get("communication", 0)
            }
        )
    
    def _calculate_coordination_objective(self, assignments: Dict, task_assignments: Dict) -> float:
        """Calculate objective value for coordination solution."""
        # Default to makespan minimization
        return self._calculate_makespan(assignments)
    
    def _calculate_makespan(self, assignments: Dict) -> float:
        """Calculate makespan (latest task completion time)."""
        max_completion = 0.0
        for agent_assignments in assignments.values():
            for _, start_time, end_time in agent_assignments:
                max_completion = max(max_completion, end_time)
        return max_completion
    
    def _calculate_total_travel_time(self, assignments: Dict) -> float:
        """Calculate total travel time for all agents."""
        total_travel = 0.0
        
        for agent_id, agent_assignments in assignments.items():
            if not agent_assignments:
                continue
            
            agent = self.agents[agent_id]
            current_pos = agent.position
            
            # Sort by start time
            sorted_assignments = sorted(agent_assignments, key=lambda x: x[1])
            
            for task_id, start_time, _ in sorted_assignments:
                task = self.tasks[task_id]
                travel_distance = self.get_distance(current_pos, task.location)
                travel_time = travel_distance / agent.speed
                total_travel += travel_time
                current_pos = task.location
        
        return total_travel
    
    def _check_coordination_constraints(self, assignments: Dict, task_assignments: Dict) -> Dict[str, float]:
        """Check constraint violations."""
        violations = {
            "unassigned_tasks": 0.0,
            "agent_capacity": 0.0,
            "capability": 0.0,
            "dependencies": 0.0,
            "communication": 0.0
        }
        
        # Check unassigned tasks
        for task_id, task in enumerate(self.tasks):
            assigned_agents = len(task_assignments.get(task_id, []))
            if assigned_agents < task.required_agents:
                violations["unassigned_tasks"] += task.required_agents - assigned_agents
        
        # Check agent capacity (overlapping tasks)
        for agent_id, agent_assignments in assignments.items():
            sorted_assignments = sorted(agent_assignments, key=lambda x: x[1])
            for i in range(len(sorted_assignments) - 1):
                _, _, end_time_i = sorted_assignments[i]
                _, start_time_j, _ = sorted_assignments[i + 1]
                if end_time_i > start_time_j:  # Overlap
                    violations["agent_capacity"] += 1.0
        
        # Check capability constraints
        for agent_id, agent_assignments in assignments.items():
            agent = self.agents[agent_id]
            for task_id, _, _ in agent_assignments:
                task = self.tasks[task_id]
                if not self.agent_can_perform_task(agent, task):
                    violations["capability"] += 1.0
        
        return violations
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate coordination solution."""
        return solution.is_feasible
    
    def generate_random_instance(
        self,
        n_agents: int = 5,
        n_tasks: int = 10,
        area_size: float = 100.0,
        **kwargs
    ) -> Dict:
        """Generate random coordination instance."""
        # Clear existing data
        self.agents = []
        self.tasks = []
        
        # Define capability types
        capabilities = ["sensing", "manipulation", "navigation", "communication", "computation"]
        
        # Generate random agents
        for i in range(n_agents):
            agent_caps = np.random.choice(capabilities, size=np.random.randint(1, 4), replace=False).tolist()
            
            agent = CoordinationAgent(
                id=i,
                name=f"Agent_{i}",
                agent_type=np.random.choice(list(AgentType)),
                position=(np.random.uniform(0, area_size), np.random.uniform(0, area_size)),
                capabilities=agent_caps,
                capacity=np.random.uniform(0.5, 2.0),
                speed=np.random.uniform(1.0, 5.0),
                communication_range=np.random.uniform(15.0, 30.0),
                energy=np.random.uniform(50.0, 100.0)
            )
            self.add_agent(agent)
        
        # Generate random tasks
        for i in range(n_tasks):
            required_caps = np.random.choice(capabilities, size=np.random.randint(1, 3), replace=False).tolist()
            
            task = CoordinationTask(
                id=i,
                name=f"Task_{i}",
                location=(np.random.uniform(0, area_size), np.random.uniform(0, area_size)),
                required_capabilities=required_caps,
                required_agents=np.random.randint(1, min(3, n_agents + 1)),
                duration=np.random.uniform(1.0, 10.0),
                priority=np.random.uniform(0.5, 2.0),
                deadline=np.random.uniform(20.0, self.time_horizon) if np.random.rand() > 0.3 else None
            )
            
            # Add dependencies (simple chain)
            if i > 0 and np.random.rand() < 0.3:
                task.dependencies = [np.random.randint(0, i)]
            
            self.add_task(task)
        
        return {
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "area_size": area_size,
            "time_horizon": self.time_horizon
        }
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None, **plot_params) -> None:
        """Plot coordination solution."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            assignments = solution.variables["agent_assignments"]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Spatial view with agents, tasks, and communication
            agent_colors = plt.cm.Set1(np.linspace(0, 1, len(self.agents)))
            
            # Plot communication links
            if self.communication_graph is not None:
                for i in range(len(self.agents)):
                    for j in range(i + 1, len(self.agents)):
                        if self.communication_graph[i, j] > 0:
                            agent_i, agent_j = self.agents[i], self.agents[j]
                            ax1.plot([agent_i.position[0], agent_j.position[0]], 
                                   [agent_i.position[1], agent_j.position[1]], 
                                   'k--', alpha=0.3, linewidth=0.5)
            
            # Plot agents
            for i, agent in enumerate(self.agents):
                ax1.scatter(agent.position[0], agent.position[1], 
                           c=[agent_colors[i]], s=200, marker='o', 
                           label=f'Agent {i}', edgecolors='black')
                
                # Communication range
                circle = Circle(agent.position, agent.communication_range, 
                              fill=False, linestyle='--', alpha=0.3, color=agent_colors[i])
                ax1.add_patch(circle)
            
            # Plot tasks
            for task in self.tasks:
                ax1.scatter(task.location[0], task.location[1], 
                           c='red', s=100, marker='s', edgecolors='black')
                ax1.annotate(f'T{task.id}', task.location, xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Agent-Task Spatial Layout')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Timeline/Gantt chart
            y_pos = 0
            y_labels = []
            
            for agent_id, agent_assignments in assignments.items():
                if agent_assignments:
                    y_labels.append(f'Agent {agent_id}')
                    
                    for task_id, start_time, end_time in agent_assignments:
                        duration = end_time - start_time
                        ax2.barh(y_pos, duration, left=start_time, 
                                height=0.6, alpha=0.7, 
                                color=agent_colors[agent_id])
                        ax2.text(start_time + duration/2, y_pos, f'T{task_id}',
                               ha='center', va='center', fontsize=8, fontweight='bold')
                    
                    y_pos += 1
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Agents')
            ax2.set_yticks(range(len(y_labels)))
            ax2.set_yticklabels(y_labels)
            ax2.set_title('Task Assignment Timeline')
            ax2.grid(True, alpha=0.3)
            
            # 3. Task completion statistics
            task_assignments = solution.variables["task_assignments"]
            completed = []
            incomplete = []
            
            for task_id, task in enumerate(self.tasks):
                assigned_agents = len(task_assignments.get(task_id, []))
                if assigned_agents >= task.required_agents:
                    completed.append(task_id)
                else:
                    incomplete.append(task_id)
            
            labels = ['Completed', 'Incomplete']
            sizes = [len(completed), len(incomplete)]
            colors = ['lightgreen', 'lightcoral']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Task Completion Status')
            
            # 4. Agent workload distribution
            workloads = []
            for agent_id in range(len(self.agents)):
                agent_assignments = assignments.get(agent_id, [])
                total_work_time = sum(end - start for _, start, end in agent_assignments)
                workloads.append(total_work_time)
            
            ax4.bar(range(len(workloads)), workloads, 
                   color=[agent_colors[i] for i in range(len(workloads))], alpha=0.7)
            ax4.set_xlabel('Agent ID')
            ax4.set_ylabel('Total Work Time')
            ax4.set_title('Agent Workload Distribution')
            ax4.set_xticks(range(len(workloads)))
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            super().plot_solution(solution, save_path, **plot_params)