"""Scheduling problem implementations."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass

from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution
from spin_glass_rl.core.ising_model import IsingModel


@dataclass
class Task:
    """Task definition for scheduling problems."""
    id: int
    duration: float
    release_time: float = 0.0
    due_date: Optional[float] = None
    priority: float = 1.0
    resource_requirements: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {}


@dataclass
class Agent:
    """Agent/machine definition for scheduling problems."""
    id: int
    name: str
    capacity: Dict[str, float] = None
    availability_windows: List[Tuple[float, float]] = None
    cost_per_hour: float = 1.0
    
    def __post_init__(self):
        if self.capacity is None:
            self.capacity = {}
        if self.availability_windows is None:
            self.availability_windows = [(0.0, float('inf'))]


class SchedulingProblem(ProblemTemplate):
    """
    Generic multi-agent scheduling problem.
    
    Assigns tasks to agents/machines while minimizing makespan,
    total completion time, or other scheduling objectives.
    """
    
    def __init__(self, name: str = "Multi-Agent Scheduling"):
        super().__init__(name)
        self.tasks: List[Task] = []
        self.agents: List[Agent] = []
        self.time_horizon: float = 100.0
        self.time_discretization: int = 100
        self._variable_mapping = {}
    
    def add_task(self, task: Task) -> None:
        """Add task to scheduling problem."""
        self.tasks.append(task)
    
    def add_agent(self, agent: Agent) -> None:
        """Add agent to scheduling problem."""
        self.agents.append(agent)
    
    def encode_to_ising(
        self,
        objective: str = "makespan",
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """
        Encode scheduling problem as Ising model.
        
        Args:
            objective: Objective function ("makespan", "total_time", "weighted_completion")
            penalty_weights: Weights for constraint penalties
            
        Returns:
            IsingModel representation
        """
        if not self.tasks or not self.agents:
            raise ValueError("Must add tasks and agents before encoding")
        
        # Default penalty weights
        if penalty_weights is None:
            penalty_weights = {
                "assignment": 100.0,
                "capacity": 50.0,
                "precedence": 75.0,
                "time_window": 60.0
            }
        
        # Create binary variables: x_{i,j,t} = 1 if task i assigned to agent j starting at time t
        n_tasks = len(self.tasks)
        n_agents = len(self.agents)
        n_time_slots = self.time_discretization
        
        # Total number of spins
        n_spins = n_tasks * n_agents * n_time_slots
        
        # Create Ising model
        self.ising_model = self.create_ising_model(n_spins)
        
        # Create variable mapping
        self._create_variable_mapping(n_tasks, n_agents, n_time_slots)
        
        # Add objective function
        self._add_objective_to_ising(objective)
        
        # Add constraints
        self._add_assignment_constraints(penalty_weights["assignment"])
        self._add_capacity_constraints(penalty_weights["capacity"])
        
        if "precedence" in penalty_weights:
            self._add_precedence_constraints(penalty_weights["precedence"])
        
        if "time_window" in penalty_weights:
            self._add_time_window_constraints(penalty_weights["time_window"])
        
        return self.ising_model
    
    def _create_variable_mapping(self, n_tasks: int, n_agents: int, n_time_slots: int) -> None:
        """Create mapping from (task, agent, time) to spin index."""
        self._variable_mapping = {}
        spin_idx = 0
        
        for task_id in range(n_tasks):
            for agent_id in range(n_agents):
                for time_slot in range(n_time_slots):
                    key = (task_id, agent_id, time_slot)
                    self._variable_mapping[key] = spin_idx
                    spin_idx += 1
    
    def _get_spin_index(self, task_id: int, agent_id: int, time_slot: int) -> int:
        """Get spin index for given task, agent, time assignment."""
        return self._variable_mapping[(task_id, agent_id, time_slot)]
    
    def _add_objective_to_ising(self, objective: str) -> None:
        """Add objective function to Ising model."""
        if objective == "makespan":
            self._add_makespan_objective()
        elif objective == "total_time":
            self._add_total_completion_time_objective()
        elif objective == "weighted_completion":
            self._add_weighted_completion_time_objective()
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _add_makespan_objective(self) -> None:
        """Add makespan minimization objective."""
        # Makespan = max completion time
        # Approximate with quadratic penalty on late completions
        time_step = self.time_horizon / self.time_discretization
        
        for task_id, task in enumerate(self.tasks):
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                    
                    # Completion time for this assignment
                    completion_time = (time_slot + 1) * time_step + task.duration
                    
                    # Linear penalty for later completion times
                    penalty = completion_time * 0.1
                    
                    # Add to external field (linear term)
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field + penalty)
    
    def _add_total_completion_time_objective(self) -> None:
        """Add total completion time minimization."""
        time_step = self.time_horizon / self.time_discretization
        
        for task_id, task in enumerate(self.tasks):
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                    
                    completion_time = (time_slot + 1) * time_step + task.duration
                    weight = completion_time * task.priority
                    
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field + weight)
    
    def _add_weighted_completion_time_objective(self) -> None:
        """Add weighted completion time minimization."""
        time_step = self.time_horizon / self.time_discretization
        
        for task_id, task in enumerate(self.tasks):
            weight = task.priority
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                    
                    completion_time = (time_slot + 1) * time_step + task.duration
                    objective_contribution = weight * completion_time
                    
                    current_field = self.ising_model.external_fields[spin_idx].item()
                    self.ising_model.set_external_field(spin_idx, current_field + objective_contribution)
    
    def _add_assignment_constraints(self, penalty_weight: float) -> None:
        """Each task assigned to exactly one agent at one time."""
        for task_id in range(len(self.tasks)):
            # Get all spins for this task
            task_spins = []
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                    task_spins.append(spin_idx)
            
            # Add cardinality constraint: exactly one spin should be +1
            self.constraint_encoder.add_cardinality_constraint(
                task_spins, 
                k=1, 
                penalty_weight=penalty_weight,
                description=f"Task {task_id} assignment"
            )
    
    def _add_capacity_constraints(self, penalty_weight: float) -> None:
        """Agent capacity constraints."""
        for agent_id, agent in enumerate(self.agents):
            for time_slot in range(self.time_discretization):
                # Get all tasks that could be running at this time slot
                conflicting_spins = []
                
                for task_id, task in enumerate(self.tasks):
                    # Check if task overlaps with this time slot
                    task_duration_slots = int(np.ceil(task.duration * self.time_discretization / self.time_horizon))
                    
                    for start_slot in range(max(0, time_slot - task_duration_slots + 1), 
                                          min(self.time_discretization, time_slot + 1)):
                        if start_slot + task_duration_slots > time_slot:
                            spin_idx = self._get_spin_index(task_id, agent_id, start_slot)
                            conflicting_spins.append(spin_idx)
                
                # Add capacity constraint (at most one task per agent per time)
                if len(conflicting_spins) > 1:
                    self.constraint_encoder.add_cardinality_constraint(
                        conflicting_spins,
                        k=1,
                        penalty_weight=penalty_weight,
                        description=f"Agent {agent_id} capacity at time {time_slot}"
                    )
    
    def _add_precedence_constraints(self, penalty_weight: float) -> None:
        """Precedence constraints between tasks."""
        # Simple precedence: task i must complete before task j starts
        # This is a simplified version - full implementation would handle complex precedence graphs
        
        for i, task_i in enumerate(self.tasks):
            for j, task_j in enumerate(self.tasks):
                if i < j:  # Simple ordering constraint for demonstration
                    # Task i should complete before task j starts
                    for agent_i in range(len(self.agents)):
                        for agent_j in range(len(self.agents)):
                            for time_i in range(self.time_discretization):
                                for time_j in range(self.time_discretization):
                                    if time_j <= time_i:  # Violation
                                        spin_i = self._get_spin_index(i, agent_i, time_i)
                                        spin_j = self._get_spin_index(j, agent_j, time_j)
                                        
                                        # Add penalty for simultaneous assignment
                                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                                    current_coupling + penalty_weight)
    
    def _add_time_window_constraints(self, penalty_weight: float) -> None:
        """Time window constraints for tasks."""
        time_step = self.time_horizon / self.time_discretization
        
        for task_id, task in enumerate(self.tasks):
            if task.due_date is not None:
                due_slot = int(task.due_date / time_step)
                
                for agent_id in range(len(self.agents)):
                    for time_slot in range(due_slot, self.time_discretization):
                        # Penalize assignments that violate due dates
                        spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                        
                        current_field = self.ising_model.external_fields[spin_idx].item()
                        self.ising_model.set_external_field(spin_idx, 
                                                          current_field + penalty_weight)
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode spin configuration to scheduling solution."""
        # Convert Ising spins {-1, +1} to binary {0, 1}
        binary_spins = (spins + 1) // 2
        
        # Extract task assignments
        assignments = {}
        schedule = {}
        
        time_step = self.time_horizon / self.time_discretization
        
        for task_id in range(len(self.tasks)):
            assignments[task_id] = None
            
            for agent_id in range(len(self.agents)):
                for time_slot in range(self.time_discretization):
                    spin_idx = self._get_spin_index(task_id, agent_id, time_slot)
                    
                    if binary_spins[spin_idx].item() == 1:
                        start_time = time_slot * time_step
                        end_time = start_time + self.tasks[task_id].duration
                        
                        assignments[task_id] = {
                            "agent_id": agent_id,
                            "start_time": start_time,
                            "end_time": end_time
                        }
                        
                        if agent_id not in schedule:
                            schedule[agent_id] = []
                        
                        schedule[agent_id].append({
                            "task_id": task_id,
                            "start_time": start_time,
                            "end_time": end_time
                        })
        
        # Calculate objective value
        objective_value = self._calculate_objective_value(assignments)
        
        # Check feasibility
        constraint_violations = self._check_constraints(assignments)
        is_feasible = all(v == 0 for v in constraint_violations.values())
        
        return ProblemSolution(
            variables={"assignments": assignments, "schedule": schedule},
            objective_value=objective_value,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "makespan": self._calculate_makespan(assignments),
                "total_completion_time": self._calculate_total_completion_time(assignments),
                "n_assigned_tasks": sum(1 for a in assignments.values() if a is not None)
            }
        )
    
    def _calculate_objective_value(self, assignments: Dict) -> float:
        """Calculate objective value for given assignments."""
        return self._calculate_makespan(assignments)  # Default to makespan
    
    def _calculate_makespan(self, assignments: Dict) -> float:
        """Calculate makespan (maximum completion time)."""
        max_completion = 0.0
        for assignment in assignments.values():
            if assignment is not None:
                max_completion = max(max_completion, assignment["end_time"])
        return max_completion
    
    def _calculate_total_completion_time(self, assignments: Dict) -> float:
        """Calculate total completion time."""
        total_time = 0.0
        for assignment in assignments.values():
            if assignment is not None:
                total_time += assignment["end_time"]
        return total_time
    
    def _check_constraints(self, assignments: Dict) -> Dict[str, float]:
        """Check constraint violations."""
        violations = {
            "unassigned_tasks": 0.0,
            "multiple_assignments": 0.0,
            "capacity_violations": 0.0,
            "time_window_violations": 0.0
        }
        
        # Check assignment constraints
        for task_id, assignment in assignments.items():
            if assignment is None:
                violations["unassigned_tasks"] += 1.0
        
        # Check capacity constraints (simplified)
        agent_schedules = {}
        for task_id, assignment in assignments.items():
            if assignment is not None:
                agent_id = assignment["agent_id"]
                if agent_id not in agent_schedules:
                    agent_schedules[agent_id] = []
                agent_schedules[agent_id].append((assignment["start_time"], assignment["end_time"]))
        
        for agent_id, schedule in agent_schedules.items():
            # Check for overlapping tasks
            schedule.sort()  # Sort by start time
            for i in range(len(schedule) - 1):
                if schedule[i][1] > schedule[i + 1][0]:  # Overlap
                    violations["capacity_violations"] += 1.0
        
        # Check time window constraints
        for task_id, assignment in assignments.items():
            if assignment is not None:
                task = self.tasks[task_id]
                if task.due_date is not None and assignment["end_time"] > task.due_date:
                    violations["time_window_violations"] += 1.0
        
        return violations
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate solution feasibility."""
        return solution.is_feasible
    
    def generate_random_instance(
        self,
        n_tasks: int = 10,
        n_agents: int = 3,
        time_horizon: float = 100.0,
        **kwargs
    ) -> Dict:
        """Generate random scheduling instance."""
        # Clear existing tasks and agents
        self.tasks = []
        self.agents = []
        self.time_horizon = time_horizon
        
        # Generate random tasks
        for i in range(n_tasks):
            task = Task(
                id=i,
                duration=np.random.uniform(1.0, 10.0),
                release_time=np.random.uniform(0.0, time_horizon * 0.2),
                due_date=np.random.uniform(time_horizon * 0.5, time_horizon) if np.random.rand() > 0.3 else None,
                priority=np.random.uniform(0.5, 2.0)
            )
            self.add_task(task)
        
        # Generate random agents
        for i in range(n_agents):
            agent = Agent(
                id=i,
                name=f"Agent_{i}",
                cost_per_hour=np.random.uniform(10.0, 50.0)
            )
            self.add_agent(agent)
        
        return {
            "n_tasks": n_tasks,
            "n_agents": n_agents,
            "time_horizon": time_horizon
        }
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None, **plot_params) -> None:
        """Plot Gantt chart of scheduling solution."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            schedule = solution.variables["schedule"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.tasks)))
            
            y_pos = 0
            y_labels = []
            
            for agent_id in sorted(schedule.keys()):
                y_labels.append(f"Agent {agent_id}")
                
                for task_info in schedule[agent_id]:
                    task_id = task_info["task_id"]
                    start_time = task_info["start_time"]
                    duration = task_info["end_time"] - start_time
                    
                    # Draw task rectangle
                    rect = patches.Rectangle(
                        (start_time, y_pos - 0.4),
                        duration, 0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[task_id],
                        alpha=0.7
                    )
                    ax.add_patch(rect)
                    
                    # Add task label
                    ax.text(start_time + duration/2, y_pos, f'T{task_id}',
                           ha='center', va='center', fontsize=8, fontweight='bold')
                
                y_pos += 1
            
            ax.set_xlim(0, self.time_horizon)
            ax.set_ylim(-0.5, len(schedule))
            ax.set_xlabel('Time')
            ax.set_ylabel('Agents')
            ax.set_yticks(range(len(schedule)))
            ax.set_yticklabels(y_labels)
            ax.set_title(f'Schedule (Makespan: {solution.metadata["makespan"]:.2f})')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            super().plot_solution(solution, save_path, **plot_params)


class JobShopScheduling(SchedulingProblem):
    """
    Job shop scheduling problem.
    
    Specialized scheduling problem where jobs consist of sequences
    of operations that must be processed on specific machines.
    """
    
    def __init__(self):
        super().__init__("Job Shop Scheduling")
        self.jobs = []
        self.machines = []
    
    def add_job(self, operations: List[Tuple[int, float]]) -> None:
        """
        Add job with sequence of operations.
        
        Args:
            operations: List of (machine_id, processing_time) tuples
        """
        job_id = len(self.jobs)
        self.jobs.append(operations)
        
        # Convert to tasks with precedence constraints
        for op_id, (machine_id, duration) in enumerate(operations):
            task = Task(
                id=len(self.tasks),
                duration=duration,
                priority=1.0
            )
            # Store job and operation info in metadata
            task.metadata = {"job_id": job_id, "operation_id": op_id, "machine_id": machine_id}
            self.add_task(task)
    
    def add_machine(self, machine_id: int, name: str = None) -> None:
        """Add machine to job shop."""
        if name is None:
            name = f"Machine_{machine_id}"
        
        machine = Agent(id=machine_id, name=name)
        self.add_agent(machine)
        self.machines.append(machine_id)
    
    def encode_to_ising(self, **kwargs) -> IsingModel:
        """Encode job shop problem with precedence constraints."""
        # Add job precedence constraints
        kwargs["penalty_weights"] = kwargs.get("penalty_weights", {})
        kwargs["penalty_weights"]["job_precedence"] = 100.0
        
        # Call parent encoding
        model = super().encode_to_ising(**kwargs)
        
        # Add job-specific precedence constraints
        self._add_job_precedence_constraints(kwargs["penalty_weights"]["job_precedence"])
        
        return model
    
    def _add_job_precedence_constraints(self, penalty_weight: float) -> None:
        """Add precedence constraints within jobs."""
        for job_operations in self.jobs:
            # Find tasks belonging to this job
            job_tasks = []
            for task in self.tasks:
                if hasattr(task, 'metadata') and task.metadata.get("job_id") == len(job_tasks):
                    job_tasks.append(task.id)
            
            # Add precedence constraints between consecutive operations
            for i in range(len(job_tasks) - 1):
                task_i = job_tasks[i]
                task_j = job_tasks[i + 1]
                
                # Task i must complete before task j starts
                for agent_i in range(len(self.agents)):
                    for agent_j in range(len(self.agents)):
                        for time_i in range(self.time_discretization):
                            for time_j in range(time_i + 1):  # j must start after i
                                spin_i = self._get_spin_index(task_i, agent_i, time_i)
                                spin_j = self._get_spin_index(task_j, agent_j, time_j)
                                
                                # Penalize violations
                                current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                                self.ising_model.set_coupling(spin_i, spin_j, 
                                                            current_coupling + penalty_weight)