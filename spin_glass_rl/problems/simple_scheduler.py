"""Simple scheduling problem implementation for immediate functionality."""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from dataclasses import dataclass

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.core.constraints import ConstraintEncoder
from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution


@dataclass
class SimpleTask:
    """Simple task representation."""
    id: int
    duration: float
    due_date: Optional[float] = None


@dataclass
class SimpleAgent:
    """Simple agent representation."""
    id: int
    cost_rate: float = 1.0


class SimpleScheduler(ProblemTemplate):
    """Simple multi-agent scheduling problem for immediate functionality."""
    
    def __init__(self):
        super().__init__("Simple Multi-Agent Scheduler")
        self.tasks: List[SimpleTask] = []
        self.agents: List[SimpleAgent] = []
        self.time_horizon = 100.0
        
    def generate_random_instance(
        self, 
        n_tasks: int = 6, 
        n_agents: int = 3,
        max_duration: float = 20.0,
        time_horizon: float = 100.0
    ) -> Dict[str, Any]:
        """Generate random scheduling instance."""
        self.time_horizon = time_horizon
        self.tasks = []
        self.agents = []
        
        # Generate tasks
        for i in range(n_tasks):
            duration = np.random.uniform(5.0, max_duration)
            due_date = np.random.uniform(duration, time_horizon * 0.8)
            self.tasks.append(SimpleTask(i, duration, due_date))
        
        # Generate agents
        for i in range(n_agents):
            cost_rate = np.random.uniform(0.5, 2.0)
            self.agents.append(SimpleAgent(i, cost_rate))
        
        return {
            "n_tasks": n_tasks,
            "n_agents": n_agents,
            "time_horizon": time_horizon,
            "total_task_duration": sum(t.duration for t in self.tasks)
        }
    
    def encode_to_ising(self, penalty_weights: Optional[Dict] = None) -> IsingModel:
        """Encode scheduling problem as Ising model."""
        if not self.tasks or not self.agents:
            raise ValueError("No tasks or agents defined. Call generate_random_instance() first.")
        
        if penalty_weights is None:
            penalty_weights = {
                "assignment": 100.0,
                "capacity": 50.0
            }
        
        n_tasks = len(self.tasks)
        n_agents = len(self.agents)
        
        # Binary variables: x[i,j] = 1 if task i assigned to agent j
        n_spins = n_tasks * n_agents
        
        # Create Ising model
        config = IsingModelConfig(n_spins=n_spins, use_sparse=True, device="cpu")
        model = IsingModel(config)
        self.ising_model = model
        
        # Create constraint encoder
        encoder = ConstraintEncoder(model)
        
        # Variable mapping: spin index = task_id * n_agents + agent_id
        self._variable_mapping = {}
        for task_id in range(n_tasks):
            for agent_id in range(n_agents):
                spin_id = task_id * n_agents + agent_id
                self._variable_mapping[(task_id, agent_id)] = spin_id
        
        # Constraint 1: Each task assigned to exactly one agent
        for task_id in range(n_tasks):
            task_spins = [self._variable_mapping[(task_id, agent_id)] for agent_id in range(n_agents)]
            encoder.add_cardinality_constraint(
                task_spins, 
                k=1, 
                penalty_weight=penalty_weights["assignment"],
                description=f"Task {task_id} assignment"
            )
        
        # Objective: Minimize total completion time (makespan approximation)
        # Add linear terms favoring earlier assignments and cheaper agents
        for task_id, task in enumerate(self.tasks):
            for agent_id, agent in enumerate(self.agents):
                spin_idx = self._variable_mapping[(task_id, agent_id)]
                
                # Cost component: task duration * agent cost rate
                cost = task.duration * agent.cost_rate
                
                # Due date penalty: penalize if assignment might cause lateness
                due_penalty = 0.0
                if task.due_date:
                    due_penalty = max(0, task.duration - task.due_date) * 0.1
                
                # Set external field to encourage good assignments
                total_field = cost + due_penalty
                model.set_external_field(spin_idx, total_field)
        
        return model
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode spin configuration to scheduling solution."""
        if not hasattr(self, '_variable_mapping'):
            raise ValueError("Problem not encoded yet")
        
        n_tasks = len(self.tasks)
        n_agents = len(self.agents)
        
        # Extract assignments
        assignments = {}  # agent_id -> list of task_ids
        task_assignments = {}  # task_id -> agent_id
        
        for agent_id in range(n_agents):
            assignments[agent_id] = []
        
        # Decode spin configuration
        for task_id in range(n_tasks):
            assigned_agent = None
            for agent_id in range(n_agents):
                spin_idx = self._variable_mapping[(task_id, agent_id)]
                if spins[spin_idx].item() > 0:  # Spin is +1 (selected)
                    if assigned_agent is None:
                        assigned_agent = agent_id
                        assignments[agent_id].append(task_id)
                        task_assignments[task_id] = agent_id
                    else:
                        # Multiple assignments - constraint violation
                        pass
        
        # Compute objective and feasibility
        total_cost = 0.0
        makespan = 0.0
        unassigned_tasks = []
        
        for task_id, task in enumerate(self.tasks):
            if task_id in task_assignments:
                agent_id = task_assignments[task_id]
                agent = self.agents[agent_id]
                total_cost += task.duration * agent.cost_rate
            else:
                unassigned_tasks.append(task_id)
        
        # Compute makespan (simplified)
        agent_loads = {agent_id: sum(self.tasks[tid].duration for tid in task_list) 
                      for agent_id, task_list in assignments.items()}
        makespan = max(agent_loads.values()) if agent_loads else 0.0
        
        # Check feasibility
        is_feasible = len(unassigned_tasks) == 0
        
        # Constraint violations
        violations = {}
        if unassigned_tasks:
            violations["unassigned_tasks"] = len(unassigned_tasks)
        
        # Check capacity constraints (simple version)
        for agent_id, load in agent_loads.items():
            if load > self.time_horizon:
                violations[f"agent_{agent_id}_overload"] = load - self.time_horizon
        
        return ProblemSolution(
            variables={
                "assignments": assignments,
                "task_assignments": task_assignments,
                "schedule": self._create_schedule(assignments)
            },
            objective_value=makespan,  # Use makespan as primary objective
            is_feasible=is_feasible,
            constraint_violations=violations,
            metadata={
                "total_cost": total_cost,
                "makespan": makespan,
                "n_assigned_tasks": len(task_assignments),
                "agent_loads": agent_loads
            }
        )
    
    def _create_schedule(self, assignments: Dict[int, List[int]]) -> Dict:
        """Create detailed schedule from assignments."""
        schedule = {}
        
        for agent_id, task_ids in assignments.items():
            agent_schedule = []
            current_time = 0.0
            
            # Sort tasks by duration (simple heuristic)
            sorted_tasks = sorted(task_ids, key=lambda tid: self.tasks[tid].duration)
            
            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                start_time = current_time
                end_time = current_time + task.duration
                
                agent_schedule.append({
                    "task_id": task_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": task.duration
                })
                
                current_time = end_time
            
            schedule[agent_id] = agent_schedule
        
        return schedule
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate solution feasibility."""
        return solution.is_feasible and len(solution.constraint_violations) == 0
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None) -> None:
        """Plot Gantt chart of solution."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if not solution.is_feasible:
                print("Solution is not feasible - plotting may be incomplete")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
            agent_y_positions = {}
            
            schedule = solution.variables["schedule"]
            
            # Plot each agent's schedule
            for agent_id, agent_schedule in schedule.items():
                y_pos = agent_id
                agent_y_positions[agent_id] = y_pos
                
                for task_info in agent_schedule:
                    start_time = task_info["start_time"]
                    duration = task_info["duration"]
                    task_id = task_info["task_id"]
                    
                    # Create rectangle for task
                    rect = patches.Rectangle(
                        (start_time, y_pos - 0.4),
                        duration,
                        0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[task_id % len(colors)],
                        alpha=0.7
                    )
                    ax.add_patch(rect)
                    
                    # Add task label
                    ax.text(
                        start_time + duration/2,
                        y_pos,
                        f"T{task_id}",
                        ha='center',
                        va='center',
                        fontsize=8,
                        weight='bold'
                    )
            
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Agent')
            ax.set_title(f'Schedule (Makespan: {solution.metadata["makespan"]:.1f})')
            
            # Set y-axis ticks
            ax.set_yticks(list(range(len(self.agents))))
            ax.set_yticklabels([f'Agent {i}' for i in range(len(self.agents))])
            
            # Set x-axis limits
            max_time = max(solution.metadata["agent_loads"].values()) if solution.metadata["agent_loads"] else 100
            ax.set_xlim(0, max_time * 1.1)
            ax.set_ylim(-0.5, len(self.agents) - 0.5)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Schedule saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available - cannot plot schedule")
            print(f"Schedule summary:")
            print(f"  Makespan: {solution.metadata['makespan']:.1f}")
            print(f"  Total cost: {solution.metadata['total_cost']:.1f}")
            print(f"  Assigned tasks: {solution.metadata['n_assigned_tasks']}/{len(self.tasks)}")


# Quick test function for immediate validation
def quick_test():
    """Quick test of the simple scheduler."""
    print("🧪 Testing Simple Scheduler...")
    
    # Create scheduler
    scheduler = SimpleScheduler()
    
    # Generate small instance
    instance = scheduler.generate_random_instance(n_tasks=4, n_agents=2)
    print(f"Generated instance: {instance}")
    
    # Encode as Ising model
    ising_model = scheduler.encode_to_ising()
    print(f"Encoded as Ising model with {ising_model.n_spins} spins")
    
    # Create a manual solution for testing
    test_spins = torch.zeros(ising_model.n_spins)
    # Assign task 0 to agent 0, task 1 to agent 1, etc.
    test_spins[0] = 1  # Task 0 -> Agent 0
    test_spins[3] = 1  # Task 1 -> Agent 1  
    test_spins[4] = 1  # Task 2 -> Agent 0
    test_spins[7] = 1  # Task 3 -> Agent 1
    
    # Decode solution
    solution = scheduler.decode_solution(test_spins)
    print(f"Solution feasible: {solution.is_feasible}")
    print(f"Makespan: {solution.metadata['makespan']:.1f}")
    print(f"Assignments: {solution.variables['task_assignments']}")
    
    print("✅ Simple Scheduler test completed!")
    return scheduler, solution


if __name__ == "__main__":
    quick_test()