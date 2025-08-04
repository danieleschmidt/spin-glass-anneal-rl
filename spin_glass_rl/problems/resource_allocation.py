"""Resource allocation problem implementations."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass

from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution
from spin_glass_rl.core.ising_model import IsingModel


@dataclass
class Resource:
    """Resource definition."""
    id: int
    name: str
    capacity: float
    cost_per_unit: float = 1.0
    location: Optional[Tuple[float, float]] = None
    availability: float = 1.0  # Fraction of time available


@dataclass
class Demand:
    """Demand/request definition."""
    id: int
    name: str
    required_amount: float
    priority: float = 1.0
    location: Optional[Tuple[float, float]] = None
    deadline: Optional[float] = None
    penalty_late: float = 0.0


class ResourceAllocationProblem(ProblemTemplate):
    """
    Generic resource allocation problem.
    
    Assigns limited resources to demands while optimizing
    utilization, cost, or service levels.
    """
    
    def __init__(self, name: str = "Resource Allocation Problem"):
        super().__init__(name)
        self.resources: List[Resource] = []
        self.demands: List[Demand] = []
        self.allocation_costs: Optional[np.ndarray] = None  # Cost matrix [demand, resource]
        self._variable_mapping = {}
    
    def add_resource(self, resource: Resource) -> None:
        """Add resource to problem."""
        self.resources.append(resource)
    
    def add_demand(self, demand: Demand) -> None:
        """Add demand to problem."""
        self.demands.append(demand)
    
    def set_allocation_costs(self, cost_matrix: np.ndarray) -> None:
        """Set custom allocation cost matrix."""
        expected_shape = (len(self.demands), len(self.resources))
        if cost_matrix.shape != expected_shape:
            raise ValueError(f"Cost matrix shape {cost_matrix.shape} != expected {expected_shape}")
        self.allocation_costs = cost_matrix
    
    def compute_allocation_costs(self, cost_type: str = "distance") -> np.ndarray:
        """Compute allocation costs between demands and resources."""
        n_demands = len(self.demands)
        n_resources = len(self.resources)
        costs = np.ones((n_demands, n_resources))
        
        for i, demand in enumerate(self.demands):
            for j, resource in enumerate(self.resources):
                if cost_type == "distance":
                    if demand.location and resource.location:
                        dx = demand.location[0] - resource.location[0]
                        dy = demand.location[1] - resource.location[1]
                        costs[i, j] = np.sqrt(dx**2 + dy**2)
                    else:
                        costs[i, j] = 1.0  # Default unit cost
                
                elif cost_type == "capacity_ratio":
                    # Cost based on how much of resource capacity is used
                    if resource.capacity > 0:
                        costs[i, j] = demand.required_amount / resource.capacity
                    else:
                        costs[i, j] = float('inf')
                
                elif cost_type == "priority_weighted":
                    # Higher priority demands have lower cost
                    costs[i, j] = resource.cost_per_unit / demand.priority
                
                else:
                    costs[i, j] = resource.cost_per_unit
        
        self.allocation_costs = costs
        return costs
    
    def encode_to_ising(
        self,
        objective: str = "minimize_cost",
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """
        Encode resource allocation as Ising model.
        
        Args:
            objective: Optimization objective
            penalty_weights: Constraint penalty weights
        """
        if not self.demands or not self.resources:
            raise ValueError("Must add demands and resources before encoding")
        
        if penalty_weights is None:
            penalty_weights = {
                "demand_satisfaction": 100.0,
                "capacity": 50.0,
                "assignment": 75.0
            }
        
        # Ensure cost matrix exists
        if self.allocation_costs is None:
            self.compute_allocation_costs()
        
        n_demands = len(self.demands)
        n_resources = len(self.resources)
        
        # Binary variables: x_{i,j} = 1 if demand i assigned to resource j
        n_spins = n_demands * n_resources
        
        self.ising_model = self.create_ising_model(n_spins)
        
        # Create variable mapping
        self._variable_mapping = {}
        spin_idx = 0
        for demand_id in range(n_demands):
            for resource_id in range(n_resources):
                self._variable_mapping[(demand_id, resource_id)] = spin_idx
                spin_idx += 1
        
        # Add objective function
        self._add_objective_to_ising(objective)
        
        # Add constraints
        self._add_demand_satisfaction_constraints(penalty_weights["demand_satisfaction"])
        self._add_capacity_constraints(penalty_weights["capacity"])
        
        return self.ising_model
    
    def _get_spin_index(self, demand_id: int, resource_id: int) -> int:
        """Get spin index for demand-resource assignment."""
        return self._variable_mapping[(demand_id, resource_id)]
    
    def _add_objective_to_ising(self, objective: str) -> None:
        """Add objective function to Ising model."""
        if objective == "minimize_cost":
            self._add_cost_minimization_objective()
        elif objective == "maximize_satisfaction":
            self._add_satisfaction_maximization_objective()
        elif objective == "balance_load":
            self._add_load_balancing_objective()
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _add_cost_minimization_objective(self) -> None:
        """Minimize total allocation cost."""
        for demand_id in range(len(self.demands)):
            for resource_id in range(len(self.resources)):
                spin_idx = self._get_spin_index(demand_id, resource_id)
                cost = self.allocation_costs[demand_id, resource_id]
                
                # Add cost to external field
                current_field = self.ising_model.external_fields[spin_idx].item()
                self.ising_model.set_external_field(spin_idx, current_field + cost)
    
    def _add_satisfaction_maximization_objective(self) -> None:
        """Maximize demand satisfaction (minimize negative satisfaction)."""
        for demand_id, demand in enumerate(self.demands):
            for resource_id, resource in enumerate(self.resources):
                spin_idx = self._get_spin_index(demand_id, resource_id)
                
                # Satisfaction based on priority and resource quality
                satisfaction = demand.priority * resource.availability
                
                # Negative for maximization
                current_field = self.ising_model.external_fields[spin_idx].item()
                self.ising_model.set_external_field(spin_idx, current_field - satisfaction)
    
    def _add_load_balancing_objective(self) -> None:
        """Balance load across resources."""
        # Add quadratic penalty for uneven resource utilization
        for resource_i in range(len(self.resources)):
            for resource_j in range(resource_i + 1, len(self.resources)):
                # Encourage different resources to have similar loads
                for demand_i in range(len(self.demands)):
                    for demand_j in range(len(self.demands)):
                        if demand_i != demand_j:
                            spin_i = self._get_spin_index(demand_i, resource_i)
                            spin_j = self._get_spin_index(demand_j, resource_j)
                            
                            # Coupling to balance loads
                            balancing_strength = 0.1
                            current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                            self.ising_model.set_coupling(spin_i, spin_j, 
                                                        current_coupling - balancing_strength)
    
    def _add_demand_satisfaction_constraints(self, penalty_weight: float) -> None:
        """Each demand should be satisfied (at least partially)."""
        for demand_id in range(len(self.demands)):
            demand_spins = []
            for resource_id in range(len(self.resources)):
                spin_idx = self._get_spin_index(demand_id, resource_id)
                demand_spins.append(spin_idx)
            
            # At least one resource assigned to each demand
            # Using soft constraint instead of hard constraint for flexibility
            for spin_idx in demand_spins:
                current_field = self.ising_model.external_fields[spin_idx].item()
                self.ising_model.set_external_field(spin_idx, 
                                                  current_field - penalty_weight * 0.1)
    
    def _add_capacity_constraints(self, penalty_weight: float) -> None:
        """Resource capacity constraints."""
        for resource_id, resource in enumerate(self.resources):
            # Collect all demands that could use this resource
            resource_demands = []
            demand_amounts = []
            
            for demand_id, demand in enumerate(self.demands):
                if demand.required_amount <= resource.capacity:
                    spin_idx = self._get_spin_index(demand_id, resource_id)
                    resource_demands.append(spin_idx)
                    demand_amounts.append(demand.required_amount)
            
            # Simple capacity constraint: weighted sum <= capacity
            if resource_demands:
                # For now, use simplified linear constraint
                # Full implementation would need integer programming formulation
                total_demand = sum(demand_amounts)
                if total_demand > resource.capacity:
                    # Add penalty for over-capacity allocation
                    excess_factor = total_demand / resource.capacity
                    for spin_idx in resource_demands:
                        current_field = self.ising_model.external_fields[spin_idx].item()
                        self.ising_model.set_external_field(spin_idx, 
                                                          current_field + penalty_weight * excess_factor)
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode resource allocation solution."""
        binary_spins = (spins + 1) // 2
        
        # Extract assignments
        assignments = {}
        resource_loads = {r.id: 0.0 for r in self.resources}
        total_cost = 0.0
        
        for demand_id in range(len(self.demands)):
            assignments[demand_id] = []
            
            for resource_id in range(len(self.resources)):
                spin_idx = self._get_spin_index(demand_id, resource_id)
                
                if binary_spins[spin_idx].item() == 1:
                    assignments[demand_id].append(resource_id)
                    
                    # Update resource load
                    demand = self.demands[demand_id]
                    resource_loads[resource_id] += demand.required_amount
                    
                    # Update total cost
                    total_cost += self.allocation_costs[demand_id, resource_id]
        
        # Check constraints
        constraint_violations = self._check_allocation_constraints(assignments, resource_loads)
        is_feasible = all(v == 0 for v in constraint_violations.values())
        
        # Calculate satisfaction metrics
        satisfied_demands = sum(1 for assigns in assignments.values() if len(assigns) > 0)
        satisfaction_rate = satisfied_demands / len(self.demands)
        
        return ProblemSolution(
            variables={
                "assignments": assignments,
                "resource_loads": resource_loads
            },
            objective_value=total_cost,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "satisfaction_rate": satisfaction_rate,
                "satisfied_demands": satisfied_demands,
                "total_cost": total_cost,
                "avg_resource_utilization": np.mean(list(resource_loads.values()))
            }
        )
    
    def _check_allocation_constraints(self, assignments: Dict, resource_loads: Dict) -> Dict[str, float]:
        """Check constraint violations."""
        violations = {
            "unassigned_demands": 0.0,
            "capacity_violations": 0.0,
            "multiple_assignments": 0.0
        }
        
        # Check unassigned demands
        for demand_id, demand_assignments in assignments.items():
            if len(demand_assignments) == 0:
                violations["unassigned_demands"] += 1.0
            elif len(demand_assignments) > 1:
                violations["multiple_assignments"] += len(demand_assignments) - 1
        
        # Check capacity violations
        for resource_id, load in resource_loads.items():
            resource = self.resources[resource_id]
            if load > resource.capacity:
                violations["capacity_violations"] += load - resource.capacity
        
        return violations
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate resource allocation solution."""
        return solution.is_feasible
    
    def generate_random_instance(
        self,
        n_demands: int = 10,
        n_resources: int = 5,
        capacity_range: Tuple[float, float] = (10.0, 50.0),
        demand_range: Tuple[float, float] = (1.0, 15.0),
        **kwargs
    ) -> Dict:
        """Generate random resource allocation instance."""
        # Clear existing data
        self.demands = []
        self.resources = []
        
        # Generate random resources
        for i in range(n_resources):
            resource = Resource(
                id=i,
                name=f"Resource_{i}",
                capacity=np.random.uniform(*capacity_range),
                cost_per_unit=np.random.uniform(1.0, 10.0),
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                availability=np.random.uniform(0.7, 1.0)
            )
            self.add_resource(resource)
        
        # Generate random demands
        for i in range(n_demands):
            demand = Demand(
                id=i,
                name=f"Demand_{i}",
                required_amount=np.random.uniform(*demand_range),
                priority=np.random.uniform(0.5, 2.0),
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                deadline=np.random.uniform(50, 200) if np.random.rand() > 0.3 else None
            )
            self.add_demand(demand)
        
        return {
            "n_demands": n_demands,
            "n_resources": n_resources,
            "capacity_range": capacity_range,
            "demand_range": demand_range
        }
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None, **plot_params) -> None:
        """Plot resource allocation solution."""
        try:
            import matplotlib.pyplot as plt
            
            assignments = solution.variables["assignments"]
            resource_loads = solution.variables["resource_loads"]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Assignment matrix heatmap
            assignment_matrix = np.zeros((len(self.demands), len(self.resources)))
            for demand_id, resource_list in assignments.items():
                for resource_id in resource_list:
                    assignment_matrix[demand_id, resource_id] = 1
            
            im1 = ax1.imshow(assignment_matrix, cmap='Blues', aspect='auto')
            ax1.set_xlabel('Resources')
            ax1.set_ylabel('Demands')
            ax1.set_title('Assignment Matrix')
            plt.colorbar(im1, ax=ax1)
            
            # 2. Resource utilization
            resource_ids = list(resource_loads.keys())
            loads = list(resource_loads.values())
            capacities = [self.resources[i].capacity for i in resource_ids]
            
            x_pos = np.arange(len(resource_ids))
            ax2.bar(x_pos, loads, alpha=0.7, label='Current Load')
            ax2.bar(x_pos, capacities, alpha=0.3, label='Capacity', color='red')
            ax2.set_xlabel('Resource ID')
            ax2.set_ylabel('Load / Capacity')
            ax2.set_title('Resource Utilization')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(resource_ids)
            ax2.legend()
            
            # 3. Demand satisfaction
            satisfied = []
            unsatisfied = []
            for demand_id, resource_list in assignments.items():
                if len(resource_list) > 0:
                    satisfied.append(demand_id)
                else:
                    unsatisfied.append(demand_id)
            
            labels = ['Satisfied', 'Unsatisfied']
            sizes = [len(satisfied), len(unsatisfied)]
            colors = ['lightgreen', 'lightcoral']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Demand Satisfaction')
            
            # 4. Cost distribution
            if self.allocation_costs is not None:
                costs = []
                for demand_id, resource_list in assignments.items():
                    for resource_id in resource_list:
                        costs.append(self.allocation_costs[demand_id, resource_id])
                
                if costs:
                    ax4.hist(costs, bins=10, alpha=0.7, edgecolor='black')
                    ax4.set_xlabel('Allocation Cost')
                    ax4.set_ylabel('Frequency')
                    ax4.set_title('Cost Distribution')
                    ax4.axvline(np.mean(costs), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(costs):.2f}')
                    ax4.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            super().plot_solution(solution, save_path, **plot_params)
    
    def get_utilization_stats(self, solution: ProblemSolution) -> Dict:
        """Get detailed resource utilization statistics."""
        resource_loads = solution.variables["resource_loads"]
        
        utilizations = []
        for resource_id, load in resource_loads.items():
            capacity = self.resources[resource_id].capacity
            utilization = load / capacity if capacity > 0 else 0
            utilizations.append(utilization)
        
        return {
            "mean_utilization": np.mean(utilizations),
            "std_utilization": np.std(utilizations),
            "max_utilization": np.max(utilizations),
            "min_utilization": np.min(utilizations),
            "overutilized_resources": sum(1 for u in utilizations if u > 1.0),
            "underutilized_resources": sum(1 for u in utilizations if u < 0.5),
            "utilization_distribution": utilizations
        }