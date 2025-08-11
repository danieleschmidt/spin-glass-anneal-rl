"""Routing problem implementations (TSP, VRP, etc.)."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass

from spin_glass_rl.problems.base import ProblemTemplate, ProblemSolution
from spin_glass_rl.core.ising_model import IsingModel


@dataclass
class Location:
    """Location/node in routing problem."""
    id: int
    name: str
    x: float
    y: float
    demand: float = 0.0
    service_time: float = 0.0
    time_window: Optional[Tuple[float, float]] = None


@dataclass
class Vehicle:
    """Vehicle for routing problems."""
    id: int
    capacity: float
    max_distance: float = float('inf')
    max_time: float = float('inf')
    cost_per_distance: float = 1.0
    depot_id: int = 0


class RoutingProblem(ProblemTemplate):
    """
    Base class for routing problems.
    
    Provides common functionality for TSP, VRP, and variants.
    """
    
    def __init__(self, name: str = "Generic Routing Problem"):
        super().__init__(name)
        self.locations: List[Location] = []
        self.vehicles: List[Vehicle] = []
        self.distance_matrix: Optional[np.ndarray] = None
        self._variable_mapping = {}
    
    def add_location(self, location: Location) -> None:
        """Add location to routing problem."""
        self.locations.append(location)
        self._invalidate_distance_matrix()
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add vehicle to routing problem."""
        self.vehicles.append(vehicle)
    
    def set_distance_matrix(self, matrix: np.ndarray) -> None:
        """Set custom distance matrix."""
        if matrix.shape != (len(self.locations), len(self.locations)):
            raise ValueError("Distance matrix shape must match number of locations")
        self.distance_matrix = matrix
    
    def compute_distance_matrix(self, metric: str = "euclidean") -> np.ndarray:
        """Compute distance matrix between locations."""
        n_locations = len(self.locations)
        matrix = np.zeros((n_locations, n_locations))
        
        for i, loc_i in enumerate(self.locations):
            for j, loc_j in enumerate(self.locations):
                if i == j:
                    matrix[i, j] = 0.0
                elif metric == "euclidean":
                    dx = loc_i.x - loc_j.x
                    dy = loc_i.y - loc_j.y
                    matrix[i, j] = np.sqrt(dx**2 + dy**2)
                elif metric == "manhattan":
                    matrix[i, j] = abs(loc_i.x - loc_j.x) + abs(loc_i.y - loc_j.y)
                else:
                    raise ValueError(f"Unknown distance metric: {metric}")
        
        self.distance_matrix = matrix
        return matrix
    
    def _invalidate_distance_matrix(self) -> None:
        """Invalidate cached distance matrix."""
        self.distance_matrix = None
    
    def get_distance(self, i: int, j: int) -> float:
        """Get distance between locations i and j."""
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        return self.distance_matrix[i, j]
    
    def generate_random_instance(
        self,
        n_locations: int = 10,
        n_vehicles: int = 1,
        area_size: float = 100.0,
        **kwargs
    ) -> Dict:
        """Generate random routing instance."""
        # Clear existing data
        self.locations = []
        self.vehicles = []
        
        # Generate random locations
        for i in range(n_locations):
            location = Location(
                id=i,
                name=f"Loc_{i}",
                x=np.random.uniform(0, area_size),
                y=np.random.uniform(0, area_size),
                demand=np.random.uniform(1.0, 10.0) if i > 0 else 0.0  # Depot has no demand
            )
            self.add_location(location)
        
        # Generate vehicles
        for i in range(n_vehicles):
            vehicle = Vehicle(
                id=i,
                capacity=np.random.uniform(20.0, 50.0),
                depot_id=0  # All start from location 0
            )
            self.add_vehicle(vehicle)
        
        return {
            "n_locations": n_locations,
            "n_vehicles": n_vehicles,
            "area_size": area_size
        }
    
    def plot_solution(self, solution: ProblemSolution, save_path: Optional[str] = None, **plot_params) -> None:
        """Plot routing solution."""
        try:
            import matplotlib.pyplot as plt
            
            routes = solution.variables.get("routes", [])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot locations
            x_coords = [loc.x for loc in self.locations]
            y_coords = [loc.y for loc in self.locations]
            
            # Plot depot differently
            ax.scatter(x_coords[0], y_coords[0], c='red', s=200, marker='s', label='Depot')
            
            # Plot customer locations
            ax.scatter(x_coords[1:], y_coords[1:], c='blue', s=100, marker='o', label='Customers')
            
            # Add location labels
            for i, loc in enumerate(self.locations):
                ax.annotate(f'{i}', (loc.x, loc.y), xytext=(5, 5), textcoords='offset points')
            
            # Plot routes
            colors = plt.cm.Set1(np.linspace(0, 1, len(routes)))
            
            for route_idx, route in enumerate(routes):
                if len(route) > 1:
                    route_x = [self.locations[loc_id].x for loc_id in route]
                    route_y = [self.locations[loc_id].y for loc_id in route]
                    
                    ax.plot(route_x, route_y, 'o-', color=colors[route_idx], 
                           linewidth=2, markersize=6, label=f'Route {route_idx + 1}')
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'{self.name} Solution (Distance: {solution.objective_value:.2f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            super().plot_solution(solution, save_path, **plot_params)


class TSPProblem(RoutingProblem):
    """
    Traveling Salesman Problem (TSP).
    
    Find shortest tour visiting all locations exactly once.
    """
    
    def __init__(self):
        super().__init__("Traveling Salesman Problem")
    
    def encode_to_ising(
        self,
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """
        Encode TSP as Ising model using position-based encoding.
        
        Variables: x_{i,p} = 1 if city i is visited at position p in tour
        """
        # Enhanced validation
        if not self.locations:
            raise ValueError("TSP requires at least one location")
        if len(self.locations) < 2:
            raise ValueError("TSP requires at least 2 locations")
        if len(self.locations) > 1000:
            print("Warning: Large TSP instance may be slow. Consider using hierarchical methods.")
        
        # Validate locations have valid coordinates
        for i, loc in enumerate(self.locations):
            if not isinstance(loc.x, (int, float)) or not isinstance(loc.y, (int, float)):
                raise ValueError(f"Location {i} has invalid coordinates: ({loc.x}, {loc.y})")
            if not (-1e6 <= loc.x <= 1e6) or not (-1e6 <= loc.y <= 1e6):
                print(f"Warning: Location {i} has extreme coordinates: ({loc.x}, {loc.y})")
        
        if penalty_weights is None:
            penalty_weights = {
                "city_visit": 100.0,
                "position_fill": 100.0
            }
        
        # Validate penalty weights
        required_weights = {"city_visit", "position_fill"}
        missing_weights = required_weights - set(penalty_weights.keys())
        if missing_weights:
            raise ValueError(f"Missing required penalty weights: {missing_weights}")
        
        for key, weight in penalty_weights.items():
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(f"Penalty weight '{key}' must be a positive number, got: {weight}")
            if weight > 1e6:
                print(f"Warning: Very large penalty weight for '{key}': {weight}")
        
        # Auto-scale penalty weights for large problems
        n_cities = len(self.locations)
        if n_cities > 50:
            scale_factor = np.sqrt(n_cities / 50.0)
            penalty_weights = {k: v * scale_factor for k, v in penalty_weights.items()}
            print(f"Auto-scaled penalty weights by factor {scale_factor:.2f} for large problem")
        
        # Ensure distance matrix exists
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        
        n_cities = len(self.locations)
        
        # Binary variables: x_{i,p} for city i at position p
        n_spins = n_cities * n_cities
        
        self.ising_model = self.create_ising_model(n_spins)
        
        # Create variable mapping
        self._variable_mapping = {}
        spin_idx = 0
        for city in range(n_cities):
            for position in range(n_cities):
                self._variable_mapping[(city, position)] = spin_idx
                spin_idx += 1
        
        # Add objective function (minimize total distance)
        self._add_tsp_objective()
        
        # Add constraints
        self._add_city_visit_constraints(penalty_weights["city_visit"])
        self._add_position_fill_constraints(penalty_weights["position_fill"])
        
        return self.ising_model
    
    def _get_spin_index(self, city: int, position: int) -> int:
        """Get spin index for city at position."""
        return self._variable_mapping[(city, position)]
    
    def _add_tsp_objective(self) -> None:
        """Add TSP distance minimization objective."""
        n_cities = len(self.locations)
        
        for city_i in range(n_cities):
            for city_j in range(n_cities):
                if city_i != city_j:
                    distance = self.get_distance(city_i, city_j)
                    
                    # For each consecutive position pair
                    for pos in range(n_cities):
                        next_pos = (pos + 1) % n_cities
                        
                        spin_i = self._get_spin_index(city_i, pos)
                        spin_j = self._get_spin_index(city_j, next_pos)
                        
                        # Add coupling for consecutive cities
                        current_coupling = self.ising_model.couplings[spin_i, spin_j].item()
                        self.ising_model.set_coupling(spin_i, spin_j, 
                                                    current_coupling - distance)  # Negative for minimization
    
    def _add_city_visit_constraints(self, penalty_weight: float) -> None:
        """Each city visited exactly once."""
        n_cities = len(self.locations)
        
        for city in range(n_cities):
            city_spins = []
            for position in range(n_cities):
                spin_idx = self._get_spin_index(city, position)
                city_spins.append(spin_idx)
            
            self.constraint_encoder.add_cardinality_constraint(
                city_spins,
                k=1,
                penalty_weight=penalty_weight,
                description=f"City {city} visited once"
            )
    
    def _add_position_fill_constraints(self, penalty_weight: float) -> None:
        """Each position filled by exactly one city."""
        n_cities = len(self.locations)
        
        for position in range(n_cities):
            position_spins = []
            for city in range(n_cities):
                spin_idx = self._get_spin_index(city, position)
                position_spins.append(spin_idx)
            
            self.constraint_encoder.add_cardinality_constraint(
                position_spins,
                k=1,
                penalty_weight=penalty_weight,
                description=f"Position {position} filled once"
            )
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode TSP solution from spins."""
        binary_spins = (spins + 1) // 2
        n_cities = len(self.locations)
        
        # Extract tour
        tour = [-1] * n_cities
        assignment_matrix = np.zeros((n_cities, n_cities))
        
        for city in range(n_cities):
            for position in range(n_cities):
                spin_idx = self._get_spin_index(city, position)
                if binary_spins[spin_idx].item() == 1:
                    tour[position] = city
                    assignment_matrix[city, position] = 1
        
        # Calculate total distance
        total_distance = 0.0
        valid_tour = True
        
        for i in range(n_cities):
            if tour[i] == -1:
                valid_tour = False
                break
            
            current_city = tour[i]
            next_city = tour[(i + 1) % n_cities]
            total_distance += self.get_distance(current_city, next_city)
        
        # Check constraints
        constraint_violations = {
            "city_assignment": 0.0,
            "position_assignment": 0.0,
            "invalid_tour": 0.0 if valid_tour else 1.0
        }
        
        # Check city constraints
        for city in range(n_cities):
            city_assignments = assignment_matrix[city, :].sum()
            if city_assignments != 1:
                constraint_violations["city_assignment"] += abs(city_assignments - 1)
        
        # Check position constraints
        for position in range(n_cities):
            position_assignments = assignment_matrix[:, position].sum()
            if position_assignments != 1:
                constraint_violations["position_assignment"] += abs(position_assignments - 1)
        
        is_feasible = all(v == 0 for v in constraint_violations.values())
        
        return ProblemSolution(
            variables={"tour": tour, "routes": [tour] if valid_tour else []},
            objective_value=total_distance,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "tour_length": len([c for c in tour if c != -1]),
                "total_distance": total_distance
            }
        )
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate TSP solution."""
        tour = solution.variables["tour"]
        
        # Check if all cities are visited exactly once
        if len(set(tour)) != len(self.locations):
            return False
        
        # Check if tour is complete
        if -1 in tour:
            return False
        
        return True


class VRPProblem(RoutingProblem):
    """
    Vehicle Routing Problem (VRP).
    
    Find optimal routes for multiple vehicles to serve all customers.
    """
    
    def __init__(self):
        super().__init__("Vehicle Routing Problem")
    
    def encode_to_ising(
        self,
        penalty_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> IsingModel:
        """
        Encode VRP as Ising model.
        
        Variables: x_{i,j,k} = 1 if vehicle k travels from location i to j
        """
        if len(self.locations) < 2:
            raise ValueError("VRP requires at least 2 locations")
        if len(self.vehicles) < 1:
            raise ValueError("VRP requires at least 1 vehicle")
        
        if penalty_weights is None:
            penalty_weights = {
                "customer_service": 100.0,
                "vehicle_flow": 100.0,
                "capacity": 50.0,
                "depot_return": 75.0
            }
        
        # Ensure distance matrix exists
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        # Binary variables: x_{i,j,k} for arc (i,j) used by vehicle k
        n_spins = n_locations * n_locations * n_vehicles
        
        self.ising_model = self.create_ising_model(n_spins)
        
        # Create variable mapping
        self._variable_mapping = {}
        spin_idx = 0
        for i in range(n_locations):
            for j in range(n_locations):
                for k in range(n_vehicles):
                    self._variable_mapping[(i, j, k)] = spin_idx
                    spin_idx += 1
        
        # Add objective function
        self._add_vrp_objective()
        
        # Add constraints
        self._add_customer_service_constraints(penalty_weights["customer_service"])
        self._add_vehicle_flow_constraints(penalty_weights["vehicle_flow"])
        self._add_capacity_constraints(penalty_weights["capacity"])
        self._add_depot_constraints(penalty_weights["depot_return"])
        
        return self.ising_model
    
    def _get_spin_index(self, i: int, j: int, k: int) -> int:
        """Get spin index for arc (i,j) and vehicle k."""
        return self._variable_mapping[(i, j, k)]
    
    def _add_vrp_objective(self) -> None:
        """Add VRP distance minimization objective."""
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance = self.get_distance(i, j)
                    
                    for k in range(n_vehicles):
                        spin_idx = self._get_spin_index(i, j, k)
                        
                        # Add distance cost to external field
                        current_field = self.ising_model.external_fields[spin_idx].item()
                        self.ising_model.set_external_field(spin_idx, current_field + distance)
    
    def _add_customer_service_constraints(self, penalty_weight: float) -> None:
        """Each customer served exactly once."""
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        for customer in range(1, n_locations):  # Skip depot (location 0)
            # Incoming arcs to customer
            customer_spins = []
            for i in range(n_locations):
                if i != customer:
                    for k in range(n_vehicles):
                        spin_idx = self._get_spin_index(i, customer, k)
                        customer_spins.append(spin_idx)
            
            self.constraint_encoder.add_cardinality_constraint(
                customer_spins,
                k=1,
                penalty_weight=penalty_weight,
                description=f"Customer {customer} served once"
            )
    
    def _add_vehicle_flow_constraints(self, penalty_weight: float) -> None:
        """Flow conservation for each vehicle at each location."""
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        for k in range(n_vehicles):
            for location in range(n_locations):
                # Incoming and outgoing arcs must balance
                incoming_spins = []
                outgoing_spins = []
                
                for i in range(n_locations):
                    if i != location:
                        # Incoming arc
                        spin_in = self._get_spin_index(i, location, k)
                        incoming_spins.append(spin_in)
                        
                        # Outgoing arc
                        spin_out = self._get_spin_index(location, i, k)
                        outgoing_spins.append(spin_out)
                
                # Flow conservation: incoming = outgoing
                all_spins = incoming_spins + outgoing_spins
                coefficients = [1.0] * len(incoming_spins) + [-1.0] * len(outgoing_spins)
                
                self.constraint_encoder.add_equality_constraint(
                    all_spins,
                    coefficients,
                    target=0.0,
                    penalty_weight=penalty_weight,
                    description=f"Flow conservation vehicle {k} location {location}"
                )
    
    def _add_capacity_constraints(self, penalty_weight: float) -> None:
        """Vehicle capacity constraints."""
        # Simplified capacity constraint implementation
        # Full implementation would track cumulative demand along routes
        
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        for k, vehicle in enumerate(self.vehicles):
            # For each customer, penalize if total demand exceeds capacity
            for customer in range(1, n_locations):
                customer_demand = self.locations[customer].demand
                
                if customer_demand > vehicle.capacity:
                    # This customer cannot be served by this vehicle
                    for i in range(n_locations):
                        if i != customer:
                            spin_idx = self._get_spin_index(i, customer, k)
                            
                            current_field = self.ising_model.external_fields[spin_idx].item()
                            self.ising_model.set_external_field(spin_idx, 
                                                              current_field + penalty_weight)
    
    def _add_depot_constraints(self, penalty_weight: float) -> None:
        """Each vehicle starts and ends at depot."""
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        depot = 0
        
        for k in range(n_vehicles):
            # Vehicle can use at most one outgoing arc from depot
            depot_outgoing = []
            for j in range(1, n_locations):
                spin_idx = self._get_spin_index(depot, j, k)
                depot_outgoing.append(spin_idx)
            
            if depot_outgoing:
                self.constraint_encoder.add_cardinality_constraint(
                    depot_outgoing,
                    k=1,  # At most one outgoing arc
                    penalty_weight=penalty_weight,
                    description=f"Vehicle {k} depot departure"
                )
    
    def decode_solution(self, spins: torch.Tensor) -> ProblemSolution:
        """Decode VRP solution from spins."""
        binary_spins = (spins + 1) // 2
        
        n_locations = len(self.locations)
        n_vehicles = len(self.vehicles)
        
        # Extract routes for each vehicle
        routes = []
        total_distance = 0.0
        
        for k in range(n_vehicles):
            route = []
            current_location = 0  # Start at depot
            visited = set([0])
            
            while True:
                next_location = None
                
                # Find next location in route
                for j in range(n_locations):
                    if j != current_location:
                        spin_idx = self._get_spin_index(current_location, j, k)
                        if binary_spins[spin_idx].item() == 1:
                            next_location = j
                            break
                
                if next_location is None or next_location == 0:
                    # Return to depot or no next location
                    if next_location == 0 and current_location != 0:
                        route.append(0)
                        total_distance += self.get_distance(current_location, 0)
                    break
                
                if next_location in visited and next_location != 0:
                    # Avoid cycles (except returning to depot)
                    break
                
                route.append(next_location)
                total_distance += self.get_distance(current_location, next_location)
                visited.add(next_location)
                current_location = next_location
            
            if len(route) > 1:  # Only include non-empty routes
                routes.append([0] + route)  # Add depot start
        
        # Check constraints
        served_customers = set()
        for route in routes:
            for location in route:
                if location > 0:  # Not depot
                    served_customers.add(location)
        
        unserved_customers = set(range(1, n_locations)) - served_customers
        
        constraint_violations = {
            "unserved_customers": len(unserved_customers),
            "capacity_violations": 0.0,  # Simplified
            "flow_violations": 0.0  # Simplified
        }
        
        is_feasible = len(unserved_customers) == 0
        
        return ProblemSolution(
            variables={"routes": routes},
            objective_value=total_distance,
            is_feasible=is_feasible,
            constraint_violations=constraint_violations,
            metadata={
                "n_routes": len(routes),
                "total_distance": total_distance,
                "served_customers": len(served_customers),
                "unserved_customers": len(unserved_customers)
            }
        )
    
    def validate_solution(self, solution: ProblemSolution) -> bool:
        """Validate VRP solution."""
        routes = solution.variables["routes"]
        
        # Check if all customers are served
        served_customers = set()
        for route in routes:
            for location in route:
                if location > 0:  # Not depot
                    served_customers.add(location)
        
        all_customers = set(range(1, len(self.locations)))
        return served_customers == all_customers