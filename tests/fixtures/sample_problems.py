"""Sample problem instances for testing."""

import numpy as np
from typing import Dict, Any, List


def create_job_shop_scheduling_problem(
    n_jobs: int = 5,
    n_machines: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample job shop scheduling problem.
    
    Args:
        n_jobs: Number of jobs to schedule
        n_machines: Number of machines available  
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Each job has a sequence of operations on different machines
    job_sequences = []
    processing_times = []
    
    for job in range(n_jobs):
        # Random machine sequence (each machine appears once)
        machines = np.random.permutation(n_machines)
        # Random processing times
        times = np.random.randint(1, 10, size=n_machines)
        
        job_sequences.append(machines.tolist())
        processing_times.append(times.tolist())
    
    return {
        "problem_type": "job_shop_scheduling",
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "job_sequences": job_sequences,
        "processing_times": processing_times,
        "metadata": {
            "seed": seed,
            "difficulty": "easy" if n_jobs <= 5 else "medium" if n_jobs <= 10 else "hard"
        }
    }


def create_vehicle_routing_problem(
    n_customers: int = 10,
    n_vehicles: int = 3,
    depot_location: tuple = (0, 0),
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample vehicle routing problem.
    
    Args:
        n_customers: Number of customers to visit
        n_vehicles: Number of vehicles available
        depot_location: Coordinates of the depot
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Random customer locations
    customer_locations = np.random.uniform(-10, 10, size=(n_customers, 2))
    
    # Random demands
    demands = np.random.randint(1, 10, size=n_customers)
    
    # Vehicle capacities
    vehicle_capacities = np.random.randint(15, 25, size=n_vehicles)
    
    # Distance matrix (Euclidean distances)
    all_locations = np.vstack([[depot_location], customer_locations])
    n_total = n_customers + 1
    
    distance_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(
                    all_locations[i] - all_locations[j]
                )
    
    return {
        "problem_type": "vehicle_routing",
        "n_customers": n_customers,
        "n_vehicles": n_vehicles,
        "depot_location": depot_location,
        "customer_locations": customer_locations.tolist(),
        "demands": demands.tolist(),
        "vehicle_capacities": vehicle_capacities.tolist(),
        "distance_matrix": distance_matrix.tolist(),
        "metadata": {
            "seed": seed,
            "total_demand": int(np.sum(demands)),
            "total_capacity": int(np.sum(vehicle_capacities))
        }
    }


def create_facility_location_problem(
    n_facilities: int = 5,
    n_customers: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample facility location problem.
    
    Args:
        n_facilities: Number of potential facility locations
        n_customers: Number of customers to serve
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Random facility and customer locations
    facility_locations = np.random.uniform(-20, 20, size=(n_facilities, 2))
    customer_locations = np.random.uniform(-15, 15, size=(n_customers, 2))
    
    # Facility opening costs
    opening_costs = np.random.uniform(50, 200, size=n_facilities)
    
    # Customer demands
    demands = np.random.uniform(5, 15, size=n_customers)
    
    # Facility capacities
    capacities = np.random.uniform(30, 80, size=n_facilities)
    
    # Transportation costs (distance-based)
    transport_costs = np.zeros((n_facilities, n_customers))
    for i in range(n_facilities):
        for j in range(n_customers):
            distance = np.linalg.norm(
                facility_locations[i] - customer_locations[j]
            )
            transport_costs[i, j] = distance * np.random.uniform(0.8, 1.2)
    
    return {
        "problem_type": "facility_location",
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "facility_locations": facility_locations.tolist(),
        "customer_locations": customer_locations.tolist(),
        "opening_costs": opening_costs.tolist(),
        "demands": demands.tolist(),
        "capacities": capacities.tolist(),
        "transport_costs": transport_costs.tolist(),
        "metadata": {
            "seed": seed,
            "total_demand": float(np.sum(demands)),
            "total_capacity": float(np.sum(capacities)),
            "avg_opening_cost": float(np.mean(opening_costs))
        }
    }


def create_max_cut_problem(
    n_vertices: int = 20,
    edge_probability: float = 0.3,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample MAX-CUT problem.
    
    Args:
        n_vertices: Number of vertices in the graph
        edge_probability: Probability of edge between any two vertices
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Generate random graph
    adjacency_matrix = np.zeros((n_vertices, n_vertices))
    edge_weights = {}
    edges = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if np.random.random() < edge_probability:
                weight = np.random.uniform(0.1, 2.0)
                adjacency_matrix[i, j] = weight
                adjacency_matrix[j, i] = weight
                edge_weights[(i, j)] = weight
                edges.append((i, j, weight))
    
    return {
        "problem_type": "max_cut",
        "n_vertices": n_vertices,
        "n_edges": len(edges),
        "edge_probability": edge_probability,
        "adjacency_matrix": adjacency_matrix.tolist(),
        "edges": edges,
        "metadata": {
            "seed": seed,
            "total_weight": float(np.sum(adjacency_matrix) / 2),
            "avg_degree": float(2 * len(edges) / n_vertices),
            "density": float(2 * len(edges) / (n_vertices * (n_vertices - 1)))
        }
    }


def create_quadratic_assignment_problem(
    n_facilities: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample Quadratic Assignment Problem.
    
    Args:
        n_facilities: Number of facilities/locations
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Flow matrix (between facilities)
    flow_matrix = np.random.randint(0, 20, size=(n_facilities, n_facilities))
    flow_matrix = (flow_matrix + flow_matrix.T) // 2  # Make symmetric
    np.fill_diagonal(flow_matrix, 0)
    
    # Distance matrix (between locations)
    locations = np.random.uniform(-10, 10, size=(n_facilities, 2))
    distance_matrix = np.zeros((n_facilities, n_facilities))
    
    for i in range(n_facilities):
        for j in range(n_facilities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    
    return {
        "problem_type": "quadratic_assignment",
        "n_facilities": n_facilities,
        "flow_matrix": flow_matrix.tolist(),
        "distance_matrix": distance_matrix.tolist(),
        "locations": locations.tolist(),
        "metadata": {
            "seed": seed,
            "total_flow": int(np.sum(flow_matrix) // 2),
            "avg_distance": float(np.mean(distance_matrix[distance_matrix > 0]))
        }
    }


def create_portfolio_optimization_problem(
    n_assets: int = 10,
    time_horizon: int = 252,  # Trading days in a year
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sample portfolio optimization problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        time_horizon: Number of time periods for returns
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing problem specification
    """
    np.random.seed(seed)
    
    # Generate correlated asset returns
    # Random correlation matrix
    A = np.random.randn(n_assets, n_assets)
    correlation_matrix = np.dot(A, A.T)
    correlation_matrix /= np.sqrt(np.outer(np.diag(correlation_matrix), np.diag(correlation_matrix)))
    
    # Expected returns (annualized)
    expected_returns = np.random.uniform(0.02, 0.15, size=n_assets)
    
    # Volatilities (annualized)
    volatilities = np.random.uniform(0.1, 0.4, size=n_assets)
    
    # Covariance matrix
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate sample return time series
    returns = np.random.multivariate_normal(
        expected_returns / time_horizon,
        covariance_matrix / time_horizon,
        size=time_horizon
    )
    
    return {
        "problem_type": "portfolio_optimization",
        "n_assets": n_assets,
        "time_horizon": time_horizon,
        "expected_returns": expected_returns.tolist(),
        "volatilities": volatilities.tolist(),
        "correlation_matrix": correlation_matrix.tolist(),
        "covariance_matrix": covariance_matrix.tolist(),
        "returns_history": returns.tolist(),
        "metadata": {
            "seed": seed,
            "avg_expected_return": float(np.mean(expected_returns)),
            "avg_volatility": float(np.mean(volatilities)),
            "max_correlation": float(np.max(correlation_matrix[np.triu_indices(n_assets, k=1)])),
            "min_correlation": float(np.min(correlation_matrix[np.triu_indices(n_assets, k=1)]))
        }
    }


# Collection of all problem generators
PROBLEM_GENERATORS = {
    "job_shop_scheduling": create_job_shop_scheduling_problem,
    "vehicle_routing": create_vehicle_routing_problem,
    "facility_location": create_facility_location_problem,
    "max_cut": create_max_cut_problem,
    "quadratic_assignment": create_quadratic_assignment_problem,
    "portfolio_optimization": create_portfolio_optimization_problem,
}


def get_sample_problem(problem_type: str, size: str = "small", **kwargs) -> Dict[str, Any]:
    """Get a sample problem of specified type and size.
    
    Args:
        problem_type: Type of problem to generate
        size: Problem size ('small', 'medium', 'large')
        **kwargs: Additional arguments to pass to the generator
        
    Returns:
        Problem specification dictionary
    """
    if problem_type not in PROBLEM_GENERATORS:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Default size parameters
    size_params = {
        "small": {"n_jobs": 5, "n_machines": 3, "n_customers": 10, "n_vehicles": 3,
                 "n_facilities": 5, "n_vertices": 10, "n_assets": 5},
        "medium": {"n_jobs": 10, "n_machines": 5, "n_customers": 20, "n_vehicles": 5,
                  "n_facilities": 10, "n_vertices": 25, "n_assets": 15},
        "large": {"n_jobs": 20, "n_machines": 8, "n_customers": 50, "n_vehicles": 10,
                 "n_facilities": 20, "n_vertices": 50, "n_assets": 30},
    }
    
    if size not in size_params:
        raise ValueError(f"Unknown size: {size}")
    
    # Merge size parameters with kwargs
    params = size_params[size].copy()
    params.update(kwargs)
    
    # Call the appropriate generator
    generator = PROBLEM_GENERATORS[problem_type]
    
    # Filter parameters that the generator accepts
    import inspect
    sig = inspect.signature(generator)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    return generator(**valid_params)


def list_available_problems() -> List[str]:
    """List all available problem types."""
    return list(PROBLEM_GENERATORS.keys())