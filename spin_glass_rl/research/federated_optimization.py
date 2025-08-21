"""
Federated Optimization for Distributed Spin-Glass Systems.

Implements federated learning and distributed optimization approaches for 
large-scale spin-glass problems across multiple nodes/devices.
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
from enum import Enum
import json

logger = logging.getLogger(__name__)


class FederationStrategy(Enum):
    """Different federated optimization strategies."""
    FEDERATED_AVERAGING = "fed_avg"
    FEDERATED_PROXIMAL = "fed_prox"
    FEDERATED_NOVA = "fed_nova"
    CONSENSUS_OPTIMIZATION = "consensus"
    HIERARCHICAL_FEDERATION = "hierarchical"


@dataclass
class FederatedConfig:
    """Configuration for federated optimization."""
    strategy: FederationStrategy = FederationStrategy.FEDERATED_AVERAGING
    n_clients: int = 5
    client_epochs: int = 5
    global_rounds: int = 20
    client_sample_ratio: float = 1.0
    learning_rate: float = 0.01
    mu: float = 0.1  # Proximal term weight
    communication_rounds: int = 100
    privacy_budget: float = 1.0  # Differential privacy
    device: str = "cpu"
    aggregation_weights: Optional[Dict[str, float]] = None


class FederatedClient(ABC):
    """Abstract base class for federated clients."""
    
    def __init__(self, client_id: str, local_data: Any, config: FederatedConfig):
        self.client_id = client_id
        self.local_data = local_data
        self.config = config
        self.device = torch.device(config.device)
        self.local_model_state = None
        self.global_model_state = None
        
    @abstractmethod
    def local_update(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform local optimization update."""
        pass
    
    @abstractmethod
    def evaluate_local(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on local data."""
        pass
    
    def add_privacy_noise(self, gradients: Dict[str, torch.Tensor], 
                         noise_scale: float = 0.1) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        noisy_gradients = {}
        for key, grad in gradients.items():
            noise = torch.randn_like(grad) * noise_scale
            noisy_gradients[key] = grad + noise
        return noisy_gradients


class SpinGlassClient(FederatedClient):
    """Federated client for spin-glass optimization."""
    
    def __init__(self, client_id: str, local_problems: List[Any], config: FederatedConfig):
        super().__init__(client_id, local_problems, config)
        self.local_problems = local_problems
        self.optimization_history = []
        
        # Local spin-glass optimizer
        self.local_optimizer = self._initialize_local_optimizer()
    
    def _initialize_local_optimizer(self):
        """Initialize local optimization algorithm."""
        from spin_glass_rl.core.minimal_ising import MinimalAnnealer
        return MinimalAnnealer()
    
    def local_update(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform local spin-glass optimization."""
        # Update global state
        self.global_model_state = global_model_state
        
        # Local optimization on client's problems
        local_solutions = []
        local_energies = []
        
        for problem in self.local_problems:
            # Incorporate global knowledge into local optimization
            if global_model_state is not None:
                # Use global state to initialize or guide local optimization
                initial_state = self._extract_initial_state(global_model_state, problem)
            else:
                initial_state = None
            
            # Perform local optimization
            result = self.local_optimizer.anneal(problem, n_sweeps=1000, initial_state=initial_state)
            local_solutions.append(result['best_configuration'])
            local_energies.append(result['best_energy'])
        
        # Aggregate local results into model update
        local_model_state = self._aggregate_local_solutions(local_solutions, local_energies)
        
        # Store optimization history
        self.optimization_history.append({
            'round': len(self.optimization_history),
            'local_energies': local_energies,
            'avg_energy': np.mean(local_energies),
            'best_energy': min(local_energies)
        })
        
        return local_model_state
    
    def _extract_initial_state(self, global_state: Dict[str, torch.Tensor], problem) -> Optional[torch.Tensor]:
        """Extract initial spin state from global model state."""
        if 'best_configurations' in global_state and len(global_state['best_configurations']) > 0:
            # Use closest global configuration as initialization
            global_configs = global_state['best_configurations']
            if len(global_configs) > 0:
                # For simplicity, use first configuration (could be more sophisticated)
                return global_configs[0][:problem.n_spins]  # Truncate to problem size
        return None
    
    def _aggregate_local_solutions(self, solutions: List[torch.Tensor], 
                                 energies: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate local solutions into model state."""
        if not solutions:
            return {}
        
        # Find best local solution
        best_idx = np.argmin(energies)
        best_solution = solutions[best_idx]
        
        # Create ensemble of solutions
        ensemble_solution = torch.stack(solutions).float().mean(dim=0)
        ensemble_solution = torch.sign(ensemble_solution)  # Convert to ±1
        
        return {
            'best_configuration': best_solution,
            'ensemble_configuration': ensemble_solution,
            'local_energies': torch.tensor(energies),
            'client_id': torch.tensor([hash(self.client_id) % 1000]),  # Identifier
            'n_problems': torch.tensor([len(solutions)])
        }
    
    def evaluate_local(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model state on local problems."""
        if 'best_configuration' not in model_state:
            return {'avg_energy': float('inf'), 'n_problems': 0}
        
        configuration = model_state['best_configuration']
        energies = []
        
        for problem in self.local_problems:
            # Evaluate configuration on local problem
            if len(configuration) >= problem.n_spins:
                local_config = configuration[:problem.n_spins]
                energy = self._calculate_energy(problem, local_config)
                energies.append(energy)
        
        return {
            'avg_energy': np.mean(energies) if energies else float('inf'),
            'best_energy': min(energies) if energies else float('inf'),
            'n_problems': len(energies)
        }
    
    def _calculate_energy(self, problem, configuration: torch.Tensor) -> float:
        """Calculate Ising energy for configuration."""
        try:
            coupling_energy = torch.sum(problem.coupling_matrix * torch.outer(configuration, configuration))
            field_energy = torch.sum(problem.external_fields * configuration) if hasattr(problem, 'external_fields') and problem.external_fields is not None else 0
            return -(coupling_energy + field_energy).item()
        except:
            return float('inf')


class FederatedServer:
    """Federated server for coordinating distributed optimization."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_model_state = None
        self.clients = {}
        self.communication_history = []
        self.global_best_energy = float('inf')
        self.global_best_configuration = None
        
    def register_client(self, client: FederatedClient):
        """Register a new federated client."""
        self.clients[client.client_id] = client
        logger.info(f"Registered client {client.client_id}")
    
    def federated_optimization(self) -> Dict[str, Any]:
        """Perform federated optimization across all clients."""
        optimization_results = {
            'global_rounds': [],
            'communication_costs': [],
            'convergence_metrics': [],
            'client_contributions': {}
        }
        
        print(f"Starting federated optimization with {len(self.clients)} clients...")
        
        for round_idx in range(self.config.global_rounds):
            print(f"\nGlobal Round {round_idx + 1}/{self.config.global_rounds}")
            
            # Select participating clients
            participating_clients = self._select_clients()
            
            # Collect local updates
            client_updates = {}
            client_metrics = {}
            
            for client_id in participating_clients:
                client = self.clients[client_id]
                
                # Send global model to client
                local_update = client.local_update(self.global_model_state)
                client_updates[client_id] = local_update
                
                # Evaluate local performance
                local_metrics = client.evaluate_local(local_update)
                client_metrics[client_id] = local_metrics
                
                print(f"  Client {client_id}: avg_energy={local_metrics.get('avg_energy', 'N/A'):.3f}")
            
            # Aggregate client updates
            self.global_model_state = self._aggregate_updates(client_updates, client_metrics)
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            
            # Track optimization progress
            round_result = {
                'round': round_idx,
                'global_metrics': global_metrics,
                'client_metrics': client_metrics,
                'participating_clients': participating_clients
            }
            optimization_results['global_rounds'].append(round_result)
            
            # Update global best
            if global_metrics['best_energy'] < self.global_best_energy:
                self.global_best_energy = global_metrics['best_energy']
                if 'best_configuration' in self.global_model_state:
                    self.global_best_configuration = self.global_model_state['best_configuration'].clone()
            
            print(f"  Global best energy: {self.global_best_energy:.4f}")
            
            # Communication cost tracking
            comm_cost = len(participating_clients) * self._estimate_communication_cost()
            optimization_results['communication_costs'].append(comm_cost)
        
        # Final results
        optimization_results['final_global_state'] = self.global_model_state
        optimization_results['global_best_energy'] = self.global_best_energy
        optimization_results['global_best_configuration'] = self.global_best_configuration
        
        return optimization_results
    
    def _select_clients(self) -> List[str]:
        """Select participating clients for current round."""
        all_clients = list(self.clients.keys())
        n_selected = max(1, int(len(all_clients) * self.config.client_sample_ratio))
        
        # Random selection (could be more sophisticated)
        selected = np.random.choice(all_clients, n_selected, replace=False)
        return selected.tolist()
    
    def _aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                          client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using selected federation strategy."""
        if not client_updates:
            return self.global_model_state or {}
        
        if self.config.strategy == FederationStrategy.FEDERATED_AVERAGING:
            return self._federated_averaging(client_updates, client_metrics)
        elif self.config.strategy == FederationStrategy.CONSENSUS_OPTIMIZATION:
            return self._consensus_optimization(client_updates, client_metrics)
        else:
            # Default to federated averaging
            return self._federated_averaging(client_updates, client_metrics)
    
    def _federated_averaging(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                           client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """Perform federated averaging of client updates."""
        aggregated_state = {}
        
        # Collect all configurations
        configurations = []
        energies = []
        weights = []
        
        for client_id, update in client_updates.items():
            if 'best_configuration' in update:
                configurations.append(update['best_configuration'])
                
                # Weight by inverse energy (better solutions get higher weight)
                client_energy = client_metrics[client_id].get('best_energy', float('inf'))
                if client_energy != float('inf'):
                    weight = 1.0 / (1.0 + abs(client_energy))  # Inverse energy weighting
                else:
                    weight = 0.1  # Small weight for invalid solutions
                
                weights.append(weight)
                energies.append(client_energy)
        
        if configurations:
            # Weighted average of configurations
            weights = torch.tensor(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Handle different configuration sizes
            max_size = max(len(config) for config in configurations)
            padded_configs = []
            
            for config in configurations:
                if len(config) < max_size:
                    # Pad with random values
                    padding = torch.randint(0, 2, (max_size - len(config),)) * 2 - 1
                    padded_config = torch.cat([config, padding])
                else:
                    padded_config = config[:max_size]
                padded_configs.append(padded_config.float())
            
            # Weighted average
            stacked_configs = torch.stack(padded_configs)
            averaged_config = torch.sum(stacked_configs * weights.unsqueeze(1), dim=0)
            
            # Convert back to ±1
            averaged_config = torch.sign(averaged_config)
            averaged_config[averaged_config == 0] = 1  # Handle zeros
            
            aggregated_state['best_configuration'] = averaged_config
            aggregated_state['average_energy'] = torch.tensor(np.mean(energies))
            aggregated_state['best_energy'] = torch.tensor(min(energies))
        
        return aggregated_state
    
    def _consensus_optimization(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                              client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """Perform consensus-based optimization."""
        # Find client with best performance
        best_client = min(client_metrics.keys(), key=lambda k: client_metrics[k].get('best_energy', float('inf')))
        best_update = client_updates[best_client]
        
        # Use best client's solution as global state
        return best_update
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model across all clients."""
        if self.global_model_state is None:
            return {'avg_energy': float('inf'), 'best_energy': float('inf')}
        
        all_energies = []
        
        for client in self.clients.values():
            metrics = client.evaluate_local(self.global_model_state)
            if metrics['n_problems'] > 0:
                all_energies.append(metrics['avg_energy'])
        
        if all_energies:
            return {
                'avg_energy': np.mean(all_energies),
                'best_energy': min(all_energies),
                'std_energy': np.std(all_energies),
                'n_evaluations': len(all_energies)
            }
        else:
            return {'avg_energy': float('inf'), 'best_energy': float('inf')}
    
    def _estimate_communication_cost(self) -> int:
        """Estimate communication cost for current round."""
        # Simple model: cost proportional to model size
        if self.global_model_state:
            total_params = sum(tensor.numel() for tensor in self.global_model_state.values())
            return total_params * 4  # 4 bytes per float32
        return 0


class HierarchicalFederatedOptimizer:
    """Hierarchical federated optimizer for large-scale problems."""
    
    def __init__(self, config: FederatedConfig, hierarchy_levels: int = 2):
        self.config = config
        self.hierarchy_levels = hierarchy_levels
        self.regional_servers = {}
        self.global_server = None
        
    def setup_hierarchy(self, clients: List[FederatedClient]):
        """Setup hierarchical federated structure."""
        # Divide clients into regions
        clients_per_region = len(clients) // self.hierarchy_levels
        
        for level in range(self.hierarchy_levels):
            start_idx = level * clients_per_region
            end_idx = start_idx + clients_per_region if level < self.hierarchy_levels - 1 else len(clients)
            
            regional_clients = clients[start_idx:end_idx]
            
            # Create regional server
            regional_config = FederatedConfig(
                n_clients=len(regional_clients),
                global_rounds=self.config.global_rounds // 2,
                client_epochs=self.config.client_epochs
            )
            regional_server = FederatedServer(regional_config)
            
            # Register clients to regional server
            for client in regional_clients:
                regional_server.register_client(client)
            
            self.regional_servers[f"region_{level}"] = regional_server
        
        # Create global server for regional coordination
        global_config = FederatedConfig(
            n_clients=len(self.regional_servers),
            global_rounds=self.config.global_rounds // 2
        )
        self.global_server = FederatedServer(global_config)
    
    def hierarchical_optimization(self) -> Dict[str, Any]:
        """Perform hierarchical federated optimization."""
        results = {
            'regional_results': {},
            'global_coordination': [],
            'final_performance': {}
        }
        
        print("Starting hierarchical federated optimization...")
        
        # Phase 1: Regional optimization
        print("\nPhase 1: Regional Optimization")
        for region_id, regional_server in self.regional_servers.items():
            print(f"Optimizing {region_id}...")
            regional_result = regional_server.federated_optimization()
            results['regional_results'][region_id] = regional_result
        
        # Phase 2: Global coordination
        print("\nPhase 2: Global Coordination")
        
        # Create pseudo-clients representing regional results
        regional_pseudo_clients = []
        for region_id, regional_server in self.regional_servers.items():
            pseudo_client = RegionalPseudoClient(
                region_id, regional_server.global_model_state, self.config
            )
            regional_pseudo_clients.append(pseudo_client)
            self.global_server.register_client(pseudo_client)
        
        # Global coordination rounds
        global_result = self.global_server.federated_optimization()
        results['global_coordination'] = global_result
        
        # Final evaluation
        final_metrics = self._evaluate_hierarchical_performance()
        results['final_performance'] = final_metrics
        
        return results
    
    def _evaluate_hierarchical_performance(self) -> Dict[str, Any]:
        """Evaluate final hierarchical optimization performance."""
        # Collect metrics from all regional servers
        all_energies = []
        total_communication_cost = 0
        
        for regional_server in self.regional_servers.values():
            if regional_server.global_model_state:
                metrics = regional_server._evaluate_global_model()
                if metrics['best_energy'] != float('inf'):
                    all_energies.append(metrics['best_energy'])
        
        return {
            'overall_best_energy': min(all_energies) if all_energies else float('inf'),
            'regional_diversity': np.std(all_energies) if len(all_energies) > 1 else 0.0,
            'n_regions': len(self.regional_servers),
            'hierarchy_levels': self.hierarchy_levels
        }


class RegionalPseudoClient(FederatedClient):
    """Pseudo-client representing regional optimization results."""
    
    def __init__(self, region_id: str, regional_state: Dict[str, torch.Tensor], config: FederatedConfig):
        super().__init__(region_id, None, config)
        self.regional_state = regional_state
    
    def local_update(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return regional state as local update."""
        return self.regional_state or {}
    
    def evaluate_local(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model state (simplified for pseudo-client)."""
        if model_state and 'best_energy' in model_state:
            return {'avg_energy': model_state['best_energy'].item(), 'n_problems': 1}
        return {'avg_energy': float('inf'), 'n_problems': 0}


# Demonstration and testing functions
def create_federated_optimization_demo():
    """Create demonstration of federated optimization."""
    print("Creating Federated Optimization Demo...")
    
    # Configuration
    config = FederatedConfig(
        strategy=FederationStrategy.FEDERATED_AVERAGING,
        n_clients=5,
        client_epochs=3,
        global_rounds=10,
        client_sample_ratio=0.8
    )
    
    # Create federated server
    server = FederatedServer(config)
    
    # Create test problems for each client
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel
    
    clients = []
    for i in range(config.n_clients):
        # Each client has different problem instances
        client_problems = [
            MinimalIsingModel(n_spins=15 + i * 2) for _ in range(3 + i)
        ]
        
        client = SpinGlassClient(f"client_{i}", client_problems, config)
        clients.append(client)
        server.register_client(client)
    
    # Run federated optimization
    print(f"\nRunning federated optimization with {len(clients)} clients...")
    results = server.federated_optimization()
    
    # Display results
    print("\n" + "="*60)
    print("FEDERATED OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Global best energy: {results['global_best_energy']:.4f}")
    print(f"Total communication rounds: {len(results['global_rounds'])}")
    print(f"Total communication cost: {sum(results['communication_costs'])} bytes")
    
    # Client performance analysis
    print(f"\nClient Performance Analysis:")
    for round_result in results['global_rounds'][-3:]:  # Last 3 rounds
        round_num = round_result['round']
        print(f"\nRound {round_num + 1}:")
        for client_id, metrics in round_result['client_metrics'].items():
            avg_energy = metrics.get('avg_energy', 'N/A')
            best_energy = metrics.get('best_energy', 'N/A')
            print(f"  {client_id}: avg={avg_energy:.3f}, best={best_energy:.3f}")
    
    return server, results


def create_hierarchical_demo():
    """Create demonstration of hierarchical federated optimization."""
    print("Creating Hierarchical Federated Optimization Demo...")
    
    # Configuration
    config = FederatedConfig(
        strategy=FederationStrategy.FEDERATED_AVERAGING,
        global_rounds=8,
        client_epochs=2
    )
    
    # Create clients
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel
    
    clients = []
    for i in range(8):  # 8 clients for 2-level hierarchy
        client_problems = [MinimalIsingModel(n_spins=12 + i) for _ in range(2)]
        client = SpinGlassClient(f"client_{i}", client_problems, config)
        clients.append(client)
    
    # Create hierarchical optimizer
    hierarchical_optimizer = HierarchicalFederatedOptimizer(config, hierarchy_levels=2)
    hierarchical_optimizer.setup_hierarchy(clients)
    
    # Run hierarchical optimization
    print("Running hierarchical federated optimization...")
    results = hierarchical_optimizer.hierarchical_optimization()
    
    # Display results
    print("\n" + "="*60)
    print("HIERARCHICAL FEDERATED OPTIMIZATION RESULTS")
    print("="*60)
    
    final_perf = results['final_performance']
    print(f"Overall best energy: {final_perf['overall_best_energy']:.4f}")
    print(f"Regional diversity: {final_perf['regional_diversity']:.4f}")
    print(f"Number of regions: {final_perf['n_regions']}")
    
    # Regional results
    print(f"\nRegional Results:")
    for region_id, regional_result in results['regional_results'].items():
        best_energy = regional_result['global_best_energy']
        rounds = len(regional_result['global_rounds'])
        print(f"  {region_id}: best_energy={best_energy:.4f}, rounds={rounds}")
    
    return hierarchical_optimizer, results


if __name__ == "__main__":
    # Run demonstrations
    print("Starting Federated Optimization Demonstrations...\n")
    
    # Standard federated optimization demo
    server, fed_results = create_federated_optimization_demo()
    
    print("\n" + "="*80)
    
    # Hierarchical federated optimization demo
    hierarchical_opt, hier_results = create_hierarchical_demo()
    
    print("\nFederated optimization demonstrations completed successfully!")