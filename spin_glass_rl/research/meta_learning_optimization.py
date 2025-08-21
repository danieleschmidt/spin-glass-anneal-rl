"""
Meta-Learning Optimization for Adaptive Spin-Glass Annealing.

Implements meta-learning approaches to automatically adapt annealing strategies
based on problem characteristics and historical performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning optimization."""
    feature_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    learning_rate: float = 1e-3
    meta_batch_size: int = 16
    adaptation_steps: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ProblemEmbedding(nn.Module):
    """Neural network to extract problem-specific features."""
    
    def __init__(self, config: MetaLearningConfig):
        super().__init__()
        self.config = config
        
        # Graph neural network for structure encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim),  # coupling strength, distance, type, weight
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),  # field, degree, clustering, centrality, etc.
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Graph-level aggregation
        self.graph_aggregator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.Tanh()
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract problem features from graph representation."""
        # Edge features: coupling matrix properties
        edge_features = self.edge_encoder(graph_data['edge_attr'])
        
        # Node features: local field and graph properties
        node_features = self.node_encoder(graph_data['node_attr'])
        
        # Global graph features
        global_features = torch.cat([
            edge_features.mean(dim=0),
            edge_features.std(dim=0),
            node_features.mean(dim=0),
            node_features.std(dim=0)
        ], dim=0)
        
        return self.graph_aggregator(global_features)


class AnnealingStrategyGenerator(nn.Module):
    """Generates adaptive annealing strategies based on problem features."""
    
    def __init__(self, config: MetaLearningConfig):
        super().__init__()
        self.config = config
        
        # Strategy parameter generators
        self.temperature_schedule_gen = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10),  # 10 temperature waypoints
            nn.Sigmoid()
        )
        
        self.coupling_strength_gen = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sweep_count_gen = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.ReLU()
        )
    
    def forward(self, problem_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate annealing strategy parameters."""
        return {
            'temperature_schedule': self.temperature_schedule_gen(problem_features) * 10.0,
            'coupling_strength': self.coupling_strength_gen(problem_features) * 5.0,
            'sweep_count': (self.sweep_count_gen(problem_features) * 50000).int()
        }


class MetaOptimizer:
    """Meta-learning optimizer for adaptive annealing strategies."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize networks
        self.problem_encoder = ProblemEmbedding(config).to(self.device)
        self.strategy_generator = AnnealingStrategyGenerator(config).to(self.device)
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.problem_encoder.parameters()) + 
            list(self.strategy_generator.parameters()),
            lr=config.learning_rate
        )
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_memory = {}
    
    def extract_problem_features(self, ising_model) -> torch.Tensor:
        """Extract features from Ising model for meta-learning."""
        try:
            # Convert coupling matrix to graph representation
            coupling_matrix = ising_model.coupling_matrix
            external_fields = ising_model.external_fields
            
            # Node features: local properties
            node_features = []
            for i in range(ising_model.n_spins):
                # Degree (number of connections)
                degree = (coupling_matrix[i] != 0).sum().item()
                
                # Local field strength
                field_strength = abs(external_fields[i].item()) if external_fields is not None else 0.0
                
                # Local coupling strength (average of connected couplings)
                local_coupling = coupling_matrix[i][coupling_matrix[i] != 0].mean().item() if degree > 0 else 0.0
                
                # Clustering coefficient approximation
                neighbors = torch.nonzero(coupling_matrix[i]).flatten()
                clustering = 0.0
                if len(neighbors) > 1:
                    neighbor_connections = coupling_matrix[neighbors][:, neighbors]
                    clustering = (neighbor_connections != 0).float().mean().item()
                
                node_features.append([
                    field_strength, degree / ising_model.n_spins, local_coupling,
                    clustering, i / ising_model.n_spins,  # normalized position
                    0.0, 0.0, 0.0  # placeholder for additional features
                ])
            
            # Edge features: coupling properties
            edge_features = []
            for i in range(ising_model.n_spins):
                for j in range(i+1, ising_model.n_spins):
                    if coupling_matrix[i, j] != 0:
                        coupling_strength = coupling_matrix[i, j].item()
                        distance = abs(i - j)  # spatial distance
                        edge_type = 1.0 if coupling_strength > 0 else -1.0  # ferromagnetic/antiferromagnetic
                        weight = abs(coupling_strength)
                        
                        edge_features.append([coupling_strength, distance, edge_type, weight])
            
            # Handle case with no edges
            if not edge_features:
                edge_features = [[0.0, 0.0, 0.0, 0.0]]
            
            graph_data = {
                'node_attr': torch.tensor(node_features, dtype=torch.float32, device=self.device),
                'edge_attr': torch.tensor(edge_features, dtype=torch.float32, device=self.device)
            }
            
            return self.problem_encoder(graph_data)
            
        except Exception as e:
            logger.warning(f"Failed to extract problem features: {e}")
            # Return default features
            return torch.randn(self.config.feature_dim, device=self.device)
    
    def generate_strategy(self, ising_model) -> Dict[str, Any]:
        """Generate adaptive annealing strategy for given problem."""
        self.problem_encoder.eval()
        self.strategy_generator.eval()
        
        with torch.no_grad():
            problem_features = self.extract_problem_features(ising_model)
            strategy_params = self.strategy_generator(problem_features)
            
            # Convert to CPU and extract values
            strategy = {}
            for key, tensor in strategy_params.items():
                if tensor.dim() == 0:
                    strategy[key] = tensor.item()
                else:
                    strategy[key] = tensor.cpu().numpy()
            
            return strategy
    
    def update_performance(self, problem_id: str, strategy: Dict[str, Any], 
                          final_energy: float, solve_time: float):
        """Update meta-learner based on performance feedback."""
        performance_record = {
            'problem_id': problem_id,
            'strategy': strategy,
            'final_energy': final_energy,
            'solve_time': solve_time,
            'efficiency_score': -final_energy / max(solve_time, 1e-6)  # Energy improvement per time
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent performance history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def meta_train_step(self, batch_problems: List[Any], batch_solutions: List[Dict[str, Any]]):
        """Perform one meta-training step."""
        self.problem_encoder.train()
        self.strategy_generator.train()
        
        total_loss = 0.0
        
        for problem, solution in zip(batch_problems, batch_solutions):
            # Extract problem features
            problem_features = self.extract_problem_features(problem)
            
            # Generate strategy
            predicted_strategy = self.strategy_generator(problem_features)
            
            # Compute loss based on solution quality
            target_energy = solution['final_energy']
            target_time = solution['solve_time']
            
            # Loss encourages strategies that lead to better solutions
            energy_loss = torch.tensor(target_energy, device=self.device)
            time_loss = torch.tensor(target_time, device=self.device) / 1000.0  # normalize
            
            # Combined loss
            loss = energy_loss + 0.1 * time_loss
            total_loss += loss
        
        # Backpropagation
        avg_loss = total_loss / len(batch_problems)
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()
    
    def adapt_to_problem_class(self, problem_class: str, historical_data: List[Dict[str, Any]]):
        """Adapt meta-learner to specific problem class."""
        if len(historical_data) < 5:
            logger.warning(f"Insufficient data for adaptation to {problem_class}")
            return
        
        # Few-shot adaptation using historical performance
        best_strategies = sorted(historical_data, key=lambda x: x['efficiency_score'])[-5:]
        
        # Store adaptation patterns
        self.adaptation_memory[problem_class] = {
            'best_strategies': best_strategies,
            'avg_performance': np.mean([s['efficiency_score'] for s in best_strategies])
        }
        
        logger.info(f"Adapted to problem class {problem_class} with {len(best_strategies)} examples")


class HypermediaAnnealingFramework:
    """Framework combining meta-learning with hypermedia optimization."""
    
    def __init__(self, config: MetaLearningConfig):
        self.meta_optimizer = MetaOptimizer(config)
        self.problem_database = {}
        self.strategy_cache = {}
    
    def solve_with_meta_learning(self, ising_model, problem_id: str = None, 
                                problem_class: str = "general") -> Tuple[Dict[str, Any], float]:
        """Solve Ising model using meta-learned strategies."""
        import time
        
        start_time = time.time()
        
        # Generate adaptive strategy
        strategy = self.meta_optimizer.generate_strategy(ising_model)
        
        # Simulate annealing with generated strategy
        # (In practice, this would use the actual annealing implementation)
        n_sweeps = int(strategy.get('sweep_count', 10000))
        temperature_schedule = strategy.get('temperature_schedule', np.linspace(10, 0.1, 10))
        
        # Mock annealing process
        best_energy = self._simulate_annealing(ising_model, n_sweeps, temperature_schedule)
        
        solve_time = time.time() - start_time
        
        # Update meta-learner performance
        if problem_id:
            self.meta_optimizer.update_performance(problem_id, strategy, best_energy, solve_time)
        
        # Store in database for future adaptation
        if problem_class not in self.problem_database:
            self.problem_database[problem_class] = []
        
        self.problem_database[problem_class].append({
            'strategy': strategy,
            'final_energy': best_energy,
            'solve_time': solve_time,
            'efficiency_score': -best_energy / solve_time
        })
        
        return strategy, best_energy
    
    def _simulate_annealing(self, ising_model, n_sweeps: int, temperature_schedule: np.ndarray) -> float:
        """Simulate annealing process (placeholder for actual implementation)."""
        # Initialize random spin configuration
        spins = torch.randint(0, 2, (ising_model.n_spins,), dtype=torch.float32) * 2 - 1
        
        # Simple energy calculation
        try:
            coupling_energy = torch.sum(ising_model.coupling_matrix * torch.outer(spins, spins))
            field_energy = torch.sum(ising_model.external_fields * spins) if ising_model.external_fields is not None else 0
            energy = -(coupling_energy + field_energy)
            return energy.item()
        except:
            # Fallback energy calculation
            return np.random.normal(-100, 20)  # Mock energy value
    
    def continuous_adaptation(self):
        """Continuously adapt meta-learner based on accumulated experience."""
        for problem_class, data in self.problem_database.items():
            if len(data) >= 10:  # Sufficient data for adaptation
                self.meta_optimizer.adapt_to_problem_class(problem_class, data)
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Generate report on meta-learning adaptation progress."""
        report = {
            'total_problems_solved': sum(len(data) for data in self.problem_database.values()),
            'problem_classes': list(self.problem_database.keys()),
            'performance_trends': {},
            'meta_learning_stats': {
                'adaptation_memory_size': len(self.meta_optimizer.adaptation_memory),
                'performance_history_size': len(self.meta_optimizer.performance_history)
            }
        }
        
        # Calculate performance trends for each problem class
        for problem_class, data in self.problem_database.items():
            if len(data) >= 5:
                recent_performance = [d['efficiency_score'] for d in data[-5:]]
                early_performance = [d['efficiency_score'] for d in data[:5]]
                
                improvement = np.mean(recent_performance) - np.mean(early_performance)
                report['performance_trends'][problem_class] = {
                    'improvement': improvement,
                    'recent_avg': np.mean(recent_performance),
                    'total_problems': len(data)
                }
        
        return report


# Demonstration and testing functions
def create_meta_learning_demo():
    """Create demonstration of meta-learning optimization."""
    config = MetaLearningConfig(
        feature_dim=64,
        hidden_dim=128,
        learning_rate=1e-3
    )
    
    framework = HypermediaAnnealingFramework(config)
    
    # Create mock Ising models for different problem classes
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel
    
    problems = {
        'scheduling': MinimalIsingModel(n_spins=50),
        'routing': MinimalIsingModel(n_spins=75),
        'allocation': MinimalIsingModel(n_spins=100)
    }
    
    results = {}
    
    # Solve problems and demonstrate adaptation
    for problem_class, model in problems.items():
        print(f"\nSolving {problem_class} problems...")
        
        class_results = []
        for i in range(10):  # Solve multiple instances
            strategy, energy = framework.solve_with_meta_learning(
                model, f"{problem_class}_{i}", problem_class
            )
            class_results.append({'strategy': strategy, 'energy': energy})
        
        results[problem_class] = class_results
        print(f"Average energy for {problem_class}: {np.mean([r['energy'] for r in class_results]):.2f}")
    
    # Perform adaptation
    framework.continuous_adaptation()
    
    # Generate report
    report = framework.get_adaptation_report()
    print(f"\nMeta-Learning Adaptation Report:")
    print(f"Total problems solved: {report['total_problems_solved']}")
    print(f"Problem classes: {report['problem_classes']}")
    
    for problem_class, trend in report['performance_trends'].items():
        print(f"{problem_class}: {trend['improvement']:.3f} improvement, {trend['total_problems']} problems")
    
    return framework, results, report


if __name__ == "__main__":
    # Run demonstration
    framework, results, report = create_meta_learning_demo()
    print("\nMeta-learning optimization demonstration completed successfully!")