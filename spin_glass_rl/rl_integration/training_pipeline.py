"""Comprehensive RL training pipeline for spin-glass guided optimization."""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from collections import deque, defaultdict

from spin_glass_rl.core.ising_model import IsingModel
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.base import ProblemTemplate
from spin_glass_rl.rl_integration.environment import SpinGlassEnvironment
from spin_glass_rl.rl_integration.hybrid_agent import HybridRLAnnealingAgent
from spin_glass_rl.rl_integration.reward_shaping import RewardShaper, RewardType
from spin_glass_rl.utils.robust_logging import get_logger
from spin_glass_rl.utils.monitoring import PerformanceMonitor

logger = get_logger("rl_training")


@dataclass
class TrainingConfig:
    """Configuration for RL training pipeline."""
    # Training parameters
    n_episodes: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    
    # Experience replay
    memory_size: int = 100000
    min_memory_size: int = 1000
    target_update_frequency: int = 100
    
    # Annealing integration
    annealing_steps_per_action: int = 10
    temperature_control: bool = True
    adaptive_annealing: bool = True
    
    # Reward shaping
    reward_type: RewardType = RewardType.ENERGY_IMPROVEMENT
    reward_scaling: float = 1.0
    curriculum_learning: bool = True
    
    # Logging and evaluation
    eval_frequency: int = 50
    save_frequency: int = 200
    log_frequency: int = 10
    
    # Device and optimization
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixed_precision: bool = True
    gradient_clipping: float = 1.0


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    episode: int
    total_reward: float
    episode_length: int
    average_energy: float
    best_energy: float
    exploration_rate: float
    learning_rate: float
    loss: float
    q_values_mean: float
    annealing_success_rate: float
    convergence_time: float


class ExperienceBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.ones(capacity, dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, priority: float = 1.0):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, use_priorities: bool = False) -> Tuple:
        """Sample batch from buffer."""
        if use_priorities and self.size > 0:
            # Prioritized experience replay
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        
        return (
            self.states[indices],
            self.actions[indices], 
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update sample priorities for prioritized replay."""
        self.priorities[indices] = priorities
    
    def __len__(self) -> int:
        return self.size


class RLTrainingPipeline:
    """Comprehensive RL training pipeline for spin-glass optimization."""
    
    def __init__(self, 
                 problem_factory: Callable[[], ProblemTemplate],
                 config: Optional[TrainingConfig] = None):
        
        self.problem_factory = problem_factory
        self.config = config or TrainingConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._setup_environment()
        self._setup_agent()
        self._setup_reward_shaper()
        self._setup_experience_buffer()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_performance = float('-inf')
        
        # Metrics tracking
        self.training_metrics: List[TrainingMetrics] = []
        self.performance_history = deque(maxlen=100)
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
    def _setup_environment(self):
        """Setup RL environment."""
        # Create sample problem to get dimensions
        sample_problem = self.problem_factory()
        sample_instance = sample_problem.generate_random_instance(n_locations=20)
        sample_ising = sample_problem.encode_to_ising()
        
        self.env = SpinGlassEnvironment(
            problem_generator=self.problem_factory,
            max_episode_steps=self.config.max_steps_per_episode,
            reward_type=self.config.reward_type
        )
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        logger.info(f"Environment setup: state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def _setup_agent(self):
        """Setup hybrid RL agent."""
        # Base annealing configuration
        annealing_config = GPUAnnealerConfig(
            n_sweeps=self.config.annealing_steps_per_action,
            initial_temp=5.0,
            final_temp=0.1,
            schedule_type=ScheduleType.GEOMETRIC,
            device=str(self.device)
        )
        
        self.agent = HybridRLAnnealingAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=self.config.learning_rate,
            annealing_config=annealing_config,
            device=self.device
        )
        
        logger.info("Hybrid RL agent initialized")
    
    def _setup_reward_shaper(self):
        """Setup reward shaping system."""
        self.reward_shaper = RewardShaper(
            reward_type=self.config.reward_type,
            scaling_factor=self.config.reward_scaling,
            baseline_subtraction=True,
            temporal_smoothing=True
        )
        
        logger.info(f"Reward shaper setup with type: {self.config.reward_type}")
    
    def _setup_experience_buffer(self):
        """Setup experience replay buffer."""
        self.experience_buffer = ExperienceBuffer(
            capacity=self.config.memory_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        logger.info(f"Experience buffer initialized with capacity: {self.config.memory_size}")
    
    def train(self, save_dir: Optional[Path] = None) -> List[TrainingMetrics]:
        """Run complete training pipeline."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting RL training for {self.config.n_episodes} episodes")
        
        with LoggingContext("rl_training"):
            for episode in range(self.config.n_episodes):
                self.current_episode = episode
                
                # Run single episode
                metrics = self._run_episode()
                self.training_metrics.append(metrics)
                self.performance_history.append(metrics.total_reward)
                
                # Update exploration rate
                self._update_exploration_rate()
                
                # Periodic evaluation and saving
                if episode % self.config.eval_frequency == 0:
                    self._evaluate_performance()
                
                if save_dir and episode % self.config.save_frequency == 0:
                    self._save_checkpoint(save_dir / f"checkpoint_episode_{episode}.pkl")
                
                # Logging
                if episode % self.config.log_frequency == 0:
                    self._log_progress(metrics)
        
        logger.info("Training completed")
        
        if save_dir:
            self._save_final_model(save_dir / "final_model.pkl")
        
        return self.training_metrics
    
    def _run_episode(self) -> TrainingMetrics:
        """Run single training episode."""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_energies = []
        episode_losses = []
        episode_q_values = []
        
        annealing_successes = 0
        annealing_attempts = 0
        
        start_time = time.time()
        
        for step in range(self.config.max_steps_per_episode):
            # Agent selects action
            action, q_value = self.agent.select_action(state, training=True)
            episode_q_values.append(q_value)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Reward shaping
            shaped_reward = self.reward_shaper.shape_reward(
                reward, state, action, next_state, info
            )
            
            # Store experience
            self.experience_buffer.add(
                state, action, shaped_reward, next_state, done, 
                priority=abs(shaped_reward) + 1e-6  # Simple priority
            )
            
            # Update metrics
            episode_reward += shaped_reward
            episode_length += 1
            self.total_steps += 1
            
            if 'energy' in info:
                episode_energies.append(info['energy'])
            
            if 'annealing_success' in info:
                annealing_attempts += 1
                if info['annealing_success']:
                    annealing_successes += 1
            
            # Training update
            if (len(self.experience_buffer) >= self.config.min_memory_size and
                self.total_steps % 4 == 0):  # Update every 4 steps
                
                loss = self._update_agent()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update target network
            if self.total_steps % self.config.target_update_frequency == 0:
                self.agent.update_target_network()
            
            state = next_state
            
            if done:
                break
        
        convergence_time = time.time() - start_time
        
        # Compute episode metrics
        metrics = TrainingMetrics(
            episode=self.current_episode,
            total_reward=episode_reward,
            episode_length=episode_length,
            average_energy=np.mean(episode_energies) if episode_energies else 0.0,
            best_energy=min(episode_energies) if episode_energies else 0.0,
            exploration_rate=self.agent.get_exploration_rate(),
            learning_rate=self.agent.get_learning_rate(),
            loss=np.mean(episode_losses) if episode_losses else 0.0,
            q_values_mean=np.mean(episode_q_values) if episode_q_values else 0.0,
            annealing_success_rate=annealing_successes / max(annealing_attempts, 1),
            convergence_time=convergence_time
        )
        
        return metrics
    
    def _update_agent(self) -> Optional[float]:
        """Update agent using experience replay."""
        if len(self.experience_buffer) < self.config.min_memory_size:
            return None
        
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size, use_priorities=True)
        states, actions, rewards, next_states, dones, indices = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute loss and update
        loss, td_errors = self.agent.update(states, actions, rewards, next_states, dones)
        
        # Update priorities
        if td_errors is not None:
            priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            self.experience_buffer.update_priorities(indices, priorities)
        
        return loss
    
    def _update_exploration_rate(self):
        """Update exploration rate using decay schedule."""
        current_epsilon = self.agent.get_exploration_rate()
        new_epsilon = max(
            self.config.epsilon_end,
            current_epsilon * self.config.epsilon_decay
        )
        self.agent.set_exploration_rate(new_epsilon)
    
    def _evaluate_performance(self):
        """Evaluate current policy performance."""
        logger.info(f"Evaluating performance at episode {self.current_episode}")
        
        # Run evaluation episodes without exploration
        eval_rewards = []
        eval_energies = []
        
        for _ in range(5):  # 5 evaluation episodes
            state = self.env.reset()
            eval_reward = 0.0
            eval_episode_energies = []
            
            for _ in range(self.config.max_steps_per_episode):
                action, _ = self.agent.select_action(state, training=False)  # No exploration
                state, reward, done, info = self.env.step(action)
                eval_reward += reward
                
                if 'energy' in info:
                    eval_episode_energies.append(info['energy'])
                
                if done:
                    break
            
            eval_rewards.append(eval_reward)
            eval_energies.extend(eval_episode_energies)
        
        mean_eval_reward = np.mean(eval_rewards)
        mean_eval_energy = np.mean(eval_energies) if eval_energies else 0.0
        
        # Update best performance
        if mean_eval_reward > self.best_performance:
            self.best_performance = mean_eval_reward
            logger.info(f"New best performance: {self.best_performance:.4f}")
        
        logger.info(f"Evaluation - Mean Reward: {mean_eval_reward:.4f}, "
                   f"Mean Energy: {mean_eval_energy:.6f}")
    
    def _log_progress(self, metrics: TrainingMetrics):
        """Log training progress."""
        recent_rewards = list(self.performance_history)[-10:]  # Last 10 episodes
        mean_recent_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        logger.info(f"Episode {metrics.episode:4d} | "
                   f"Reward: {metrics.total_reward:8.2f} | "
                   f"Recent Avg: {mean_recent_reward:8.2f} | "
                   f"Best Energy: {metrics.best_energy:8.6f} | "
                   f"Exploration: {metrics.exploration_rate:.3f} | "
                   f"Loss: {metrics.loss:.6f}")
    
    def _save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.current_episode,
            'agent_state': self.agent.state_dict(),
            'config': self.config,
            'metrics': self.training_metrics,
            'best_performance': self.best_performance
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {path}")
    
    def _save_final_model(self, path: Path):
        """Save final trained model."""
        model_data = {
            'agent_state': self.agent.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'final_performance': self.best_performance,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Final model saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.current_episode = checkpoint['episode']
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.training_metrics = checkpoint['metrics']
        self.best_performance = checkpoint['best_performance']
        
        logger.info(f"Checkpoint loaded from: {path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_metrics:
            return {"status": "No training data available"}
        
        rewards = [m.total_reward for m in self.training_metrics]
        energies = [m.best_energy for m in self.training_metrics if m.best_energy != 0]
        losses = [m.loss for m in self.training_metrics if m.loss != 0]
        
        return {
            "episodes_completed": len(self.training_metrics),
            "total_steps": self.total_steps,
            "best_performance": self.best_performance,
            "final_exploration_rate": self.training_metrics[-1].exploration_rate,
            
            # Reward statistics
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            
            # Energy statistics
            "mean_energy": np.mean(energies) if energies else 0.0,
            "best_energy": min(energies) if energies else 0.0,
            
            # Training statistics
            "mean_loss": np.mean(losses) if losses else 0.0,
            "mean_episode_length": np.mean([m.episode_length for m in self.training_metrics]),
            "mean_annealing_success_rate": np.mean([m.annealing_success_rate for m in self.training_metrics]),
            
            # Learning progress
            "reward_improvement": rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
            "convergence_indicator": np.std(rewards[-50:]) if len(rewards) >= 50 else float('inf')
        }


def demo_rl_training():
    """Demonstration of RL training pipeline."""
    from spin_glass_rl.problems.routing import TSPProblem
    
    print("RL Training Pipeline Demo")
    print("=" * 40)
    
    # Problem factory
    def tsp_factory():
        problem = TSPProblem()
        problem.generate_random_instance(n_locations=10)
        return problem
    
    # Training configuration
    config = TrainingConfig(
        n_episodes=100,
        max_steps_per_episode=50,
        learning_rate=1e-3,
        epsilon_decay=0.99,
        eval_frequency=20,
        log_frequency=5
    )
    
    # Create training pipeline
    pipeline = RLTrainingPipeline(tsp_factory, config)
    
    # Run training
    metrics = pipeline.train()
    
    # Print summary
    summary = pipeline.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_rl_training()