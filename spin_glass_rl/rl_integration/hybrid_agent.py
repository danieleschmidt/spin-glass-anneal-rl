"""Hybrid RL-Annealing agent that combines reinforcement learning with simulated annealing."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import warnings

from spin_glass_rl.core.ising_model import IsingModel
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.annealing.result import AnnealingResult
from spin_glass_rl.rl_integration.environment import SpinGlassEnv, SpinGlassEnvConfig


@dataclass
class HybridAgentConfig:
    """Configuration for hybrid RL-annealing agent."""
    
    # RL Agent parameters
    learning_rate: float = 3e-4
    hidden_dims: List[int] = (256, 128, 64)
    activation: str = "relu"
    
    # Training parameters
    batch_size: int = 32
    buffer_size: int = 10000
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Soft update factor
    update_frequency: int = 4
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Hybrid strategy parameters
    annealing_weight: float = 0.7  # Weight of annealing vs RL decisions
    adaptation_rate: float = 0.01  # How fast to adapt the weight
    
    # Performance thresholds
    rl_improvement_threshold: float = 0.1  # When RL is considered better
    annealing_fallback_threshold: float = 0.05  # When to fall back to pure annealing
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_type: str = "adam"
    gradient_clip: Optional[float] = 1.0


class DQNNetwork(nn.Module):
    """Deep Q-Network for learning annealing control policies."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.LayerNorm(hidden_dim),  # Layer normalization for stability
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in batch]
        
        states = torch.stack([exp[0] for exp in experiences]).to(self.device)
        actions = torch.tensor([exp[1] for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp[3] for exp in experiences]).to(self.device)
        dones = torch.tensor([exp[4] for exp in experiences], dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class HybridRLAnnealer:
    """
    Hybrid agent that combines reinforcement learning with simulated annealing.
    
    The agent learns when to rely on RL decisions vs traditional annealing,
    and adapts its strategy based on performance.
    """
    
    def __init__(self, config: HybridAgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize networks (will be set up when env is provided)
        self.q_network: Optional[DQNNetwork] = None
        self.target_network: Optional[DQNNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size, self.device)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        self.update_count = 0
        
        # Hybrid strategy state
        self.current_annealing_weight = config.annealing_weight
        self.rl_performance_history = deque(maxlen=100)
        self.annealing_performance_history = deque(maxlen=100)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        # Environment and annealer (to be set)
        self.env: Optional[SpinGlassEnv] = None
        self.pure_annealer: Optional[GPUAnnealer] = None
    
    def setup(self, env: SpinGlassEnv):
        """Setup the agent with the given environment."""
        self.env = env
        
        # Get dimensions from environment
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        
        # Initialize networks
        self.q_network = DQNNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Initialize pure annealer for comparison
        annealer_config = GPUAnnealerConfig(
            n_sweeps=env.config.annealer_sweeps * 5,  # More sweeps for pure annealing
            initial_temp=env.config.temperature_range[1],
            final_temp=env.config.temperature_range[0],
            schedule_type=ScheduleType.GEOMETRIC
        )
        self.pure_annealer = GPUAnnealer(annealer_config)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if not training or np.random.random() > self.epsilon:
            # Exploit: use Q-network
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
        else:
            # Explore: random action
            action = np.random.randint(self.env.action_space.n)
        
        return action
    
    def hybrid_decision(self, state: torch.Tensor, problem: IsingModel) -> Tuple[int, str]:
        """
        Make hybrid decision combining RL and annealing insights.
        
        Returns:
            Action and decision source ("rl", "annealing", or "hybrid")
        """
        # Get RL action recommendation
        rl_action = self.select_action(state, training=False)
        
        # Get annealing recommendation (heuristic based)
        annealing_action = self._get_annealing_recommendation(state, problem)
        
        # Decide which to use based on current strategy weight
        if np.random.random() < self.current_annealing_weight:
            # Use annealing recommendation
            return annealing_action, "annealing"
        else:
            # Use RL recommendation
            return rl_action, "rl"
    
    def _get_annealing_recommendation(self, state: torch.Tensor, problem: IsingModel) -> int:
        """Get action recommendation based on annealing heuristics."""
        # Extract features from state
        state_np = state.cpu().numpy()
        
        # Simple heuristics for temperature control
        current_energy = state_np[0] * 100.0  # Denormalize
        current_temp = state_np[3] * 10.0     # Denormalize
        
        # Heuristic: if temperature is high and energy is high, cool down faster
        # if temperature is low, cool down slower
        
        if current_temp > 5.0:
            # High temperature: aggressive cooling
            recommended_action = 2  # Lower temperature significantly
        elif current_temp > 1.0:
            # Medium temperature: moderate cooling
            recommended_action = 4  # Moderate cooling
        else:
            # Low temperature: careful cooling
            recommended_action = 6  # Slow cooling
        
        # Clamp to valid action space
        max_action = self.env.action_space.n - 1
        return min(recommended_action, max_action)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Compute Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.config.update_frequency == 0:
            self._soft_update_target_network()
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def _soft_update_target_network(self):
        """Soft update of target network."""
        for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * main_param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        if self.env is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        episode_reward = 0.0
        episode_length = 0
        episode_losses = []
        
        # Track decision sources
        decision_sources = {"rl": 0, "annealing": 0, "hybrid": 0}
        
        done = False
        while not done:
            # Select action using hybrid strategy
            action, source = self.hybrid_decision(state, self.env.current_problem)
            decision_sources[source] += 1
            
            # Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update exploration
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            self.steps_done += 1
        
        # Update performance tracking
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Evaluate hybrid strategy performance
        self._evaluate_strategy_performance(episode_reward, decision_sources)
        
        # Return training statistics
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "average_loss": np.mean(episode_losses) if episode_losses else 0.0,
            "epsilon": self.epsilon,
            "annealing_weight": self.current_annealing_weight,
            "decisions": decision_sources
        }
    
    def _evaluate_strategy_performance(self, episode_reward: float, decision_sources: Dict[str, int]):
        """Evaluate and adapt the hybrid strategy."""
        # Calculate performance metrics
        rl_ratio = decision_sources["rl"] / max(1, sum(decision_sources.values()))
        
        if rl_ratio > 0.5:
            # Mostly RL decisions
            self.rl_performance_history.append(episode_reward)
        else:
            # Mostly annealing decisions
            self.annealing_performance_history.append(episode_reward)
        
        # Adapt annealing weight based on recent performance
        if len(self.rl_performance_history) >= 10 and len(self.annealing_performance_history) >= 10:
            rl_performance = np.mean(list(self.rl_performance_history)[-10:])
            annealing_performance = np.mean(list(self.annealing_performance_history)[-10:])
            
            # If RL is performing better, decrease annealing weight
            if rl_performance > annealing_performance + self.config.rl_improvement_threshold:
                self.current_annealing_weight = max(0.1, 
                    self.current_annealing_weight - self.config.adaptation_rate)
            
            # If annealing is performing much better, increase annealing weight
            elif annealing_performance > rl_performance + self.config.annealing_fallback_threshold:
                self.current_annealing_weight = min(0.9,
                    self.current_annealing_weight + self.config.adaptation_rate)
    
    def pure_annealing_baseline(self, problem: IsingModel) -> AnnealingResult:
        """Run pure annealing as baseline for comparison."""
        if self.pure_annealer is None:
            raise RuntimeError("Pure annealer not initialized")
        
        # Reset problem to initial state
        problem.reset_to_random()
        
        # Run pure annealing
        result = self.pure_annealer.anneal(problem)
        return result
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the hybrid agent performance."""
        eval_rewards = []
        eval_lengths = []
        eval_energies = []
        
        # Disable training mode
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation
        
        try:
            for episode in range(n_episodes):
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                
                episode_reward = 0.0
                episode_length = 0
                
                done = False
                while not done:
                    action = self.select_action(state, training=False)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                    episode_reward += reward
                    episode_length += 1
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_energies.append(info.get("best_energy", 0.0))
        
        finally:
            # Restore training epsilon
            self.epsilon = original_epsilon
        
        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "mean_best_energy": np.mean(eval_energies),
            "success_rate": sum(1 for r in eval_rewards if r > 0) / len(eval_rewards)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": {
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "training_losses": self.training_losses
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.q_network is not None:
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load training stats
            stats = checkpoint.get("training_stats", {})
            self.episode_rewards = stats.get("episode_rewards", [])
            self.episode_lengths = stats.get("episode_lengths", [])
            self.training_losses = stats.get("training_losses", [])
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.steps_done,
            "current_epsilon": self.epsilon,
            "current_annealing_weight": self.current_annealing_weight,
            "mean_episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "mean_episode_length": np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0,
            "mean_training_loss": np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
        }


def create_hybrid_rl_annealer(env_config: Optional[SpinGlassEnvConfig] = None,
                             agent_config: Optional[HybridAgentConfig] = None) -> Tuple[HybridRLAnnealer, SpinGlassEnv]:
    """Factory function to create hybrid RL-annealing agent with environment."""
    
    if env_config is None:
        env_config = SpinGlassEnvConfig()
    
    if agent_config is None:
        agent_config = HybridAgentConfig()
    
    # Create environment
    env = SpinGlassEnv(env_config)
    
    # Create agent
    agent = HybridRLAnnealer(agent_config)
    agent.setup(env)
    
    return agent, env