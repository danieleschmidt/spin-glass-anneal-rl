"""Reinforcement Learning environment wrapper for spin-glass optimization."""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from spin_glass_rl.core.ising_model import IsingModel
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig
from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.problems.base import ProblemTemplate


@dataclass
class SpinGlassEnvConfig:
    """Configuration for spin-glass RL environment."""
    
    # Environment parameters
    n_spins: int = 100
    max_steps: int = 1000
    temperature_range: Tuple[float, float] = (0.01, 10.0)
    
    # Action space configuration
    action_type: str = "discrete"  # "discrete", "continuous", "hybrid"
    n_discrete_actions: int = 10  # for discrete action space
    
    # Observation space configuration
    observation_type: str = "full"  # "full", "local", "global_features"
    local_neighborhood_size: int = 5  # for local observations
    
    # Reward configuration
    reward_type: str = "energy_delta"  # "energy_delta", "acceptance_rate", "mixed"
    energy_scale_factor: float = 1.0
    exploration_bonus: float = 0.01
    
    # Annealing configuration
    annealer_sweeps: int = 10  # sweeps per RL step
    adaptive_temperature: bool = True
    
    # Problem configuration
    problem_generator: Optional[str] = None  # "random_ising", "tsp", "vrp", etc.
    problem_size_range: Tuple[int, int] = (50, 200)
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SpinGlassEnv(gym.Env):
    """
    Gymnasium environment for reinforcement learning on spin-glass optimization.
    
    The RL agent learns to guide the annealing process by making decisions
    about temperature schedules, spin update strategies, or problem decomposition.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: SpinGlassEnvConfig):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize problem and annealer
        self.current_problem: Optional[IsingModel] = None
        self.annealer: Optional[GPUAnnealer] = None
        self.current_step = 0
        self.episode_rewards = []
        
        # Environment state
        self.current_energy = 0.0
        self.best_energy = float('inf')
        self.current_temperature = config.temperature_range[1]  # Start hot
        self.energy_history = []
        self.temperature_history = []
        self.acceptance_history = []
        
        # Define action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()
        
        # Initialize annealer
        self._setup_annealer()
    
    def _setup_action_space(self):
        """Setup the action space based on configuration."""
        if self.config.action_type == "discrete":
            # Discrete actions: adjust temperature by predefined amounts
            self.action_space = spaces.Discrete(self.config.n_discrete_actions)
            
        elif self.config.action_type == "continuous":
            # Continuous actions: temperature multiplier, sweep count
            self.action_space = spaces.Box(
                low=np.array([0.1, 0.1]),  # [temp_multiplier, sweep_fraction]
                high=np.array([2.0, 2.0]),
                dtype=np.float32
            )
            
        elif self.config.action_type == "hybrid":
            # Hybrid: discrete temperature strategy + continuous parameters
            self.action_space = spaces.Dict({
                "strategy": spaces.Discrete(5),  # Different annealing strategies
                "parameters": spaces.Box(
                    low=np.array([0.1, 0.1]),
                    high=np.array([2.0, 2.0]),
                    dtype=np.float32
                )
            })
        else:
            raise ValueError(f"Unknown action_type: {self.config.action_type}")
    
    def _setup_observation_space(self):
        """Setup the observation space based on configuration."""
        if self.config.observation_type == "full":
            # Full spin configuration + global features
            spin_dim = self.config.n_spins
            global_features = 10  # energy, temp, step, etc.
            obs_dim = spin_dim + global_features
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            
        elif self.config.observation_type == "local":
            # Local neighborhoods around each spin
            local_size = self.config.local_neighborhood_size
            global_features = 10
            obs_dim = local_size + global_features
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            
        elif self.config.observation_type == "global_features":
            # Only global statistical features
            feature_dim = 20  # energy stats, temperature, acceptance rate, etc.
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(feature_dim,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown observation_type: {self.config.observation_type}")
    
    def _setup_annealer(self):
        """Setup the GPU annealer."""
        annealer_config = GPUAnnealerConfig(
            n_sweeps=self.config.annealer_sweeps,
            initial_temp=self.config.temperature_range[1],
            final_temp=self.config.temperature_range[0],
            schedule_type=ScheduleType.LINEAR,  # Will be controlled by RL agent
            record_interval=1
        )
        
        self.annealer = GPUAnnealer(annealer_config)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Generate new problem instance
        self.current_problem = self._generate_problem()
        
        # Reset environment state
        self.current_step = 0
        self.current_energy = self.current_problem.compute_energy()
        self.best_energy = self.current_energy
        self.current_temperature = self.config.temperature_range[1]
        
        # Clear histories
        self.energy_history = [self.current_energy]
        self.temperature_history = [self.current_temperature]
        self.acceptance_history = [0.0]
        self.episode_rewards = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one step in the environment."""
        if self.current_problem is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Process action to get annealing parameters
        temperature, n_sweeps = self._process_action(action)
        
        # Perform annealing steps
        prev_energy = self.current_energy
        new_energy, acceptance_rate = self._perform_annealing_step(temperature, n_sweeps)
        
        # Update state
        self.current_energy = new_energy
        self.current_temperature = temperature
        self.current_step += 1
        
        # Update histories
        self.energy_history.append(new_energy)
        self.temperature_history.append(temperature)
        self.acceptance_history.append(acceptance_rate)
        
        # Update best energy
        if new_energy < self.best_energy:
            self.best_energy = new_energy
        
        # Calculate reward
        reward = self._calculate_reward(prev_energy, new_energy, acceptance_rate)
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.config.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_problem(self) -> IsingModel:
        """Generate a new problem instance."""
        if self.config.problem_generator == "random_ising":
            return self._generate_random_ising()
        elif self.config.problem_generator == "tsp":
            return self._generate_tsp_problem()
        else:
            # Default: random Ising model
            return self._generate_random_ising()
    
    def _generate_random_ising(self) -> IsingModel:
        """Generate random Ising model."""
        n_spins = np.random.randint(*self.config.problem_size_range)
        
        # Create Ising model
        model = IsingModel(n_spins, device=self.device)
        
        # Add random couplings (sparse)
        n_couplings = min(n_spins * 2, n_spins * (n_spins - 1) // 4)
        for _ in range(n_couplings):
            i, j = np.random.choice(n_spins, 2, replace=False)
            strength = np.random.uniform(-2.0, 2.0)
            model.add_coupling(i, j, strength)
        
        # Add random external fields
        for i in range(n_spins):
            if np.random.random() < 0.3:  # 30% of spins have external field
                field = np.random.uniform(-1.0, 1.0)
                model.add_field(i, field)
        
        return model
    
    def _generate_tsp_problem(self) -> IsingModel:
        """Generate TSP problem as Ising model."""
        # This would require importing TSP problem class
        # For now, return random Ising model
        warnings.warn("TSP problem generation not implemented, using random Ising")
        return self._generate_random_ising()
    
    def _process_action(self, action) -> Tuple[float, int]:
        """Process RL action to get annealing parameters."""
        if self.config.action_type == "discrete":
            # Map discrete action to temperature adjustment
            action_value = int(action)
            temp_multiplier = 0.1 + (action_value / (self.config.n_discrete_actions - 1)) * 1.8
            new_temperature = self.current_temperature * temp_multiplier
            n_sweeps = self.config.annealer_sweeps
            
        elif self.config.action_type == "continuous":
            # Direct continuous control
            temp_multiplier, sweep_fraction = action
            new_temperature = self.current_temperature * temp_multiplier
            n_sweeps = max(1, int(self.config.annealer_sweeps * sweep_fraction))
            
        elif self.config.action_type == "hybrid":
            # Hybrid action processing
            strategy = action["strategy"]
            params = action["parameters"]
            
            # Different strategies for temperature control
            if strategy == 0:  # Exponential decay
                new_temperature = self.current_temperature * params[0]
            elif strategy == 1:  # Linear decay
                new_temperature = max(
                    self.config.temperature_range[0],
                    self.current_temperature - params[0]
                )
            else:  # Other strategies...
                new_temperature = self.current_temperature * params[0]
            
            n_sweeps = max(1, int(self.config.annealer_sweeps * params[1]))
        
        # Clamp temperature to valid range
        new_temperature = np.clip(
            new_temperature,
            self.config.temperature_range[0],
            self.config.temperature_range[1]
        )
        
        return new_temperature, n_sweeps
    
    def _perform_annealing_step(self, temperature: float, n_sweeps: int) -> Tuple[float, float]:
        """Perform annealing steps and return new energy and acceptance rate."""
        if self.annealer is None or self.current_problem is None:
            return self.current_energy, 0.0
        
        # Update annealer configuration
        self.annealer.config.n_sweeps = n_sweeps
        self.annealer.config.initial_temp = temperature
        self.annealer.config.final_temp = temperature  # Constant temperature
        
        # Run annealing
        result = self.annealer.anneal(self.current_problem)
        
        # Update problem with best configuration
        self.current_problem.set_spins(result.best_configuration)
        
        # Return energy and acceptance rate
        acceptance_rate = result.final_acceptance_rate if result.acceptance_rate_history else 0.0
        return result.best_energy, acceptance_rate
    
    def _calculate_reward(self, prev_energy: float, new_energy: float, acceptance_rate: float) -> float:
        """Calculate reward based on energy improvement and other factors."""
        if self.config.reward_type == "energy_delta":
            # Reward based on energy improvement
            energy_delta = prev_energy - new_energy
            reward = energy_delta * self.config.energy_scale_factor
            
            # Add exploration bonus
            if acceptance_rate > 0:
                reward += self.config.exploration_bonus * acceptance_rate
        
        elif self.config.reward_type == "acceptance_rate":
            # Reward based on maintaining good acceptance rate
            target_acceptance = 0.44  # Optimal acceptance rate
            reward = -abs(acceptance_rate - target_acceptance)
            
            # Bonus for energy improvement
            if new_energy < prev_energy:
                reward += (prev_energy - new_energy) * 0.1
        
        elif self.config.reward_type == "mixed":
            # Mixed reward combining energy and acceptance rate
            energy_reward = (prev_energy - new_energy) * self.config.energy_scale_factor
            acceptance_reward = min(acceptance_rate, 0.7)  # Cap acceptance reward
            reward = energy_reward + acceptance_reward * 0.1
        
        else:
            # Default: energy delta
            reward = (prev_energy - new_energy) * self.config.energy_scale_factor
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_problem is None:
            # Return zero observation if no problem
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if self.config.observation_type == "full":
            # Full spin configuration + global features
            spins = self.current_problem.get_spins().cpu().numpy()
            global_features = self._get_global_features()
            observation = np.concatenate([spins, global_features]).astype(np.float32)
            
        elif self.config.observation_type == "local":
            # Local neighborhood features (simplified)
            # For now, return global features
            observation = self._get_global_features()
            
        elif self.config.observation_type == "global_features":
            # Only global statistical features
            observation = self._get_global_features()
        
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Ensure observation matches expected shape
        if observation.shape[0] != self.observation_space.shape[0]:
            observation = np.resize(observation, self.observation_space.shape)
        
        return observation.astype(np.float32)
    
    def _get_global_features(self) -> np.ndarray:
        """Get global features for observation."""
        features = []
        
        # Energy features
        features.append(self.current_energy / 100.0)  # Normalized current energy
        features.append(self.best_energy / 100.0)     # Normalized best energy
        if len(self.energy_history) > 1:
            energy_trend = self.energy_history[-1] - self.energy_history[-2]
            features.append(energy_trend / 10.0)      # Energy change trend
        else:
            features.append(0.0)
        
        # Temperature features
        features.append(self.current_temperature / 10.0)  # Normalized temperature
        temp_range = self.config.temperature_range
        temp_progress = ((self.current_temperature - temp_range[0]) / 
                        (temp_range[1] - temp_range[0]))
        features.append(temp_progress)  # Temperature progress
        
        # Step features
        features.append(self.current_step / self.config.max_steps)  # Progress
        
        # Acceptance rate features
        if self.acceptance_history:
            features.append(self.acceptance_history[-1])  # Current acceptance rate
            features.append(np.mean(self.acceptance_history))  # Average acceptance
        else:
            features.extend([0.0, 0.0])
        
        # Energy statistics
        if len(self.energy_history) > 1:
            features.append(np.std(self.energy_history[-10:]) / 10.0)  # Recent energy std
        else:
            features.append(0.0)
        
        # Pad to minimum size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)  # Limit to reasonable size
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if energy hasn't improved for many steps
        if len(self.energy_history) > 100:
            recent_energies = self.energy_history[-50:]
            if np.std(recent_energies) < 1e-6:  # Very small changes
                return True
        
        # Terminate if temperature is very low and no improvement
        if (self.current_temperature < self.config.temperature_range[0] * 1.1 and
            len(self.energy_history) > 20 and
            self.energy_history[-1] >= self.energy_history[-20]):
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        info = {
            "current_energy": self.current_energy,
            "best_energy": self.best_energy,
            "current_temperature": self.current_temperature,
            "step": self.current_step,
            "n_spins": self.current_problem.n_spins if self.current_problem else 0
        }
        
        if self.acceptance_history:
            info["acceptance_rate"] = self.acceptance_history[-1]
        
        if len(self.episode_rewards) > 0:
            info["episode_reward"] = sum(self.episode_rewards)
            info["average_reward"] = np.mean(self.episode_rewards)
        
        return info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            # Print current state
            print(f"Step {self.current_step}: Energy={self.current_energy:.4f}, "
                  f"Best={self.best_energy:.4f}, Temp={self.current_temperature:.4f}")
            return None
        
        elif mode == "rgb_array":
            # Return RGB array representation (would need matplotlib)
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Energy history plot
                ax1.plot(self.energy_history)
                ax1.set_title("Energy History")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Energy")
                
                # Temperature history plot
                ax2.plot(self.temperature_history)
                ax2.set_title("Temperature History")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Temperature")
                
                # Convert to RGB array
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                
                return buf
            
            except ImportError:
                print("Matplotlib not available for rendering")
                return None
        
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'annealer') and self.annealer is not None:
            # Clear GPU memory if using CUDA
            if hasattr(self.annealer, 'memory_optimizer') and self.annealer.memory_optimizer:
                self.annealer.memory_optimizer.clear_memory_cache()
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get information about the current problem."""
        if self.current_problem is None:
            return {}
        
        return {
            "n_spins": self.current_problem.n_spins,
            "n_couplings": torch.count_nonzero(self.current_problem.couplings).item(),
            "coupling_strength_mean": torch.mean(torch.abs(self.current_problem.couplings)).item(),
            "field_strength_mean": torch.mean(torch.abs(self.current_problem.external_fields)).item(),
        }


def make_spin_glass_env(config: Optional[SpinGlassEnvConfig] = None) -> SpinGlassEnv:
    """Factory function to create SpinGlassEnv."""
    if config is None:
        config = SpinGlassEnvConfig()
    return SpinGlassEnv(config)