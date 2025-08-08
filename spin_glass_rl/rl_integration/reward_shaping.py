"""Reward shaping utilities for RL-guided spin-glass optimization."""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
import math

from spin_glass_rl.core.ising_model import IsingModel


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    
    # Base reward components
    energy_reward_weight: float = 1.0
    acceptance_rate_weight: float = 0.1
    temperature_penalty_weight: float = 0.05
    
    # Advanced reward components
    exploration_bonus_weight: float = 0.02
    convergence_bonus_weight: float = 0.5
    efficiency_bonus_weight: float = 0.1
    
    # Reward shaping parameters
    energy_normalization: str = "adaptive"  # "fixed", "adaptive", "relative"
    reward_clipping: Optional[Tuple[float, float]] = (-10.0, 10.0)
    reward_smoothing: float = 0.1  # Exponential moving average factor
    
    # Performance thresholds
    good_acceptance_rate: float = 0.44  # Optimal for Metropolis
    acceptance_tolerance: float = 0.1
    convergence_threshold: float = 1e-6
    exploration_decay: float = 0.995
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 3
    difficulty_ramp: float = 1.2


class RewardComponent(ABC):
    """Abstract base class for reward components."""
    
    @abstractmethod
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        """Compute reward component value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset component state."""
        pass


class EnergyReward(RewardComponent):
    """Reward based on energy improvement."""
    
    def __init__(self, weight: float = 1.0, normalization: str = "adaptive"):
        self.weight = weight
        self.normalization = normalization
        self.energy_history = deque(maxlen=100)
        self.energy_scale = 1.0
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        prev_energy = state_info.get("current_energy", 0.0)
        new_energy = next_state_info.get("current_energy", 0.0)
        
        # Raw energy improvement
        energy_delta = prev_energy - new_energy
        
        # Update history and scale
        self.energy_history.append(abs(energy_delta))
        if len(self.energy_history) > 10:
            if self.normalization == "adaptive":
                self.energy_scale = max(1e-6, np.std(self.energy_history))
            elif self.normalization == "relative":
                self.energy_scale = max(1e-6, np.mean(self.energy_history))
        
        # Normalized reward
        if self.energy_scale > 0:
            normalized_delta = energy_delta / self.energy_scale
        else:
            normalized_delta = energy_delta
        
        return self.weight * normalized_delta
    
    def reset(self):
        self.energy_history.clear()
        self.energy_scale = 1.0


class AcceptanceRateReward(RewardComponent):
    """Reward for maintaining good acceptance rate."""
    
    def __init__(self, weight: float = 0.1, target_rate: float = 0.44, tolerance: float = 0.1):
        self.weight = weight
        self.target_rate = target_rate
        self.tolerance = tolerance
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        acceptance_rate = next_state_info.get("acceptance_rate", 0.0)
        
        # Distance from target acceptance rate
        distance = abs(acceptance_rate - self.target_rate)
        
        if distance <= self.tolerance:
            # Good acceptance rate
            reward = 1.0 - (distance / self.tolerance)
        else:
            # Poor acceptance rate
            reward = -distance
        
        return self.weight * reward
    
    def reset(self):
        pass


class TemperatureReward(RewardComponent):
    """Reward/penalty for temperature control."""
    
    def __init__(self, weight: float = 0.05):
        self.weight = weight
        self.prev_temperature = None
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        current_temp = state_info.get("current_temperature", 1.0)
        new_temp = next_state_info.get("current_temperature", 1.0)
        
        # Penalize too rapid temperature changes
        if self.prev_temperature is not None:
            temp_change_rate = abs(new_temp - self.prev_temperature) / max(self.prev_temperature, 1e-6)
            rapid_change_penalty = -temp_change_rate if temp_change_rate > 0.5 else 0.0
        else:
            rapid_change_penalty = 0.0
        
        # Penalize very high or very low temperatures at wrong times
        step = next_state_info.get("step", 0)
        max_steps = next_state_info.get("max_steps", 1000)
        progress = step / max(max_steps, 1)
        
        # Expected temperature based on progress
        expected_temp = 10.0 * (1 - progress) + 0.01 * progress
        temp_alignment = -abs(new_temp - expected_temp) / expected_temp
        
        self.prev_temperature = new_temp
        
        return self.weight * (rapid_change_penalty + temp_alignment * 0.1)
    
    def reset(self):
        self.prev_temperature = None


class ExplorationReward(RewardComponent):
    """Reward for exploration and diverse sampling."""
    
    def __init__(self, weight: float = 0.02, decay: float = 0.995):
        self.weight = weight
        self.decay = decay
        self.visited_states = set()
        self.exploration_bonus_scale = 1.0
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        # Create state signature (simplified)
        energy = next_state_info.get("current_energy", 0.0)
        temp = next_state_info.get("current_temperature", 1.0)
        state_signature = (round(energy, 2), round(temp, 3), action)
        
        # Exploration bonus for new states
        if state_signature not in self.visited_states:
            self.visited_states.add(state_signature)
            exploration_bonus = self.exploration_bonus_scale
        else:
            exploration_bonus = 0.0
        
        # Decay exploration bonus over time
        self.exploration_bonus_scale *= self.decay
        
        return self.weight * exploration_bonus
    
    def reset(self):
        self.visited_states.clear()
        self.exploration_bonus_scale = 1.0


class ConvergenceReward(RewardComponent):
    """Reward for achieving convergence."""
    
    def __init__(self, weight: float = 0.5, threshold: float = 1e-6, window: int = 20):
        self.weight = weight
        self.threshold = threshold
        self.window = window
        self.energy_window = deque(maxlen=window)
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        current_energy = next_state_info.get("current_energy", 0.0)
        self.energy_window.append(current_energy)
        
        if len(self.energy_window) < self.window:
            return 0.0
        
        # Check for convergence
        energy_std = np.std(self.energy_window)
        if energy_std < self.threshold:
            # Converged - give bonus based on quality
            mean_energy = np.mean(self.energy_window)
            convergence_bonus = 1.0 / (1.0 + abs(mean_energy))
            return self.weight * convergence_bonus
        
        return 0.0
    
    def reset(self):
        self.energy_window.clear()


class EfficiencyReward(RewardComponent):
    """Reward for computational efficiency."""
    
    def __init__(self, weight: float = 0.1):
        self.weight = weight
        self.step_count = 0
        self.best_energy_step = 0
        self.best_energy = float('inf')
    
    def compute(self, state_info: Dict[str, Any], action: int, next_state_info: Dict[str, Any]) -> float:
        self.step_count += 1
        current_energy = next_state_info.get("current_energy", 0.0)
        
        # Track when best energy was found
        if current_energy < self.best_energy:
            self.best_energy = current_energy
            self.best_energy_step = self.step_count
        
        # Reward early convergence to good solution
        if self.step_count > 50:
            efficiency = (self.step_count - self.best_energy_step + 1) / self.step_count
            efficiency_bonus = (1.0 - efficiency) * 0.5  # Bonus for finding good solution early
        else:
            efficiency_bonus = 0.0
        
        return self.weight * efficiency_bonus
    
    def reset(self):
        self.step_count = 0
        self.best_energy_step = 0
        self.best_energy = float('inf')


class RewardShaper:
    """
    Comprehensive reward shaping for RL-guided spin-glass optimization.
    
    Combines multiple reward components with configurable weights and
    supports adaptive reward scaling and curriculum learning.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        
        # Initialize reward components
        self.components = {
            "energy": EnergyReward(config.energy_reward_weight, config.energy_normalization),
            "acceptance": AcceptanceRateReward(config.acceptance_rate_weight, 
                                             config.good_acceptance_rate, 
                                             config.acceptance_tolerance),
            "temperature": TemperatureReward(config.temperature_penalty_weight),
            "exploration": ExplorationReward(config.exploration_bonus_weight, 
                                           config.exploration_decay),
            "convergence": ConvergenceReward(config.convergence_bonus_weight, 
                                           config.convergence_threshold),
            "efficiency": EfficiencyReward(config.efficiency_bonus_weight)
        }
        
        # Reward smoothing
        self.smoothed_reward = 0.0
        
        # Curriculum learning
        self.current_stage = 0
        self.episode_count = 0
        
        # Performance tracking
        self.reward_history = deque(maxlen=1000)
        self.component_histories = {name: deque(maxlen=100) for name in self.components.keys()}
    
    def compute_reward(self, state_info: Dict[str, Any], action: int, 
                      next_state_info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward from all components.
        
        Returns:
            Total reward and individual component contributions
        """
        component_rewards = {}
        total_reward = 0.0
        
        # Compute reward from each component
        for name, component in self.components.items():
            try:
                reward = component.compute(state_info, action, next_state_info)
                component_rewards[name] = reward
                total_reward += reward
                
                # Track component history
                self.component_histories[name].append(reward)
                
            except Exception as e:
                # Handle component errors gracefully
                component_rewards[name] = 0.0
                print(f"Warning: Error in reward component {name}: {e}")
        
        # Apply curriculum learning scaling
        if self.config.use_curriculum:
            curriculum_scale = self._get_curriculum_scale()
            total_reward *= curriculum_scale
        
        # Apply reward smoothing
        self.smoothed_reward = (self.config.reward_smoothing * total_reward + 
                               (1 - self.config.reward_smoothing) * self.smoothed_reward)
        
        # Apply reward clipping
        if self.config.reward_clipping is not None:
            min_reward, max_reward = self.config.reward_clipping
            total_reward = np.clip(total_reward, min_reward, max_reward)
        
        # Track reward history
        self.reward_history.append(total_reward)
        
        return total_reward, component_rewards
    
    def _get_curriculum_scale(self) -> float:
        """Get curriculum learning scale factor."""
        if not self.config.use_curriculum:
            return 1.0
        
        episodes_per_stage = 1000 // self.config.curriculum_stages
        stage = min(self.episode_count // episodes_per_stage, self.config.curriculum_stages - 1)
        
        # Gradually increase difficulty/complexity
        scale = 1.0 + (stage * self.config.difficulty_ramp / self.config.curriculum_stages)
        return scale
    
    def reset(self):
        """Reset all reward components for new episode."""
        for component in self.components.values():
            component.reset()
        
        self.smoothed_reward = 0.0
        self.episode_count += 1
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get reward statistics."""
        stats = {
            "total_episodes": self.episode_count,
            "current_stage": self.current_stage,
            "smoothed_reward": self.smoothed_reward
        }
        
        # Overall reward statistics
        if self.reward_history:
            stats.update({
                "mean_reward": np.mean(self.reward_history),
                "std_reward": np.std(self.reward_history),
                "min_reward": np.min(self.reward_history),
                "max_reward": np.max(self.reward_history)
            })
        
        # Component-wise statistics
        component_stats = {}
        for name, history in self.component_histories.items():
            if history:
                component_stats[name] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "weight": getattr(self.config, f"{name}_reward_weight", 0.0)
                }
        
        stats["components"] = component_stats
        
        return stats
    
    def update_weights(self, component_weights: Dict[str, float]):
        """Update component weights dynamically."""
        for name, weight in component_weights.items():
            if name in self.components:
                if hasattr(self.components[name], 'weight'):
                    self.components[name].weight = weight
    
    def add_custom_component(self, name: str, component: RewardComponent):
        """Add custom reward component."""
        self.components[name] = component
        self.component_histories[name] = deque(maxlen=100)
    
    def analyze_reward_balance(self) -> Dict[str, Any]:
        """Analyze reward component balance and suggest adjustments."""
        analysis = {}
        
        if len(self.reward_history) < 50:
            return {"status": "insufficient_data"}
        
        # Check component contributions
        component_contributions = {}
        for name, history in self.component_histories.items():
            if history:
                mean_contribution = np.mean(history)
                std_contribution = np.std(history)
                component_contributions[name] = {
                    "mean": mean_contribution,
                    "std": std_contribution,
                    "relative_importance": abs(mean_contribution) / (np.sum([abs(np.mean(h)) for h in self.component_histories.values() if h]) + 1e-6)
                }
        
        # Identify dominant components
        sorted_components = sorted(component_contributions.items(), 
                                 key=lambda x: x[1]["relative_importance"], reverse=True)
        
        analysis["component_contributions"] = component_contributions
        analysis["dominant_component"] = sorted_components[0][0] if sorted_components else None
        
        # Check for reward signal issues
        recent_rewards = list(self.reward_history)[-50:]
        analysis["signal_quality"] = {
            "mean": np.mean(recent_rewards),
            "std": np.std(recent_rewards),
            "trend": "increasing" if recent_rewards[-10:] > recent_rewards[:10] else "decreasing",
            "stability": np.std(recent_rewards) < 0.1
        }
        
        # Suggest adjustments
        suggestions = []
        if component_contributions.get("energy", {}).get("relative_importance", 0) > 0.8:
            suggestions.append("Consider increasing other component weights to balance energy reward")
        
        if analysis["signal_quality"]["std"] < 0.01:
            suggestions.append("Reward signal may be too stable - consider increasing exploration")
        
        analysis["suggestions"] = suggestions
        
        return analysis


def create_reward_shaper(config: Optional[RewardConfig] = None) -> RewardShaper:
    """Factory function to create reward shaper."""
    if config is None:
        config = RewardConfig()
    return RewardShaper(config)