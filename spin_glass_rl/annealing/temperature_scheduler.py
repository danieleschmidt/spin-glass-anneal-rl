"""Temperature scheduling algorithms for simulated annealing."""

from typing import Optional, Callable, List
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ScheduleType(Enum):
    """Available temperature schedules."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"  
    GEOMETRIC = "geometric"
    LOGARITHMIC = "logarithmic"
    POWER_LAW = "power_law"
    ADAPTIVE = "adaptive"
    FAST = "fast"
    BOLTZMANN = "boltzmann"
    CUSTOM = "custom"


@dataclass
class ScheduleConfig:
    """Configuration for temperature schedules."""
    schedule_type: ScheduleType
    initial_temp: float
    final_temp: float
    total_sweeps: int
    
    # Schedule-specific parameters
    alpha: float = 0.95  # Geometric cooling factor
    k: float = 1.0       # Power law exponent
    c: float = 1.0       # Logarithmic/Boltzmann constant
    
    # Adaptive parameters
    target_acceptance: float = 0.44  # Target acceptance rate
    adaptation_window: int = 100     # Window for adaptation
    adaptation_rate: float = 0.1     # Rate of temperature adjustment


class TemperatureSchedule(ABC):
    """Abstract base class for temperature schedules."""
    
    def __init__(self, config: ScheduleConfig):
        self.config = config
        self.current_sweep = 0
        self.current_temp = config.initial_temp
        self.temperature_history: List[float] = [config.initial_temp]
    
    @abstractmethod
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        pass
    
    @abstractmethod
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature and return new value."""
        pass
    
    def reset(self) -> None:
        """Reset schedule to initial state."""
        self.current_sweep = 0
        self.current_temp = self.config.initial_temp
        self.temperature_history = [self.config.initial_temp]


class LinearSchedule(TemperatureSchedule):
    """Linear temperature decrease: T(t) = T_0 - (T_0 - T_f) * t/T_max."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        if sweep >= self.config.total_sweeps:
            return self.config.final_temp
        
        progress = sweep / self.config.total_sweeps
        temp = self.config.initial_temp - (
            self.config.initial_temp - self.config.final_temp
        ) * progress
        
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class ExponentialSchedule(TemperatureSchedule):
    """Exponential schedule: T(t) = T_0 * exp(-λt)."""
    
    def __init__(self, config: ScheduleConfig):
        super().__init__(config)
        # Calculate lambda to reach final temp at total sweeps
        if config.final_temp > 0:
            self.lambda_param = -np.log(config.final_temp / config.initial_temp) / config.total_sweeps
        else:
            self.lambda_param = 0.01  # Default decay rate
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        temp = self.config.initial_temp * np.exp(-self.lambda_param * sweep)
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class GeometricSchedule(TemperatureSchedule):
    """Geometric schedule: T(t+1) = α * T(t)."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        temp = self.config.initial_temp * (self.config.alpha ** sweep)
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class LogarithmicSchedule(TemperatureSchedule):
    """Logarithmic schedule: T(t) = c / log(1 + t)."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        if sweep == 0:
            return self.config.initial_temp
        
        temp = self.config.c / np.log(1 + sweep)
        temp = temp * self.config.initial_temp / self.config.c  # Scale to initial temp
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class PowerLawSchedule(TemperatureSchedule):
    """Power law schedule: T(t) = T_0 / (1 + t)^k."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        temp = self.config.initial_temp / ((1 + sweep) ** self.config.k)
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class FastSchedule(TemperatureSchedule):
    """Fast annealing schedule: T(t) = T_0 / t."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        if sweep == 0:
            return self.config.initial_temp
        
        temp = self.config.initial_temp / sweep
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class BoltzmannSchedule(TemperatureSchedule):
    """Boltzmann schedule: T(t) = T_0 / log(1 + t)."""
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        if sweep == 0:
            return self.config.initial_temp
        
        temp = self.config.initial_temp / np.log(1 + sweep)
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class AdaptiveSchedule(TemperatureSchedule):
    """Adaptive schedule that adjusts based on acceptance rate."""
    
    def __init__(self, config: ScheduleConfig):
        super().__init__(config)
        self.acceptance_history: List[float] = []
        self.base_schedule = GeometricSchedule(config)  # Use geometric as base
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature at given sweep."""
        return self.current_temp
    
    def update(self, sweep: int, acceptance_rate: Optional[float] = None, **kwargs) -> float:
        """Update temperature based on acceptance rate."""
        self.current_sweep = sweep
        
        if acceptance_rate is not None:
            self.acceptance_history.append(acceptance_rate)
        
        # Base temperature from geometric schedule
        base_temp = self.base_schedule.get_temperature(sweep)
        
        # Adapt based on recent acceptance rate
        if len(self.acceptance_history) >= self.config.adaptation_window:
            recent_acceptance = np.mean(
                self.acceptance_history[-self.config.adaptation_window:]
            )
            
            # Adjust temperature based on acceptance rate
            if recent_acceptance > self.config.target_acceptance:
                # Acceptance too high, decrease temperature faster
                adjustment = 1.0 - self.config.adaptation_rate
            elif recent_acceptance < self.config.target_acceptance:
                # Acceptance too low, decrease temperature slower
                adjustment = 1.0 + self.config.adaptation_rate
            else:
                adjustment = 1.0
            
            self.current_temp = max(base_temp * adjustment, self.config.final_temp)
        else:
            self.current_temp = base_temp
        
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class CustomSchedule(TemperatureSchedule):
    """Custom temperature schedule with user-defined function."""
    
    def __init__(self, config: ScheduleConfig, schedule_func: Callable[[int], float]):
        super().__init__(config)
        self.schedule_func = schedule_func
    
    def get_temperature(self, sweep: int) -> float:
        """Get temperature using custom function."""
        temp = self.schedule_func(sweep)
        return max(temp, self.config.final_temp)
    
    def update(self, sweep: int, **kwargs) -> float:
        """Update temperature."""
        self.current_sweep = sweep  
        self.current_temp = self.get_temperature(sweep)
        self.temperature_history.append(self.current_temp)
        return self.current_temp


class TemperatureScheduler:
    """
    Factory and manager for temperature schedules.
    
    Provides easy interface for creating and managing different
    temperature scheduling algorithms.
    """
    
    _schedule_classes = {
        ScheduleType.LINEAR: LinearSchedule,
        ScheduleType.EXPONENTIAL: ExponentialSchedule,
        ScheduleType.GEOMETRIC: GeometricSchedule,
        ScheduleType.LOGARITHMIC: LogarithmicSchedule,
        ScheduleType.POWER_LAW: PowerLawSchedule,
        ScheduleType.ADAPTIVE: AdaptiveSchedule,
        ScheduleType.FAST: FastSchedule,
        ScheduleType.BOLTZMANN: BoltzmannSchedule,
        ScheduleType.CUSTOM: CustomSchedule,
    }
    
    @classmethod
    def create_schedule(
        self,
        schedule_type: ScheduleType,
        initial_temp: float,
        final_temp: float,
        total_sweeps: int,
        custom_func: Optional[Callable[[int], float]] = None,
        **kwargs
    ) -> TemperatureSchedule:
        """
        Create temperature schedule.
        
        Args:
            schedule_type: Type of schedule
            initial_temp: Starting temperature
            final_temp: Final temperature
            total_sweeps: Total number of sweeps
            custom_func: Custom function for CUSTOM schedule type
            **kwargs: Additional schedule parameters
            
        Returns:
            TemperatureSchedule instance
        """
        config = ScheduleConfig(
            schedule_type=schedule_type,
            initial_temp=initial_temp, 
            final_temp=final_temp,
            total_sweeps=total_sweeps,
            **kwargs
        )
        
        schedule_class = self._schedule_classes[schedule_type]
        
        if schedule_type == ScheduleType.CUSTOM:
            if custom_func is None:
                raise ValueError("custom_func required for CUSTOM schedule type")
            return schedule_class(config, custom_func)
        else:
            return schedule_class(config)
    
    @classmethod
    def get_available_schedules(cls) -> List[str]:
        """Get list of available schedule types."""
        return [schedule.value for schedule in ScheduleType]
    
    @classmethod
    def compare_schedules(
        cls,
        initial_temp: float,
        final_temp: float, 
        total_sweeps: int,
        schedule_types: Optional[List[ScheduleType]] = None
    ) -> dict:
        """
        Compare temperature trajectories of different schedules.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            total_sweeps: Total sweeps
            schedule_types: List of schedules to compare
            
        Returns:
            Dictionary with temperature trajectories
        """
        if schedule_types is None:
            schedule_types = [
                ScheduleType.LINEAR,
                ScheduleType.EXPONENTIAL,
                ScheduleType.GEOMETRIC,
                ScheduleType.LOGARITHMIC,
            ]
        
        results = {}
        
        for schedule_type in schedule_types:
            if schedule_type == ScheduleType.CUSTOM:
                continue  # Skip custom without function
            
            schedule = cls.create_schedule(
                schedule_type, initial_temp, final_temp, total_sweeps
            )
            
            temperatures = []
            for sweep in range(0, total_sweeps, max(1, total_sweeps // 100)):
                temperatures.append(schedule.get_temperature(sweep))
            
            results[schedule_type.value] = temperatures
        
        return results
    
    @classmethod
    def plot_schedules(
        cls,
        initial_temp: float,
        final_temp: float,
        total_sweeps: int,
        schedule_types: Optional[List[ScheduleType]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot temperature schedules for comparison."""
        try:
            import matplotlib.pyplot as plt
            
            results = cls.compare_schedules(
                initial_temp, final_temp, total_sweeps, schedule_types
            )
            
            plt.figure(figsize=(10, 6))
            
            x_points = np.linspace(0, total_sweeps, len(list(results.values())[0]))
            
            for schedule_name, temperatures in results.items():
                plt.plot(x_points, temperatures, label=schedule_name, linewidth=2)
            
            plt.xlabel('Sweep')
            plt.ylabel('Temperature')
            plt.title('Temperature Schedule Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    @classmethod
    def recommend_schedule(
        cls,
        problem_size: int,
        time_budget: int,
        convergence_preference: str = "balanced"  # "fast", "quality", "balanced"
    ) -> tuple[ScheduleType, dict]:
        """
        Recommend temperature schedule based on problem characteristics.
        
        Args:
            problem_size: Number of variables/spins
            time_budget: Available computation time (sweeps)
            convergence_preference: Optimization preference
            
        Returns:
            (schedule_type, parameters) tuple
        """
        # Simple heuristics for schedule recommendation
        if convergence_preference == "fast":
            if problem_size < 1000:
                return ScheduleType.FAST, {"k": 1.0}
            else:
                return ScheduleType.EXPONENTIAL, {"alpha": 0.99}
        
        elif convergence_preference == "quality":
            if time_budget > 10000:
                return ScheduleType.LOGARITHMIC, {"c": 10.0}
            else:
                return ScheduleType.GEOMETRIC, {"alpha": 0.95}
        
        else:  # balanced
            if problem_size < 1000:
                return ScheduleType.GEOMETRIC, {"alpha": 0.95}
            else:
                return ScheduleType.ADAPTIVE, {
                    "alpha": 0.95,
                    "target_acceptance": 0.44,
                    "adaptation_window": 100
                }
    
    def __repr__(self) -> str:
        """String representation."""
        available = ", ".join(self.get_available_schedules())
        return f"TemperatureScheduler(available_schedules=[{available}])"