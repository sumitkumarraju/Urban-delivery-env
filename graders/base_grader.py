"""Base Grader — Abstract scoring interface for all task graders.

All graders must:
- Be deterministic (same inputs → same score)
- Return a float between 0.0 and 1.0
- Be reproducible across runs
"""

from abc import ABC, abstractmethod

from env import DeliveryEnvironment, TaskConfig
from models.action import DeliveryAction
from models.observation import DeliveryObservation


class BaseGrader(ABC):
    """Abstract base class for deterministic environment graders."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def run_episode(self, actions: list[int]) -> dict:
        """Run a complete episode with the given action sequence.

        Args:
            actions: List of action integers (0-5).

        Returns:
            Dictionary with episode statistics.
        """
        env = DeliveryEnvironment(self.config)
        obs = env.reset()

        total_reward = 0.0
        steps = 0
        fuel_used = self.config.initial_fuel

        for action_int in actions:
            if obs.done:
                break
            action = DeliveryAction(action=action_int)
            obs, reward_info = env.step(action)
            total_reward = reward_info.cumulative_reward
            steps += 1

        fuel_used = self.config.initial_fuel - obs.vehicle.fuel

        return {
            "packages_delivered": obs.packages_delivered,
            "packages_total": obs.packages_total,
            "total_reward": total_reward,
            "steps": steps,
            "max_steps": self.config.max_steps,
            "fuel_used": fuel_used,
            "initial_fuel": self.config.initial_fuel,
            "done": obs.done,
            "final_observation": obs,
        }

    @abstractmethod
    def score(self, episode_stats: dict) -> float:
        """Calculate a score between 0.0 and 1.0.

        Args:
            episode_stats: Dict from run_episode().

        Returns:
            Float score in [0.0, 1.0].
        """
        ...

    def grade(self, actions: list[int]) -> float:
        """Run episode and return grade. Convenience method.

        Args:
            actions: List of action integers.

        Returns:
            Float score in [0.0, 1.0].
        """
        stats = self.run_episode(actions)
        raw_score = self.score(stats)
        return max(0.0, min(1.0, raw_score))
