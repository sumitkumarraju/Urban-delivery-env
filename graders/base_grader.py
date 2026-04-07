"""Base Grader — Abstract scoring interface for all task graders.

All graders must:
- Be deterministic (same inputs → same score)
- Return a float strictly between 0 and 1 (OpenEnv validators reject 0.0 and 1.0)
- Be reproducible across runs
"""

from abc import ABC, abstractmethod

from env import DeliveryEnvironment, TaskConfig
from models.action import DeliveryAction
from models.observation import DeliveryObservation

# Linear map [0, 1] -> (eps, 1-eps) so platform checks 0 < score < 1 pass.
# Epsilon must be large enough that :.4f formatting never rounds to "0.0000"/"1.0000".
_GRADE_EPS = 0.001


def _clamp_unit(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to_open_unit_interval(clamped: float) -> float:
    """Map a [0, 1] value into a strict subset of (0, 1)."""
    return _GRADE_EPS + (1.0 - 2.0 * _GRADE_EPS) * clamped


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
        """Calculate a raw score on the conceptual [0.0, 1.0] scale.

        Args:
            episode_stats: Dict from run_episode().

        Returns:
            Float score in [0.0, 1.0] before open-interval normalization in grade().
        """
        ...

    def grade(self, actions: list[int]) -> float:
        """Run episode and return grade. Convenience method."""
        stats = self.run_episode(actions)
        raw_score = self.score(stats)
        return _to_open_unit_interval(_clamp_unit(raw_score))

    def grade_with_explanation(self, actions: list[int]) -> tuple[float, dict]:
        """Run episode and return grade with full scoring breakdown.

        Returns:
            Tuple of (score, explanation_dict). Score is strictly in (0, 1).
        """
        stats = self.run_episode(actions)
        if hasattr(self, 'score_with_explanation'):
            score, explanation = self.score_with_explanation(stats)
            mapped = _to_open_unit_interval(_clamp_unit(score))
            explanation["final_score"] = mapped
            return mapped, explanation
        raw_score = self.score(stats)
        mapped = _to_open_unit_interval(_clamp_unit(raw_score))
        return mapped, {"final_score": mapped}
