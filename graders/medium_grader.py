"""Medium Grader — Multi-factor scoring.

Score = 0.5 * completion + 0.3 * fuel_efficiency + 0.2 * time_efficiency

Balances delivery completion with resource management and speed.
"""

from graders.base_grader import BaseGrader
from tasks.medium import MEDIUM_TASK


class MediumGrader(BaseGrader):
    """Grades the medium task with weighted multi-factor scoring."""

    def __init__(self):
        super().__init__(MEDIUM_TASK)

    def score(self, episode_stats: dict) -> float:
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]

        completion = delivered / total if total > 0 else 0.0
        fuel_used = episode_stats["fuel_used"]
        initial_fuel = episode_stats["initial_fuel"]
        fuel_efficiency = max(0.0, 1.0 - (fuel_used / initial_fuel)) if initial_fuel > 0 else 0.0
        steps = episode_stats["steps"]
        max_steps = episode_stats["max_steps"]
        time_efficiency = max(0.0, 1.0 - (steps / max_steps)) if max_steps > 0 else 0.0

        return 0.5 * completion + 0.3 * fuel_efficiency + 0.2 * time_efficiency

    def score_with_explanation(self, episode_stats: dict) -> tuple[float, dict]:
        """Return score with explanation breakdown."""
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]

        completion = delivered / total if total > 0 else 0.0
        fuel_used = episode_stats["fuel_used"]
        initial_fuel = episode_stats["initial_fuel"]
        fuel_efficiency = max(0.0, 1.0 - (fuel_used / initial_fuel)) if initial_fuel > 0 else 0.0
        steps = episode_stats["steps"]
        max_steps = episode_stats["max_steps"]
        time_efficiency = max(0.0, 1.0 - (steps / max_steps)) if max_steps > 0 else 0.0

        final_score = 0.5 * completion + 0.3 * fuel_efficiency + 0.2 * time_efficiency

        explanation = {
            "completion": {"raw": completion, "weight": 0.5, "weighted": 0.5 * completion},
            "fuel_efficiency": {"raw": fuel_efficiency, "weight": 0.3, "weighted": 0.3 * fuel_efficiency,
                                "fuel_used": fuel_used, "initial_fuel": initial_fuel},
            "time_efficiency": {"raw": time_efficiency, "weight": 0.2, "weighted": 0.2 * time_efficiency,
                                "steps": steps, "max_steps": max_steps},
            "final_score": final_score,
        }
        return final_score, explanation
