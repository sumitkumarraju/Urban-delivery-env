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

        # Completion ratio (50%)
        completion = delivered / total if total > 0 else 0.0

        # Fuel efficiency (30%) — ratio of fuel remaining
        fuel_used = episode_stats["fuel_used"]
        initial_fuel = episode_stats["initial_fuel"]
        fuel_efficiency = max(0.0, 1.0 - (fuel_used / initial_fuel)) if initial_fuel > 0 else 0.0

        # Time efficiency (20%) — ratio of steps used vs max
        steps = episode_stats["steps"]
        max_steps = episode_stats["max_steps"]
        time_efficiency = max(0.0, 1.0 - (steps / max_steps)) if max_steps > 0 else 0.0

        score = 0.5 * completion + 0.3 * fuel_efficiency + 0.2 * time_efficiency
        return score


if __name__ == "__main__":
    grader = MediumGrader()
    import random
    random.seed(123)
    actions = [random.randint(0, 5) for _ in range(200)]
    score = grader.grade(actions)
    print(f"Medium Task Score: {score:.4f}")
