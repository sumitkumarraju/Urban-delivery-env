"""Hard Grader — Full weighted scoring with all factors.

Score = 0.4 * completion + 0.2 * fuel_efficiency + 0.2 * deadline_compliance
        + 0.1 * priority_accuracy + 0.1 * reward_normalized

This grader evaluates mastery across all environment dimensions.
"""

from graders.base_grader import BaseGrader
from tasks.hard import HARD_TASK


class HardGrader(BaseGrader):
    """Grades the hard task with comprehensive weighted scoring."""

    def __init__(self):
        super().__init__(HARD_TASK)

    def score(self, episode_stats: dict) -> float:
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]
        obs = episode_stats["final_observation"]

        # Completion ratio (40%)
        completion = delivered / total if total > 0 else 0.0

        # Fuel efficiency (20%) — clamped to [0, 1] since refueling can make fuel_used negative
        fuel_used = episode_stats["fuel_used"]
        initial_fuel = episode_stats["initial_fuel"]
        fuel_efficiency = max(0.0, min(1.0, 1.0 - (fuel_used / initial_fuel))) if initial_fuel > 0 else 0.0

        # Deadline compliance (20%) — uses per-package delivery_step
        deadline_met = 0
        deadline_total = 0
        for pkg in obs.packages:
            if pkg.delivered and pkg.deadline is not None:
                deadline_total += 1
                if pkg.delivery_step is not None and pkg.delivery_step <= pkg.deadline:
                    deadline_met += 1
        deadline_compliance = (deadline_met / deadline_total) if deadline_total > 0 else 1.0

        # Priority accuracy (10%)
        priority_delivered = 0
        priority_total = 0
        for pkg in obs.packages:
            if pkg.priority > 0:
                priority_total += 1
                if pkg.delivered:
                    priority_delivered += 1
        priority_accuracy = (priority_delivered / priority_total) if priority_total > 0 else 1.0

        # Reward normalized (10%)
        total_reward = episode_stats["total_reward"]
        max_possible = total * 20 + total * 15 + total * 10 + 50
        reward_normalized = max(0.0, min(1.0, total_reward / max_possible)) if max_possible > 0 else 0.0

        score = (
            0.4 * completion
            + 0.2 * fuel_efficiency
            + 0.2 * deadline_compliance
            + 0.1 * priority_accuracy
            + 0.1 * reward_normalized
        )
        return score

    def score_with_explanation(self, episode_stats: dict) -> tuple[float, dict]:
        """Return score along with a breakdown of each factor."""
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]
        obs = episode_stats["final_observation"]

        completion = delivered / total if total > 0 else 0.0
        fuel_used = episode_stats["fuel_used"]
        initial_fuel = episode_stats["initial_fuel"]
        fuel_efficiency = max(0.0, min(1.0, 1.0 - (fuel_used / initial_fuel))) if initial_fuel > 0 else 0.0

        deadline_met = 0
        deadline_total = 0
        for pkg in obs.packages:
            if pkg.delivered and pkg.deadline is not None:
                deadline_total += 1
                if pkg.delivery_step is not None and pkg.delivery_step <= pkg.deadline:
                    deadline_met += 1
        deadline_compliance = (deadline_met / deadline_total) if deadline_total > 0 else 1.0

        priority_delivered = 0
        priority_total = 0
        for pkg in obs.packages:
            if pkg.priority > 0:
                priority_total += 1
                if pkg.delivered:
                    priority_delivered += 1
        priority_accuracy = (priority_delivered / priority_total) if priority_total > 0 else 1.0

        total_reward = episode_stats["total_reward"]
        max_possible = total * 20 + total * 15 + total * 10 + 50
        reward_normalized = max(0.0, min(1.0, total_reward / max_possible)) if max_possible > 0 else 0.0

        final_score = (
            0.4 * completion
            + 0.2 * fuel_efficiency
            + 0.2 * deadline_compliance
            + 0.1 * priority_accuracy
            + 0.1 * reward_normalized
        )

        explanation = {
            "completion": {"raw": completion, "weight": 0.4, "weighted": 0.4 * completion},
            "fuel_efficiency": {"raw": fuel_efficiency, "weight": 0.2, "weighted": 0.2 * fuel_efficiency},
            "deadline_compliance": {"raw": deadline_compliance, "weight": 0.2, "weighted": 0.2 * deadline_compliance,
                                     "met": deadline_met, "total": deadline_total},
            "priority_accuracy": {"raw": priority_accuracy, "weight": 0.1, "weighted": 0.1 * priority_accuracy,
                                   "delivered": priority_delivered, "total": priority_total},
            "reward_normalized": {"raw": reward_normalized, "weight": 0.1, "weighted": 0.1 * reward_normalized},
            "final_score": final_score,
        }
        return final_score, explanation
