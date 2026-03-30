"""Easy Grader — Simple completion-based scoring.

Score = packages_delivered / total_packages

This is the most straightforward metric: did you deliver the packages?
"""

from graders.base_grader import BaseGrader
from tasks.easy import EASY_TASK


class EasyGrader(BaseGrader):
    """Grades the easy task based on delivery completion ratio."""

    def __init__(self):
        super().__init__(EASY_TASK)

    def score(self, episode_stats: dict) -> float:
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]
        if total == 0:
            return 0.0
        return delivered / total

    def score_with_explanation(self, episode_stats: dict) -> tuple[float, dict]:
        """Return score with explanation breakdown."""
        delivered = episode_stats["packages_delivered"]
        total = episode_stats["packages_total"]
        completion = delivered / total if total > 0 else 0.0
        return completion, {
            "completion": {"delivered": delivered, "total": total, "ratio": completion},
            "final_score": completion,
        }
