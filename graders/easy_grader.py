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


if __name__ == "__main__":
    grader = EasyGrader()
    # Demo with random actions
    import random
    random.seed(42)
    actions = [random.randint(0, 5) for _ in range(100)]
    score = grader.grade(actions)
    print(f"Easy Task Score: {score:.4f}")
