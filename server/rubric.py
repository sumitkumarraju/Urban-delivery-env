"""Rubric for the Urban Delivery Environment (RFC 004 compliant).

Computes step-level rewards based on the delivery environment's
internal reward signal, normalized to [0, 1].
"""

from typing import Any

from openenv.core.rubrics.base import Rubric


class DeliveryRubric(Rubric):
    """Step-level reward for delivery actions.

    Returns the per-step reward from the environment observation,
    normalized to [0, 1] range for compatibility with the OpenEnv
    training infrastructure.
    """

    def forward(self, action: Any, observation: Any) -> float:
        reward = getattr(observation, "reward", None)
        if reward is None:
            return 0.0
        return max(0.0, min(1.0, (float(reward) + 10.0) / 60.0))


class CompletionRubric(Rubric):
    """Episode-level rubric based on delivery completion ratio."""

    def forward(self, action: Any, observation: Any) -> float:
        detail = getattr(observation, "observation_detail", {})
        if not detail:
            return 0.0
        delivered = detail.get("packages_delivered", 0)
        total = detail.get("packages_total", 1)
        if total == 0:
            return 0.0
        return delivered / total


class UrbanDeliveryRubric(Rubric):
    """Composite rubric combining step rewards and completion tracking."""

    def __init__(self):
        super().__init__()
        self.step_reward = DeliveryRubric()
        self.completion = CompletionRubric()

    def forward(self, action: Any, observation: Any) -> float:
        step_score = self.step_reward(action, observation)
        completion_score = self.completion(action, observation)
        return 0.6 * completion_score + 0.4 * step_score
