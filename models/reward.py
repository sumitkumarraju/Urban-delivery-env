"""Typed Reward model for the Urban Delivery Environment."""

from pydantic import BaseModel, Field


class RewardInfo(BaseModel):
    """Detailed reward breakdown for a single step.

    Reward Shaping Design:
        - Package delivery:     +20 base
        - Priority bonus:       +15 for urgent, +10 for fragile
        - Deadline met:         +10
        - Deadline missed:      -5
        - Movement:             -1 per step
        - Traffic penalty:      -5 for entering traffic cell
        - Refuel at station:    +5
        - Fuel empty:           -10 (episode terminates)
        - All packages done:    +50 completion bonus
        - Weather slow:         -2 for moving in rain
    """
    total_reward: float = Field(description="Net reward for this step")
    step_reward: float = Field(description="Reward earned this step only")
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Itemized reward components for this step",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total cumulative reward across all steps",
    )
