"""
OpenEnv-compatible Observation subclass for the Urban Delivery Environment.

Fields added here (beyond done/reward/metadata) are serialized into the
`observation` dict by OpenEnv's serialize_observation.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Observation


class DeliveryObservationResponse(Observation):
    """Observation returned by the Urban Delivery Environment's reset/step."""

    model_config = {"extra": "allow"}

    status: str = Field(default="ready", description="Environment status")
    task: str = Field(default="easy", description="Current task difficulty")
    message: str = Field(default="", description="Human-readable message")
    vehicle_position: List[int] = Field(default_factory=list, description="[row, col]")
    fuel_remaining: float = Field(default=100.0, description="Current fuel level")
    packages_delivered: int = Field(default=0, description="Packages delivered so far")
    packages_total: int = Field(default=0, description="Total packages to deliver")
    step_count: int = Field(default=0, description="Current step in the episode")
    carrying_count: int = Field(default=0, description="Packages currently carried")
    hint: str = Field(default="", description="Natural language hint for LLM agents")
    grid_size: int = Field(default=5, description="Grid dimensions")
    observation_detail: Dict[str, Any] = Field(
        default_factory=dict, description="Full observation data"
    )
