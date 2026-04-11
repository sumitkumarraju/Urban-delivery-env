"""Extended observation carrying delivery-specific fields beyond done/reward."""

from typing import Any, Dict, List, Optional
from pydantic import ConfigDict, Field
from openenv.core.env_server.types import Observation


class DeliveryObservationResponse(Observation):
    """Rich observation returned by reset() and step() on the server side.

    Inherits done/reward/metadata from Observation base class. Adds
    delivery-specific fields for vehicle state, package tracking, and
    natural language hints.
    """

    model_config = ConfigDict(extra="forbid")

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
