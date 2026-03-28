"""Typed Observation model for the Urban Delivery Environment."""

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


class PackagePriority(IntEnum):
    NORMAL = 0
    URGENT = 1
    FRAGILE = 2


class WeatherType(IntEnum):
    CLEAR = 0
    RAIN = 1
    FOG = 2


class PackageInfo(BaseModel):
    """Information about a single package."""
    package_id: int
    pickup_position: list[int] = Field(description="[row, col] of pickup location")
    delivery_position: list[int] = Field(description="[row, col] of delivery target")
    picked_up: bool = False
    delivered: bool = False
    priority: int = Field(default=0, description="0=normal, 1=urgent, 2=fragile")
    deadline: Optional[int] = Field(default=None, description="Max steps to deliver, or None")


class VehicleState(BaseModel):
    """Current vehicle state."""
    position: list[int] = Field(description="[row, col] on the grid")
    fuel: float = Field(description="Remaining fuel (0-100)")
    carrying: list[int] = Field(default_factory=list, description="Package IDs being carried")


class DeliveryObservation(BaseModel):
    """Full observation of the environment state.

    Provides everything an agent needs to make decisions:
    - Vehicle position and fuel
    - Package locations, statuses, priorities, deadlines
    - Traffic grid (which cells are congested)
    - Fuel station locations
    - Weather conditions
    - Time tracking
    """
    vehicle: VehicleState
    packages: list[PackageInfo]
    traffic_grid: list[list[int]] = Field(description="2D grid: 0=clear, 1=traffic")
    fuel_stations: list[list[int]] = Field(description="List of [row, col] fuel stations")
    grid_size: int
    time_elapsed: int = 0
    max_steps: int = 200
    weather: int = Field(default=0, description="0=clear, 1=rain, 2=fog")
    total_reward: float = 0.0
    packages_delivered: int = 0
    packages_total: int = 0
    done: bool = False
    message: str = ""
