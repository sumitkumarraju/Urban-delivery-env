"""
Urban Delivery Environment — OpenEnv MCP Server Integration.

Wraps the core DeliveryEnvironment to expose it through OpenEnv's
MCPEnvironment interface with FastMCP tools for agent interaction.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from server.observation import DeliveryObservationResponse
from fastmcp import FastMCP

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DeliveryEnvironment, TaskConfig
from models.action import DeliveryAction
from tasks import ALL_TASKS


class UrbanDeliveryEnvironment(MCPEnvironment):
    """OpenEnv MCPEnvironment that wraps the Urban Delivery simulation.

    Exposes the environment through MCP tools:
    - move(direction): Move the vehicle (up/down/left/right)
    - deliver(): Attempt to deliver a carried package
    - refuel(): Refuel at a fuel station
    - get_observation(): Get current environment state
    - set_task(difficulty): Switch task difficulty
    """

    def __init__(self, task_name: str = "easy"):
        mcp = FastMCP("urban_delivery_env")
        self._current_task = task_name
        self._config = ALL_TASKS[task_name]
        self._env = DeliveryEnvironment(self._config)
        self._last_reward_info = None

        @mcp.tool
        def move(direction: str) -> dict:
            """Move the delivery vehicle in the specified direction.

            Args:
                direction: One of 'up', 'down', 'left', 'right'

            Returns:
                Dictionary with observation and reward info
            """
            direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
            d = direction.lower().strip()
            if d not in direction_map:
                return {"error": f"Invalid direction '{d}'. Use: up, down, left, right"}

            action = DeliveryAction(action=direction_map[d])
            obs, reward = self._env.step(action)
            self._last_reward_info = reward
            return {
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
            }

        @mcp.tool
        def deliver() -> dict:
            """Attempt to deliver a package at the current position.

            Returns:
                Dictionary with observation and reward info
            """
            action = DeliveryAction(action=4)
            obs, reward = self._env.step(action)
            self._last_reward_info = reward
            return {
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
            }

        @mcp.tool
        def refuel() -> dict:
            """Refuel the vehicle at the current position (must be at fuel station).

            Returns:
                Dictionary with observation and reward info
            """
            action = DeliveryAction(action=5)
            obs, reward = self._env.step(action)
            self._last_reward_info = reward
            return {
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
            }

        @mcp.tool
        def get_observation() -> dict:
            """Get the current state of the environment without taking an action.

            Returns:
                Full environment observation including vehicle, packages, grid
            """
            return self._env.get_state_summary()

        @mcp.tool
        def set_task(difficulty: str) -> dict:
            """Set the task difficulty level.

            Args:
                difficulty: One of 'easy', 'medium', 'hard'

            Returns:
                Task configuration details
            """
            d = difficulty.lower().strip()
            if d not in ALL_TASKS:
                return {"error": f"Invalid difficulty '{d}'. Use: easy, medium, hard"}

            self._current_task = d
            self._config = ALL_TASKS[d]
            self._env = DeliveryEnvironment(self._config)
            obs = self._env.reset()
            return {
                "task": d,
                "config": {
                    "grid_size": self._config.grid_size,
                    "num_packages": self._config.num_packages,
                    "initial_fuel": self._config.initial_fuel,
                    "has_traffic": self._config.has_traffic,
                    "dynamic_traffic": self._config.dynamic_traffic,
                    "has_deadlines": self._config.has_deadlines,
                    "has_priorities": self._config.has_priorities,
                    "has_weather": self._config.has_weather,
                    "max_steps": self._config.max_steps,
                },
                "observation": obs.model_dump(),
            }

        @mcp.tool
        def get_hint() -> dict:
            """Get a natural language hint about what to do next.

            Returns:
                Dictionary with hint text and key distances
            """
            summary = self._env.get_state_summary()
            return {
                "hint": summary["hint"],
                "nearest_package_distance": summary["nearest_package_distance"],
                "nearest_fuel_station_distance": summary["nearest_fuel_station_distance"],
                "carrying": summary["carrying_count"],
                "max_carrying": summary["max_carrying"],
            }

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DeliveryObservationResponse:
        """Reset the environment to initial state."""
        self._env = DeliveryEnvironment(self._config)
        obs = self._env.reset(seed=seed)
        summary = self._env.get_state_summary()

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return DeliveryObservationResponse(
            done=False,
            reward=0.0,
            status="ready",
            task=self._current_task,
            message=f"Urban Delivery Environment ready! Task: {self._current_task}",
            vehicle_position=list(obs.vehicle.position),
            fuel_remaining=obs.vehicle.fuel,
            packages_delivered=obs.packages_delivered,
            packages_total=obs.packages_total,
            step_count=0,
            carrying_count=summary.get("carrying_count", 0),
            hint=summary.get("hint", ""),
            grid_size=self._config.grid_size,
            observation_detail=obs.model_dump(),
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DeliveryObservationResponse:
        """Handle non-MCP actions (fallback)."""
        return DeliveryObservationResponse(
            done=False,
            reward=0.0,
            status="error",
            task=self._current_task,
            message=f"Unknown action type: {type(action).__name__}. "
                    "Use MCP tools: move, deliver, refuel, get_observation, set_task.",
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Delegates to MCPEnvironment for MCP actions."""
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
