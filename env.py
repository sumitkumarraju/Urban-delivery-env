"""Core simulation: grid navigation, fuel, traffic, weather, package delivery."""

import random
from dataclasses import dataclass, field
from typing import Optional

from models.action import ActionType, DeliveryAction
from models.observation import (
    DeliveryObservation,
    PackageInfo,
    PackagePriority,
    VehicleState,
    WeatherType,
)
from models.reward import RewardInfo


@dataclass
class TaskConfig:
    """Configuration for a specific task difficulty level."""
    name: str
    grid_size: int
    num_packages: int
    initial_fuel: float
    has_traffic: bool
    dynamic_traffic: bool = False
    has_deadlines: bool = False
    has_priorities: bool = False
    has_weather: bool = False
    num_fuel_stations: int = 2
    max_steps: int = 200
    seed: int = 42
    max_carrying: int = 3


@dataclass
class Package:
    """Internal package state."""
    package_id: int
    pickup_row: int
    pickup_col: int
    delivery_row: int
    delivery_col: int
    picked_up: bool = False
    delivered: bool = False
    priority: PackagePriority = PackagePriority.NORMAL
    deadline: Optional[int] = None
    delivery_step: Optional[int] = None


class DeliveryEnvironment:
    """Core delivery simulation engine.

    Manages a grid-based city where a vehicle navigates to pick up and
    deliver packages while dealing with traffic, fuel, weather, and deadlines.
    """

    def __init__(self, config: TaskConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._message = ""

        self._vehicle_row = 0
        self._vehicle_col = 0
        self._fuel = config.initial_fuel

        self._carrying: list[int] = []
        self._packages: list[Package] = []
        self._traffic_grid: list[list[int]] = []
        self._fuel_stations: list[tuple[int, int]] = []
        self._weather = WeatherType.CLEAR

        self._initialize()

    def _initialize(self) -> None:
        """Set up the grid, packages, traffic, and fuel stations."""
        gs = self.config.grid_size

        # Place vehicle at random position
        self._vehicle_row = self._rng.randint(0, gs - 1)
        self._vehicle_col = self._rng.randint(0, gs - 1)

        # Generate traffic grid
        self._traffic_grid = [[0] * gs for _ in range(gs)]
        if self.config.has_traffic:
            num_traffic = max(1, gs * gs // 5)
            for _ in range(num_traffic):
                r, c = self._rng.randint(0, gs - 1), self._rng.randint(0, gs - 1)
                if (r, c) != (self._vehicle_row, self._vehicle_col):
                    self._traffic_grid[r][c] = 1

        # Place fuel stations
        self._fuel_stations = []
        placed = set()
        placed.add((self._vehicle_row, self._vehicle_col))
        for _ in range(self.config.num_fuel_stations):
            attempts = 0
            while attempts < 100:
                r, c = self._rng.randint(0, gs - 1), self._rng.randint(0, gs - 1)
                if (r, c) not in placed:
                    self._fuel_stations.append((r, c))
                    placed.add((r, c))
                    self._traffic_grid[r][c] = 0  # Clear traffic at fuel stations
                    break
                attempts += 1

        # Generate packages with unique pickup/delivery locations
        self._packages = []
        for i in range(self.config.num_packages):
            attempts = 0
            while attempts < 200:
                pr, pc = self._rng.randint(0, gs - 1), self._rng.randint(0, gs - 1)
                dr, dc = self._rng.randint(0, gs - 1), self._rng.randint(0, gs - 1)
                if (pr, pc) != (dr, dc) and (pr, pc) not in placed:
                    break
                attempts += 1

            priority = PackagePriority.NORMAL
            deadline = None

            if self.config.has_priorities:
                priority = PackagePriority(self._rng.choice([0, 0, 1, 2]))

            if self.config.has_deadlines:
                dist = abs(pr - dr) + abs(pc - dc)
                base_time = dist * 3 + 10
                if priority == PackagePriority.URGENT:
                    deadline = max(base_time, 15)
                else:
                    deadline = base_time + self._rng.randint(5, 20)

            pkg = Package(
                package_id=i,
                pickup_row=pr,
                pickup_col=pc,
                delivery_row=dr,
                delivery_col=dc,
                priority=priority,
                deadline=deadline,
            )
            self._packages.append(pkg)
            placed.add((pr, pc))

        # Weather
        if self.config.has_weather:
            self._weather = WeatherType(self._rng.choice([0, 0, 0, 1, 2]))

    def reset(self, seed: Optional[int] = None) -> DeliveryObservation:
        """Reset the environment to initial state.

        Args:
            seed: Optional seed override. Uses config seed if None.

        Returns:
            Initial observation.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self.config.seed)

        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._message = "Environment reset. Ready for delivery!"
        self._fuel = self.config.initial_fuel
        self._carrying = []
        self._initialize()

        return self._get_observation()

    def step(self, action: DeliveryAction) -> tuple[DeliveryObservation, RewardInfo]:
        """Execute one step in the environment.

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward_info).
        """
        if self._done:
            return self._get_observation(), RewardInfo(
                total_reward=0.0,
                step_reward=0.0,
                breakdown={"episode_done": 0.0},
                cumulative_reward=self._total_reward,
            )

        self._step_count += 1
        reward_breakdown: dict[str, float] = {}
        step_reward = 0.0

        act = ActionType(action.action)

        # Dynamic traffic update
        if self.config.dynamic_traffic and self._step_count % 5 == 0:
            self._update_traffic()

        # Weather changes occasionally
        if self.config.has_weather and self._step_count % 15 == 0:
            self._weather = WeatherType(self._rng.choice([0, 0, 0, 1, 2]))

        if act in (ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT):
            step_reward, reward_breakdown = self._handle_movement(act)
        elif act == ActionType.DELIVER:
            step_reward, reward_breakdown = self._handle_deliver()
        elif act == ActionType.REFUEL:
            step_reward, reward_breakdown = self._handle_refuel()

        # Check fuel depletion
        if self._fuel <= 0 and not self._done:
            self._fuel = 0
            self._done = True
            step_reward += -10.0
            reward_breakdown["fuel_empty"] = -10.0
            self._message = "Out of fuel! Episode terminated."

        # Check max steps
        if self._step_count >= self.config.max_steps and not self._done:
            self._done = True
            self._message = "Max steps reached. Episode terminated."

        # Check all delivered
        all_delivered = all(p.delivered for p in self._packages)
        if all_delivered and not self._done:
            step_reward += 50.0
            reward_breakdown["all_delivered_bonus"] = 50.0
            self._done = True
            self._message = "All packages delivered! Congratulations!"

        self._total_reward += step_reward

        reward_info = RewardInfo(
            total_reward=self._total_reward,
            step_reward=step_reward,
            breakdown=reward_breakdown,
            cumulative_reward=self._total_reward,
        )

        return self._get_observation(), reward_info

    def _handle_movement(self, direction: ActionType) -> tuple[float, dict[str, float]]:
        """Process a movement action."""
        reward = 0.0
        breakdown: dict[str, float] = {}

        dr, dc = {
            ActionType.MOVE_UP: (-1, 0),
            ActionType.MOVE_DOWN: (1, 0),
            ActionType.MOVE_LEFT: (0, -1),
            ActionType.MOVE_RIGHT: (0, 1),
        }[direction]

        new_r = self._vehicle_row + dr
        new_c = self._vehicle_col + dc

        gs = self.config.grid_size
        if 0 <= new_r < gs and 0 <= new_c < gs:
            self._vehicle_row = new_r
            self._vehicle_col = new_c

            # Base movement cost
            fuel_cost = 1.0
            reward -= 1.0
            breakdown["movement"] = -1.0

            # Traffic penalty
            if self._traffic_grid[new_r][new_c] == 1:
                reward -= 5.0
                fuel_cost += 1.0
                breakdown["traffic_penalty"] = -5.0

            # Weather penalty
            if self._weather == WeatherType.RAIN:
                reward -= 2.0
                fuel_cost += 0.5
                breakdown["weather_rain"] = -2.0
            elif self._weather == WeatherType.FOG:
                reward -= 1.0
                breakdown["weather_fog"] = -1.0

            self._fuel -= fuel_cost

            # Auto-pickup: pick up packages at current position (respect capacity)
            for pkg in self._packages:
                if (not pkg.picked_up and not pkg.delivered
                        and pkg.pickup_row == new_r and pkg.pickup_col == new_c
                        and pkg.package_id not in self._carrying):
                    if len(self._carrying) >= self.config.max_carrying:
                        self._message = f"Vehicle full ({self.config.max_carrying} max)! Deliver before picking up more."
                        break
                    pkg.picked_up = True
                    self._carrying.append(pkg.package_id)
                    reward += 2.0
                    breakdown[f"pickup_pkg_{pkg.package_id}"] = 2.0
                    self._message = f"Picked up package {pkg.package_id}!"
        else:
            reward -= 2.0
            breakdown["wall_collision"] = -2.0
            self._message = "Hit the wall! Can't move there."

        return reward, breakdown

    def _handle_deliver(self) -> tuple[float, dict[str, float]]:
        """Process a deliver action."""
        reward = 0.0
        breakdown: dict[str, float] = {}
        delivered_any = False

        for pkg_id in list(self._carrying):
            pkg = self._packages[pkg_id]
            if (pkg.delivery_row == self._vehicle_row
                    and pkg.delivery_col == self._vehicle_col):
                pkg.delivered = True
                pkg.delivery_step = self._step_count  # Track exact delivery time
                self._carrying.remove(pkg_id)
                delivered_any = True

                # Base delivery reward
                reward += 20.0
                breakdown[f"deliver_pkg_{pkg_id}"] = 20.0

                # Priority bonus
                if pkg.priority == PackagePriority.URGENT:
                    reward += 15.0
                    breakdown[f"urgent_bonus_{pkg_id}"] = 15.0
                elif pkg.priority == PackagePriority.FRAGILE:
                    reward += 10.0
                    breakdown[f"fragile_bonus_{pkg_id}"] = 10.0

                # Deadline check — uses per-package delivery_step, not total steps
                if pkg.deadline is not None:
                    if pkg.delivery_step <= pkg.deadline:
                        reward += 10.0
                        breakdown[f"deadline_met_{pkg_id}"] = 10.0
                    else:
                        reward -= 5.0
                        breakdown[f"deadline_missed_{pkg_id}"] = -5.0

                self._message = f"Delivered package {pkg_id}!"

        if not delivered_any:
            reward -= 1.0
            breakdown["failed_deliver"] = -1.0
            if not self._carrying:
                self._message = "No packages to deliver! Pick up a package first."
            else:
                self._message = "Not at a delivery location for any carried package."

        return reward, breakdown

    def _handle_refuel(self) -> tuple[float, dict[str, float]]:
        """Process a refuel action."""
        reward = 0.0
        breakdown: dict[str, float] = {}

        at_station = any(
            r == self._vehicle_row and c == self._vehicle_col
            for r, c in self._fuel_stations
        )

        if at_station:
            old_fuel = self._fuel
            self._fuel = min(100.0, self._fuel + 30.0)
            gained = self._fuel - old_fuel
            if gained > 0:
                reward += 5.0
                breakdown["refuel"] = 5.0
                self._message = f"Refueled! +{gained:.0f} fuel. Current: {self._fuel:.0f}"
            else:
                reward -= 0.5
                breakdown["tank_full"] = -0.5
                self._message = "Tank already full!"
        else:
            reward -= 1.0
            breakdown["not_at_station"] = -1.0
            self._message = "Not at a fuel station! Find one on the map."

        return reward, breakdown

    def _update_traffic(self) -> None:
        """Dynamically update traffic patterns."""
        gs = self.config.grid_size
        station_set = set(self._fuel_stations)

        for r in range(gs):
            for c in range(gs):
                if (r, c) in station_set:
                    continue
                if (r, c) == (self._vehicle_row, self._vehicle_col):
                    continue
                if self._rng.random() < 0.15:
                    self._traffic_grid[r][c] = 1 - self._traffic_grid[r][c]

    def _get_observation(self) -> DeliveryObservation:
        """Build current observation."""
        packages_info = []
        delivered_count = 0
        for pkg in self._packages:
            if pkg.delivered:
                delivered_count += 1
            packages_info.append(PackageInfo(
                package_id=pkg.package_id,
                pickup_position=[pkg.pickup_row, pkg.pickup_col],
                delivery_position=[pkg.delivery_row, pkg.delivery_col],
                picked_up=pkg.picked_up,
                delivered=pkg.delivered,
                priority=pkg.priority.value,
                deadline=pkg.deadline,
                delivery_step=pkg.delivery_step,
            ))

        return DeliveryObservation(
            vehicle=VehicleState(
                position=[self._vehicle_row, self._vehicle_col],
                fuel=round(self._fuel, 1),
                carrying=list(self._carrying),
            ),
            packages=packages_info,
            traffic_grid=self._traffic_grid,
            fuel_stations=[[r, c] for r, c in self._fuel_stations],
            grid_size=self.config.grid_size,
            time_elapsed=self._step_count,
            max_steps=self.config.max_steps,
            weather=self._weather.value,
            total_reward=round(self._total_reward, 2),
            packages_delivered=delivered_count,
            packages_total=len(self._packages),
            done=self._done,
            message=self._message,
        )

    def get_state_summary(self) -> dict:
        """Return a compact state dict for LLM context."""
        obs = self._get_observation()
        vr, vc = obs.vehicle.position

        # Compute nearest undelivered package distance
        nearest_pkg_dist = None
        nearest_station_dist = None
        for p in obs.packages:
            if not p.picked_up and not p.delivered:
                d = abs(p.pickup_position[0] - vr) + abs(p.pickup_position[1] - vc)
                if nearest_pkg_dist is None or d < nearest_pkg_dist:
                    nearest_pkg_dist = d
            elif p.picked_up and not p.delivered:
                d = abs(p.delivery_position[0] - vr) + abs(p.delivery_position[1] - vc)
                if nearest_pkg_dist is None or d < nearest_pkg_dist:
                    nearest_pkg_dist = d
        for s in obs.fuel_stations:
            d = abs(s[0] - vr) + abs(s[1] - vc)
            if nearest_station_dist is None or d < nearest_station_dist:
                nearest_station_dist = d

        # Build natural language hint
        hint = self._build_hint(obs, nearest_pkg_dist, nearest_station_dist)

        return {
            "position": obs.vehicle.position,
            "fuel": obs.vehicle.fuel,
            "carrying": obs.vehicle.carrying,
            "carrying_count": len(obs.vehicle.carrying),
            "max_carrying": self.config.max_carrying,
            "packages": [
                {
                    "id": p.package_id,
                    "pickup": p.pickup_position,
                    "delivery": p.delivery_position,
                    "picked_up": p.picked_up,
                    "delivered": p.delivered,
                    "priority": ["normal", "urgent", "fragile"][p.priority],
                    "deadline": p.deadline,
                }
                for p in obs.packages
            ],
            "fuel_stations": obs.fuel_stations,
            "grid_size": obs.grid_size,
            "traffic_cells": [
                [r, c]
                for r in range(obs.grid_size)
                for c in range(obs.grid_size)
                if obs.traffic_grid[r][c] == 1
            ],
            "weather": ["clear", "rain", "fog"][obs.weather],
            "step": obs.time_elapsed,
            "max_steps": obs.max_steps,
            "delivered": obs.packages_delivered,
            "total": obs.packages_total,
            "reward": obs.total_reward,
            "done": obs.done,
            "nearest_package_distance": nearest_pkg_dist,
            "nearest_fuel_station_distance": nearest_station_dist,
            "hint": hint,
        }

    def _build_hint(self, obs, nearest_pkg_dist, nearest_station_dist) -> str:
        """Generate a natural language hint for LLM agents."""
        parts = []
        vr, vc = obs.vehicle.position
        fuel = obs.vehicle.fuel

        # Fuel warning
        if fuel <= 5:
            parts.append(f"CRITICAL: Fuel at {fuel:.0f}! Refuel immediately or you will die.")
        elif fuel <= 15:
            parts.append(f"WARNING: Fuel low ({fuel:.0f}). Consider refueling soon.")

        # Carrying status
        carrying = obs.vehicle.carrying
        if carrying:
            # Find delivery targets for carried packages
            targets = []
            for p in obs.packages:
                if p.package_id in carrying and not p.delivered:
                    targets.append(f"pkg {p.package_id} → [{p.delivery_position[0]},{p.delivery_position[1]}]")
            parts.append(f"Carrying {len(carrying)} package(s): {', '.join(targets)}. Navigate to delivery location and use DELIVER.")
        else:
            # Find nearest undelivered package
            undelivered = [p for p in obs.packages if not p.picked_up and not p.delivered]
            if undelivered:
                urgent = [p for p in undelivered if p.priority == 1]
                if urgent:
                    p = urgent[0]
                    parts.append(f"URGENT package {p.package_id} at [{p.pickup_position[0]},{p.pickup_position[1]}] — pick it up first!")
                else:
                    parts.append(f"{len(undelivered)} packages waiting for pickup. Navigate to a pickup location.")
            else:
                parts.append("All packages picked up or delivered!")

        # Weather
        weather_names = ["clear", "rainy (extra fuel cost)", "foggy (reduced visibility)"]
        if obs.weather > 0:
            parts.append(f"Weather: {weather_names[obs.weather]}.")

        # Deadline warnings
        for p in obs.packages:
            if p.deadline and not p.delivered and p.picked_up:
                remaining = p.deadline - obs.time_elapsed
                if remaining <= 5:
                    parts.append(f"DEADLINE ALERT: Package {p.package_id} must be delivered in {remaining} steps!")

        return " ".join(parts) if parts else "All clear. Proceed with deliveries."
