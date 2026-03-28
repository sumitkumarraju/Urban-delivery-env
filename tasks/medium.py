"""Medium Task — Intermediate delivery challenge.

Grid: 8x8 | Packages: 3 | Fuel: 60 | Traffic: Static | Deadlines: Some
Introduces traffic avoidance and basic time management.
"""

from env import TaskConfig

MEDIUM_TASK = TaskConfig(
    name="medium",
    grid_size=8,
    num_packages=3,
    initial_fuel=60.0,
    has_traffic=True,
    dynamic_traffic=False,
    has_deadlines=True,
    has_priorities=False,
    has_weather=False,
    num_fuel_stations=3,
    max_steps=200,
    seed=123,
)
