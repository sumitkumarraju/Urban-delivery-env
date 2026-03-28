"""Easy Task — Beginner-friendly delivery scenario.

Grid: 5x5 | Packages: 2 | Fuel: 100 | Traffic: None | Deadlines: None
Perfect for learning the basics of navigation and delivery.
"""

from env import TaskConfig

EASY_TASK = TaskConfig(
    name="easy",
    grid_size=5,
    num_packages=2,
    initial_fuel=100.0,
    has_traffic=False,
    dynamic_traffic=False,
    has_deadlines=False,
    has_priorities=False,
    has_weather=False,
    num_fuel_stations=2,
    max_steps=100,
    seed=42,
)
