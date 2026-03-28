"""Hard Task — Expert delivery scenario.

Grid: 10x10 | Packages: 5 | Fuel: 40 | Traffic: Dynamic | All features enabled
Dynamic traffic, weather effects, package priorities, and tight deadlines.
"""

from env import TaskConfig

HARD_TASK = TaskConfig(
    name="hard",
    grid_size=10,
    num_packages=5,
    initial_fuel=40.0,
    has_traffic=True,
    dynamic_traffic=True,
    has_deadlines=True,
    has_priorities=True,
    has_weather=True,
    num_fuel_stations=3,
    max_steps=300,
    seed=456,
)
