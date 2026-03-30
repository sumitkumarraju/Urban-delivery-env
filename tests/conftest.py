"""Shared test fixtures for the Urban Delivery Environment."""

import pytest
from env import DeliveryEnvironment, TaskConfig
from models.action import DeliveryAction


@pytest.fixture
def easy_config():
    """Minimal config for fast, deterministic testing."""
    return TaskConfig(
        name="test_easy",
        grid_size=3,
        num_packages=1,
        initial_fuel=10.0,
        has_traffic=False,
        max_steps=10,
        seed=1,
        num_fuel_stations=1,
        max_carrying=3,
    )


@pytest.fixture
def hard_config():
    """Full-featured config for comprehensive testing."""
    return TaskConfig(
        name="test_hard",
        grid_size=5,
        num_packages=2,
        initial_fuel=20.0,
        has_traffic=True,
        dynamic_traffic=True,
        has_deadlines=True,
        has_priorities=True,
        has_weather=True,
        max_steps=50,
        seed=42,
        num_fuel_stations=2,
        max_carrying=2,
    )


@pytest.fixture
def easy_env(easy_config):
    """Ready-to-use easy environment."""
    env = DeliveryEnvironment(easy_config)
    env.reset()
    return env
