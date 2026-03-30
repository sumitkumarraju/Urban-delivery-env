"""
Unit tests for the core DeliveryEnvironment.
Tests critical edge cases: walls, fuel, delivery, capacity, hints.
"""
import pytest
from env import DeliveryEnvironment, TaskConfig
from models.action import DeliveryAction, ActionType


@pytest.fixture
def mock_config():
    return TaskConfig(
        name="test",
        grid_size=3,
        num_packages=1,
        initial_fuel=10.0,
        has_traffic=False,
        max_steps=10,
        seed=1,
        num_fuel_stations=1,
        max_carrying=3,
    )


def test_environment_initialization(mock_config):
    """Verify that the grid, packages, and vehicles start correctly."""
    env = DeliveryEnvironment(mock_config)
    obs = env.reset()

    assert obs.grid_size == 3
    assert obs.vehicle.fuel == 10.0
    assert len(obs.packages) == 1
    assert obs.done is False


def test_wall_collision_penalty():
    """Verify the -2 wall penalty prevents moving out of bounds."""
    config = TaskConfig(name="test_wall", grid_size=3, num_packages=1,
                        initial_fuel=10.0, has_traffic=False, max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset(seed=1)

    env._vehicle_row = 0
    env._vehicle_col = 0

    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_UP.value))
    assert obs.vehicle.position == [0, 0]
    assert reward.step_reward == -2.0
    assert "wall_collision" in reward.breakdown


def test_fuel_depletion_termination():
    """Verify that reaching 0 fuel triggers 'done' and a -10 penalty."""
    config = TaskConfig(name="test_fuel", grid_size=5, num_packages=1,
                        initial_fuel=1.0, has_traffic=False, max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset()

    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_DOWN.value))

    assert obs.vehicle.fuel == 0.0
    assert obs.done is True
    assert "fuel_empty" in reward.breakdown
    assert reward.breakdown["fuel_empty"] == -10.0


def test_fuel_station_refill():
    """Verify refueling at a station adds fuel correctly."""
    config = TaskConfig(name="test_refilling", grid_size=2, num_packages=1,
                        initial_fuel=10.0, has_traffic=False, num_fuel_stations=1,
                        max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset()

    r, c = env._fuel_stations[0]
    env._vehicle_row = r
    env._vehicle_col = c
    env._fuel = 10.0

    obs, reward = env.step(DeliveryAction(action=ActionType.REFUEL.value))
    assert obs.vehicle.fuel == 40.0
    assert "refuel" in reward.breakdown
    assert reward.breakdown["refuel"] == 5.0


def test_package_pickup_and_delivery():
    """Verify full pickup and delivery cycle rewards."""
    config = TaskConfig(name="test_delivery", grid_size=5, num_packages=1,
                        initial_fuel=10.0, has_traffic=False, max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset()

    pkg = env._packages[0]
    assert len(env._carrying) == 0

    env._vehicle_row = max(0, pkg.pickup_row - 1)
    env._vehicle_col = pkg.pickup_col

    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_DOWN.value))

    assert pkg.picked_up is True
    assert pkg.package_id in env._carrying
    assert f"pickup_pkg_{pkg.package_id}" in reward.breakdown

    env._vehicle_row = pkg.delivery_row
    env._vehicle_col = pkg.delivery_col

    obs, reward = env.step(DeliveryAction(action=ActionType.DELIVER.value))
    assert pkg.delivered is True
    assert pkg.delivery_step is not None  # delivery_step tracked
    assert pkg.package_id not in env._carrying
    assert f"deliver_pkg_{pkg.package_id}" in reward.breakdown
    assert obs.done is True
    assert "all_delivered_bonus" in reward.breakdown


def test_carrying_capacity():
    """Verify vehicle cannot pick up more than max_carrying packages."""
    config = TaskConfig(name="test_cap", grid_size=10, num_packages=4,
                        initial_fuel=100.0, has_traffic=False, max_carrying=2)
    env = DeliveryEnvironment(config)
    env.reset()

    # Directly teleport onto each pickup and trigger auto-pickup via a small move
    picked_up = 0
    for pkg in env._packages:
        if picked_up >= 3:  # Try to pick up 3 but capacity is 2
            break
        # Place adjacent then move onto it
        env._vehicle_row = pkg.pickup_row
        env._vehicle_col = max(0, pkg.pickup_col - 1)
        if env._vehicle_col == pkg.pickup_col:  # already on it, shift other way
            env._vehicle_col = min(config.grid_size - 1, pkg.pickup_col + 1)

        env.step(DeliveryAction(action=ActionType.MOVE_RIGHT.value if env._vehicle_col < pkg.pickup_col else ActionType.MOVE_LEFT.value))
        picked_up += 1

    # Should have at most 2 (the capacity limit)
    assert len(env._carrying) <= 2


def test_natural_language_hint():
    """Verify the NL hint is generated and contains useful info."""
    config = TaskConfig(name="test_hint", grid_size=5, num_packages=2,
                        initial_fuel=50.0, has_traffic=False, max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset()

    summary = env.get_state_summary()
    assert "hint" in summary
    assert isinstance(summary["hint"], str)
    assert len(summary["hint"]) > 0
    assert "nearest_package_distance" in summary
    assert "nearest_fuel_station_distance" in summary
    assert "carrying_count" in summary
    assert "max_carrying" in summary


def test_low_fuel_hint():
    """Verify fuel warning appears in hint when fuel is critically low."""
    config = TaskConfig(name="test_fuel_hint", grid_size=5, num_packages=1,
                        initial_fuel=5.0, has_traffic=False, max_carrying=3)
    env = DeliveryEnvironment(config)
    env.reset()

    summary = env.get_state_summary()
    assert "CRITICAL" in summary["hint"] or "WARNING" in summary["hint"]
