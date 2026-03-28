"""
Unit tests for the core DeliveryEnvironment.
Adheres strictly to 'test-driven-development' and 'verification-before-completion'.
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
    config = TaskConfig(name="test_wall", grid_size=3, num_packages=1, initial_fuel=10.0, has_traffic=False)
    env = DeliveryEnvironment(config)
    obs = env.reset(seed=1) # Vehicle likely not at boundary, force it
    
    # We force the vehicle to 0,0 for deterministic testing
    env._vehicle_row = 0
    env._vehicle_col = 0
    
    # Try moving UP out of bounds
    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_UP.value))
    assert obs.vehicle.position == [0, 0] # Did not move
    assert reward.step_reward == -2.0 # Only wall penalty applied, movement cost skipped
    assert "wall_collision" in reward.breakdown


def test_fuel_depletion_termination():
    """Verify that reaching 0 fuel triggers 'done' and a -10 penalty."""
    config = TaskConfig(name="test_fuel", grid_size=5, num_packages=1, initial_fuel=1.0, has_traffic=False)
    env = DeliveryEnvironment(config)
    env.reset()
    
    # Try one move. Base movement costs 1.0 fuel.
    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_DOWN.value))
    
    # Fuel should be 0, and episode should terminate
    assert obs.vehicle.fuel == 0.0
    assert obs.done is True
    assert "fuel_empty" in reward.breakdown
    assert reward.breakdown["fuel_empty"] == -10.0


def test_fuel_station_refill():
    """Verify refueling at a station adds fuel correctly."""
    config = TaskConfig(name="test_refilling", grid_size=2, num_packages=1, initial_fuel=10.0, has_traffic=False, num_fuel_stations=1)
    env = DeliveryEnvironment(config)
    env.reset()

    # Teleport to the fuel station
    r, c = env._fuel_stations[0]
    env._vehicle_row = r
    env._vehicle_col = c
    env._fuel = 10.0

    obs, reward = env.step(DeliveryAction(action=ActionType.REFUEL.value))
    assert obs.vehicle.fuel == 40.0 # 10 + 30
    assert "refuel" in reward.breakdown
    assert reward.breakdown["refuel"] == 5.0


def test_package_pickup_and_delivery():
    """Verify full pickup and delivery cycle rewards."""
    config = TaskConfig(name="test_delivery", grid_size=5, num_packages=1, initial_fuel=10.0, has_traffic=False)
    env = DeliveryEnvironment(config)
    env.reset()

    pkg = env._packages[0]
    
    # Not carrying
    assert len(env._carrying) == 0
    
    # Teleport to pickup location
    env._vehicle_row = pkg.pickup_row
    env._vehicle_col = pkg.pickup_col
    
    # Move normally (in place essentially) to trigger pickup logic since pickup triggers post-move.
    # We will teleport adjacent and move on top.
    env._vehicle_row = max(0, pkg.pickup_row - 1)
    env._vehicle_col = pkg.pickup_col
    
    obs, reward = env.step(DeliveryAction(action=ActionType.MOVE_DOWN.value))
    
    assert pkg.picked_up is True
    assert pkg.package_id in env._carrying
    assert f"pickup_pkg_{pkg.package_id}" in reward.breakdown
    
    # Teleport to delivery
    env._vehicle_row = pkg.delivery_row
    env._vehicle_col = pkg.delivery_col
    
    obs, reward = env.step(DeliveryAction(action=ActionType.DELIVER.value))
    assert pkg.delivered is True
    assert pkg.package_id not in env._carrying
    assert f"deliver_pkg_{pkg.package_id}" in reward.breakdown
    
    # Since it's the only package, 'all_delivered' should trigger.
    assert obs.done is True
    assert "all_delivered_bonus" in reward.breakdown
