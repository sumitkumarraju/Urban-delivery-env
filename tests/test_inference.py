"""Tests for the inference heuristic agent.

Verifies that the heuristic agent can solve tasks with reasonable scores.
"""

import pytest
from inference import heuristic_action, run_task
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


def test_heuristic_easy_task_delivers():
    """Heuristic agent should deliver at least 1 package on easy."""
    actions = run_task("easy", None, "", False)
    grader = EasyGrader()
    score = grader.grade(actions)
    assert score > 0.4, f"Easy score too low: {score}"


def test_heuristic_medium_task_nonzero():
    """Heuristic agent should score above baseline on medium."""
    actions = run_task("medium", None, "", False)
    grader = MediumGrader()
    score = grader.grade(actions)
    assert score > 0.1, f"Medium score too low: {score}"


def test_heuristic_hard_task_nonzero():
    """Heuristic agent should score above baseline on hard."""
    actions = run_task("hard", None, "", False)
    grader = HardGrader()
    score = grader.grade(actions)
    assert score > 0.1, f"Hard score too low: {score}"


def test_heuristic_action_never_crashes():
    """Heuristic must handle any valid state dict without crashing."""
    state = {
        "position": [0, 0],
        "fuel": 50.0,
        "carrying": [],
        "packages": [
            {
                "id": 0,
                "pickup": [2, 2],
                "delivery": [4, 4],
                "picked_up": False,
                "delivered": False,
                "priority": "normal",
            }
        ],
        "fuel_stations": [[1, 1]],
        "grid_size": 5,
        "traffic_cells": [[0, 1]],
    }
    action = heuristic_action(state)
    assert 0 <= action <= 5


def test_heuristic_low_fuel_refuels():
    """When fuel is critical, heuristic should head to fuel station."""
    state = {
        "position": [1, 0],
        "fuel": 3.0,
        "carrying": [],
        "packages": [
            {
                "id": 0,
                "pickup": [4, 4],
                "delivery": [0, 0],
                "picked_up": False,
                "delivered": False,
                "priority": "normal",
            }
        ],
        "fuel_stations": [[1, 1]],
        "grid_size": 5,
        "traffic_cells": [],
    }
    action = heuristic_action(state)
    assert action == 3, "Should move right toward fuel station at [1,1]"


def test_heuristic_at_station_refuels():
    """When at fuel station with low fuel, should refuel."""
    state = {
        "position": [1, 1],
        "fuel": 3.0,
        "carrying": [],
        "packages": [
            {
                "id": 0,
                "pickup": [4, 4],
                "delivery": [0, 0],
                "picked_up": False,
                "delivered": False,
                "priority": "normal",
            }
        ],
        "fuel_stations": [[1, 1]],
        "grid_size": 5,
        "traffic_cells": [],
    }
    action = heuristic_action(state)
    assert action == 5, "Should refuel when at station with low fuel"


def test_heuristic_delivers_when_at_target():
    """When carrying and at delivery location, should deliver."""
    state = {
        "position": [3, 3],
        "fuel": 50.0,
        "carrying": [0],
        "packages": [
            {
                "id": 0,
                "pickup": [1, 1],
                "delivery": [3, 3],
                "picked_up": True,
                "delivered": False,
                "priority": "normal",
            }
        ],
        "fuel_stations": [[0, 0]],
        "grid_size": 5,
        "traffic_cells": [],
    }
    action = heuristic_action(state)
    assert action == 4, "Should deliver when at delivery location"
