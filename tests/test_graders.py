"""
Unit tests for the Graders.
Adheres to 'verification-before-completion' rule.
"""
import pytest
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


def test_graders_instantiation():
    """Ensure all graders init cleanly without side-effects."""
    e = EasyGrader()
    m = MediumGrader()
    h = HardGrader()
    assert e.config.name == "easy"
    assert m.config.name == "medium"
    assert h.config.name == "hard"


def test_graders_score_bounds():
    """Mock an episode stats dict to verify that the scoring math bounds to [0.0, 1.0]."""
    mock_stats = {
        "packages_delivered": 2,
        "packages_total": 2,
        "total_reward": 50,
        "steps": 10,
        "max_steps": 100,
        "fuel_used": 10.0,
        "initial_fuel": 100.0,
        "done": True,
        "final_observation": None  # HardGrader calculates offline the priorities, we will mock them
    }

    # Easy
    easy = EasyGrader()
    assert easy.score(mock_stats) == 1.0

    # Medium weights -> 0.5*1.0 + 0.3*0.9 + 0.2*0.9 = 0.5 + 0.27 + 0.18 = 0.95
    medium = MediumGrader()
    assert medium.score(mock_stats) == pytest.approx(0.95)
