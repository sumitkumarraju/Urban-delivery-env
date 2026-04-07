"""
Unit tests for the Graders.
Tests scoring bounds, explanations, and determinism.
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


def test_easy_grader_with_explanation():
    """Test that easy grader returns a valid explanation."""
    grader = EasyGrader()
    import random
    random.seed(42)
    actions = [random.randint(0, 3) for _ in range(50)]
    score, explanation = grader.grade_with_explanation(actions)
    assert 0.0 < score < 1.0
    assert "completion" in explanation
    assert "final_score" in explanation


def test_medium_grader_with_explanation():
    """Test that medium grader explains all 3 factors."""
    grader = MediumGrader()
    import random
    random.seed(123)
    actions = [random.randint(0, 3) for _ in range(100)]
    score, explanation = grader.grade_with_explanation(actions)
    assert 0.0 < score < 1.0
    assert "completion" in explanation
    assert "fuel_efficiency" in explanation
    assert "time_efficiency" in explanation


def test_hard_grader_with_explanation():
    """Test that hard grader explains all 5 factors."""
    grader = HardGrader()
    import random
    random.seed(456)
    actions = [random.randint(0, 5) for _ in range(200)]
    score, explanation = grader.grade_with_explanation(actions)
    assert 0.0 < score < 1.0
    assert "completion" in explanation
    assert "fuel_efficiency" in explanation
    assert "deadline_compliance" in explanation
    assert "priority_accuracy" in explanation
    assert "reward_normalized" in explanation


def test_grader_determinism():
    """Verify same actions always produce same score."""
    grader = HardGrader()
    import random
    random.seed(789)
    actions = [random.randint(0, 5) for _ in range(100)]
    score1 = grader.grade(actions)
    score2 = grader.grade(actions)
    assert score1 == score2, "Grader must be deterministic"


@pytest.mark.parametrize("GraderClass", [EasyGrader, MediumGrader, HardGrader])
def test_empty_actions_score_strictly_in_open_interval(GraderClass):
    """Zero actions → worst case score must still be strictly > 0 and < 1."""
    grader = GraderClass()
    score = grader.grade([])
    assert 0.0 < score < 1.0, f"{GraderClass.__name__} grade([]) = {score} not in (0, 1)"


@pytest.mark.parametrize("GraderClass", [EasyGrader, MediumGrader, HardGrader])
def test_explanation_final_score_matches_returned_score(GraderClass):
    """explanation['final_score'] must equal the returned mapped score."""
    grader = GraderClass()
    import random
    random.seed(42)
    actions = [random.randint(0, 5) for _ in range(50)]
    score, explanation = grader.grade_with_explanation(actions)
    assert explanation["final_score"] == score, (
        f"{GraderClass.__name__}: explanation final_score {explanation['final_score']} != returned {score}"
    )
    assert 0.0 < explanation["final_score"] < 1.0


@pytest.mark.parametrize("GraderClass", [EasyGrader, MediumGrader, HardGrader])
def test_score_never_exactly_zero_or_one(GraderClass):
    """Exhaustive check: scores from various action sequences never hit the boundaries."""
    grader = GraderClass()
    import random
    for seed in range(10):
        random.seed(seed)
        actions = [random.randint(0, 5) for _ in range(100)]
        score = grader.grade(actions)
        assert 0.0 < score < 1.0, f"seed={seed}: score={score}"
        score_ex, _ = grader.grade_with_explanation(actions)
        assert 0.0 < score_ex < 1.0, f"seed={seed}: score_with_explanation={score_ex}"
