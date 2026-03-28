"""Task configurations for the Urban Delivery Environment."""

from .easy import EASY_TASK
from .medium import MEDIUM_TASK
from .hard import HARD_TASK

ALL_TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

__all__ = ["EASY_TASK", "MEDIUM_TASK", "HARD_TASK", "ALL_TASKS"]
