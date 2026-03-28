"""Typed Action model for the Urban Delivery Environment."""

from enum import IntEnum

from pydantic import BaseModel, Field


class ActionType(IntEnum):
    """Available actions for the delivery agent."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    DELIVER = 4
    REFUEL = 5


ACTION_NAMES = {
    ActionType.MOVE_UP: "Move Up",
    ActionType.MOVE_DOWN: "Move Down",
    ActionType.MOVE_LEFT: "Move Left",
    ActionType.MOVE_RIGHT: "Move Right",
    ActionType.DELIVER: "Deliver Package",
    ActionType.REFUEL: "Refuel",
}


class DeliveryAction(BaseModel):
    """Action to be taken by the delivery agent.

    Actions:
        0 = Move Up (row - 1)
        1 = Move Down (row + 1)
        2 = Move Left (col - 1)
        3 = Move Right (col + 1)
        4 = Deliver Package (at current position)
        5 = Refuel (at fuel station)
    """
    action: int = Field(
        ge=0,
        le=5,
        description="Action index: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=DELIVER, 5=REFUEL",
    )

    @property
    def action_type(self) -> ActionType:
        return ActionType(self.action)

    @property
    def name(self) -> str:
        return ACTION_NAMES[self.action_type]
