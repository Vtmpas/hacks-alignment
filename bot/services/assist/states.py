from aiogram.fsm.state import State, StatesGroup


class AssistUserStates(StatesGroup):
    """
    Defines the different states for an assist user in the state machine.

    Inherits from:
        StatesGroup: Base class for defining a group of states in the state machine.

    Attributes:
        free (State): The state representing when the user is free.
        busy (State): The state representing when the user is busy.
    """

    free: State = State()
    busy: State = State()
