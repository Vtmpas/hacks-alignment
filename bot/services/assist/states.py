from aiogram.fsm.state import State, StatesGroup


class AssistUserStates(StatesGroup):
    free: State = State()
    busy: State = State()
