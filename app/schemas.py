from typing import Dict, List

from pydantic import BaseModel


class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str


class RequestModel(BaseModel):
    query: str


class ResponseModel(BaseModel):
    text: str


class ThoughtsModel(BaseModel):
    text: str
    reasoning: str
    plan: str
    criticism: str
    speak: str


class CommandModel(BaseModel):
    name: str
    args: Dict[str, str]


class PredictOutputModel(BaseModel):
    thoughts: ThoughtsModel
    command: CommandModel


DummyModelOutput = PredictOutputModel(
    thoughts=ThoughtsModel(
        text="thought",
        reasoning="reasoning",
        plan="- short bulleted\n- list that conveys\n- long-term plan",
        criticism="constructive self-criticism",
        speak="thoughts summary to say to user",
    ),
    command=CommandModel(
        name="command name",
        args={"arg name": "value"},
    ),
)
