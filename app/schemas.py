from typing import List

from pydantic import BaseModel


class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str


class RequestModel(BaseModel):
    query: str


class ResponseModel(BaseModel):
    text: str
