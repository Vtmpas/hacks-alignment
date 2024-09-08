from typing import List

from pydantic import BaseModel


class ValidationError(BaseModel):
    """
    Represents an error that occurs during validation.

    Attributes:
        loc (List[str]): A list of locations (fields) where the validation error occurred.
        msg (str): A descriptive error message explaining the issue.
        type (str): The type or category of the validation error.
    """

    loc: List[str]
    msg: str
    type: str


class RequestModel(BaseModel):
    """
    Represents a request model containing a user's query.

    Attributes:
        query (str): The input query provided by the user.
    """

    query: str


class ResponseModel(BaseModel):
    """
    Represents a response model containing the generated text response.

    Attributes:
        text (str): The generated text or answer to the user's query.
    """

    text: str
