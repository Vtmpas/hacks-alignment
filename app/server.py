# server.py
from typing import Union

import litserve as ls

from app.schemas import ResponseModel, ValidationError


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.llm = lambda x: ResponseModel(text="dummy")

    def decode_request(self, request, **kwargs):
        return request["input"]

    def predict(self, x, **kwargs) -> Union[ResponseModel, ValidationError]:
        return self.llm(x)

    def encode_response(self, output, **kwargs):
        return {"output": output}


if __name__ == "__main__":
    ls.LitServer(
        lit_api=SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        api_path="/predict",
        stream=False,
    ).run(port=8000)
