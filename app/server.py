# server.py
import time

import litserve as ls
from schemas import DummyModelOutput, PredictOutputModel, RequestModel, ResponseModel


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = lambda x: DummyModelOutput

    def decode_request(self, request: RequestModel, **kwargs) -> str:
        return request.query

    def predict(self, x: str, **kwargs) -> PredictOutputModel:
        time.sleep(1)
        return self.model(x)

    def encode_response(self, output: PredictOutputModel, **kwargs) -> ResponseModel:
        return ResponseModel(text=str(output))


if __name__ == "__main__":
    ls.LitServer(
        lit_api=SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        api_path="/predict",
        stream=False,
    ).run(port=8000)
