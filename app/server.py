# server.py
from typing import cast

import litserve as ls
from huggingface_hub import snapshot_download
from schemas import PredictOutputModel, RequestModel, ResponseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class SimpleLitAPI(ls.LitAPI):
    sampling_params: SamplingParams
    lora: LoRARequest
    llm: LLM

    def setup(self, device):
        self.lora = LoRARequest(
            lora_name="sft",
            lora_int_id=1,
            lora_path=snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test"),
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
        )
        self.llm = LLM(model="AnatoliiPotapov/T-lite-instruct-0.1", enable_lora=True)

    def decode_request(self, request: RequestModel, **kwargs) -> str:
        return request.query

    def predict(self, prompt: str, **kwargs) -> PredictOutputModel:
        response = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
        )

        return cast(PredictOutputModel, response)

    def encode_response(self, output: PredictOutputModel, **kwargs) -> ResponseModel:
        return ResponseModel(text=str(output))


if __name__ == "__main__":
    ls.LitServer(
        lit_api=SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        api_path="/assist",
        stream=False,
    ).run(port=8000)
