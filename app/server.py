# server.py
import json

import litserve as ls
from fastapi import HTTPException
from huggingface_hub import snapshot_download
from pydantic import ValidationError as PydanticValidationError
from schemas import PredictOutputModel, RequestModel, ResponseModel, ValidationError
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
            lora_path=snapshot_download(
                repo_id="Vtmpas/hack-ada-lora",
                token="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",
                revision="master",
            ),
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
            lora_request=self.lora,
        )

        output = response[0].outputs[0].text

        try:
            return PredictOutputModel(**json.loads(output))

        except PydanticValidationError as error:
            errors = error.errors()
            raise HTTPException(
                422,
                detail=str(
                    ValidationError(
                        loc=[error_["loc"] for error_ in errors] if errors else [],
                        msg=errors[0]["msg"] if errors else "Unknown Validation error",
                        type=errors[0]["type"] if errors else "PydanticValidationError",
                    )
                ),
            ) from None

        except json.decoder.JSONDecodeError as error:
            raise HTTPException(
                422,
                detail=str(
                    ValidationError(
                        loc=[str(error.pos)],
                        msg=f"{error.msg} as pos {error.pos}. Got: {output}",
                        type="JSONDecodeError",
                    )
                ),
            ) from None

    def encode_response(self, output: PredictOutputModel, **kwargs) -> ResponseModel:
        return ResponseModel(text=str(output))


if __name__ == "__main__":
    ls.LitServer(
        lit_api=SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        api_path="/assist",
        stream=False,
        timeout=300,
    ).run(port=8000)
