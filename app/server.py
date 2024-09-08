# server.py

import json

import litserve as ls
from fastapi import HTTPException
from huggingface_hub import snapshot_download
from jsonformer_vllm import JsonformerVLLM, json_schema
from schemas import RequestModel, ResponseModel, ValidationError
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class SimpleLitAPI(ls.LitAPI):
    sampling_params: SamplingParams
    lora: LoRARequest
    llm: LLM
    tokenizer: AutoTokenizer

    def setup(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("AnatoliiPotapov/T-lite-instruct-0.1")
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
            stop=["<|eot_id|>"],
        )
        self.llm = LLM(
            model="AnatoliiPotapov/T-lite-instruct-0.1",
            enable_lora=True,
            dtype="half",
            tensor_parallel_size=4,
        )

    def decode_request(self, request: RequestModel, **kwargs) -> str:
        return request.query

    def predict(self, prompt: str, **kwargs) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [   {"role": "sysytem", "content": 'Отвечай в основном на русском. Выбранные аргументы "args" тоже должны быть на русском языке'},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True
        )
        response = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
            lora_request=self.lora,
        )

        output = response[0].outputs[0].text

        try:
            return json.dumps(json.loads(output), ensure_ascii=False, indent=2)

        except json.decoder.JSONDecodeError as error:
            print(error, "forcing JsonFormer")
            try:
                jsonformer = JsonformerVLLM(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    json_schema=json_schema,
                    prompt=prompt,
                    debug=False,  # Enable debug mode to see detailed output
                )
                generated_data = jsonformer()
                return json.dumps(generated_data, ensure_ascii=False, indent=2)

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

        except Exception as error:
            raise HTTPException(
                500,
                detail=f"Uncaught exception: {error}. Got: {output}",
            ) from None

    def encode_response(self, output: str, **kwargs) -> ResponseModel:
        return ResponseModel(text=output)


if __name__ == "__main__":
    ls.LitServer(
        lit_api=SimpleLitAPI(),
        accelerator="auto",
        max_batch_size=1,
        api_path="/assist",
        stream=False,
        timeout=300,
    ).run(port=8000)
