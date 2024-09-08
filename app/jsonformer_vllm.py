import json
from typing import Any, Dict, List, Union

import numpy as np
from jsonformer.logits_processors import (
    OutputNumbersTokens,
)
from termcolor import cprint
from vllm import LLM, SamplingParams

GENERATION_MARKER = "|GENERATION|"

json_schema = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "reasoning": {"type": "string"},
                "plan": {"type": "string"},
                "criticism": {"type": "string"},
                "speak": {"type": "string"},
            },
        },
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object", "properties": {"arg_name": {"type": "string"}}},
            },
        },
    },
}


class JsonformerVLLM:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        llm: LLM,
        tokenizer: Any,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)

        response = self.llm.generate(
            prompt, SamplingParams(max_tokens=self.max_number_tokens, temperature=temperature or self.temperature)
        )

        response_text = response[0].outputs[0].text
        response_text = response_text.strip().rstrip(".")
        self.debug("[generate_number]", response_text)
        try:
            return float(response_text)
        except ValueError:
            if iterations > 3:
                return -1
            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations + 1)

    def generate_boolean(self) -> bool:
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        response = self.llm.generate(prompt)
        logits = np.array(response[0].outputs[0].token_ids).ravel()

        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        true_match, mtv = (
            np.argmax(np.where(logits == true_token_id, 1, 0)),
            np.max(np.where(logits == true_token_id, 1, 0)),
        )
        false_match, mfv = (
            np.argmax(np.where(logits == false_token_id, 1, 0)),
            np.max(np.where(logits == false_token_id, 1, 0)),
        )

        result = true_match < false_match if mtv != 0 or mfv != 0 else False
        self.debug("[generate_boolean]", str(result))

        return result

    def generate_string(self, temperature=None) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)

        print(self.tokenizer.encode(prompt))

        response = self.llm.generate(
            prompt, SamplingParams(max_tokens=self.max_number_tokens, temperature=temperature or self.temperature)
        )

        response_text = response[0].outputs[0].text

        self.debug("[generate_string]", "|" + response_text + "|")

        if response_text.count('"') < 1:
            return response_text

        return response_text.split('"')[0].strip()

    def generate_object(self, properties: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        prompt = self.get_prompt()
        for _ in range(self.max_array_length):
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            obj.pop()
            response = self.llm.generate(prompt)
            logits = response[0].outputs[0].token_ids

            top_indices = logits[:30]

            found_comma = False
            found_close_bracket = False

            for token_id in top_indices:
                decoded_token = self.tokenizer.decode([token_id])
                if "," in decoded_token:
                    found_comma = True
                    break
                if "]" in decoded_token:
                    found_close_bracket = True
                    break

            if found_close_bracket or not found_comma:
                break

        return obj

    def get_prompt(self):
        template = """{prompt}\nформатируй ответ в следующем JSON формате:\n{schema}\nРезультат: {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(self.json_schema["properties"], self.value)
        return generated_data
