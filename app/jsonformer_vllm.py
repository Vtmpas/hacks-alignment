import json
from typing import Any, Dict, List, Union

import numpy as np
from jsonformer.logits_processors import OutputNumbersTokens
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
    """
    Class for generating structured data in JSON format using a large language model (LLM).
    It integrates with a pre-defined JSON schema to generate specific fields like numbers,
    strings, booleans, objects, and arrays based on the schema structure.

    Attributes:
        llm (LLM): The language model instance for text generation.
        tokenizer (Any): The tokenizer used for encoding and decoding text with the LLM.
        json_schema (Dict[str, Any]): JSON schema defining the structure of the output.
        prompt (str): The initial text prompt for guiding the model's response.
        debug_on (bool): If `True`, enables debug prints during the generation process.
        max_array_length (int): Maximum number of elements in generated arrays.
        max_number_tokens (int): Maximum number of tokens for number generation.
        temperature (float): The temperature parameter for controlling randomness in generation.
        max_string_token_length (int): Maximum token length for generating strings.
        generation_marker (str): Marker used to indicate in-progress generation within the prompt.
    """

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
        """
        Initializes the JsonformerVLLM class with required model, tokenizer, schema,
        and generation parameters.

        Args:
            llm (LLM): The language model used for generation.
            tokenizer (Any): Tokenizer for encoding and decoding prompts.
            json_schema (Dict[str, Any]): The JSON schema that defines the expected
                output structure.
            prompt (str): Initial prompt text for generation.
            debug (bool, optional): Enables debugging information. Defaults to False.
            max_array_length (int, optional): Maximum elements for arrays. Defaults to 10.
            max_number_tokens (int, optional): Max tokens for numbers. Defaults to 6.
            temperature (float, optional): Randomness control for generation. Defaults to 1.0.
            max_string_token_length (int, optional): Max tokens for strings. Defaults to 10.
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt
        self.debug_on = debug
        self.max_array_length = max_array_length
        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length
        self.generation_marker = "|GENERATION|"
        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        """
        Prints debugging information to the console.

        Args:
            caller (str): The name of the calling function.
            value (str): The value to print for debugging purposes.
            is_prompt (bool, optional): If True, color the output differently to indicate
                it's a prompt. Defaults to False.
        """
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0) -> float:
        """
        Generates a number based on the provided prompt and schema. Retries up to three
        times if the generation fails to produce a valid number.

        Args:
            temperature (Union[float, None], optional): Temperature to control randomness
                in generation. If not provided, uses the default temperature.
            iterations (int, optional): The number of retries attempted. Defaults to 0.

        Returns:
            float: The generated number or -1 if the generation fails after 3 attempts.
        """
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)

        response = self.llm.generate(
            prompt, SamplingParams(max_tokens=self.max_number_tokens, temperature=temperature or self.temperature)
        )

        response_text = response[0].outputs[0].text.strip().rstrip(".")
        self.debug("[generate_number]", response_text)

        try:
            return float(response_text)
        except ValueError:
            if iterations > 3:
                return -1
            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations + 1)

    def generate_boolean(self) -> bool:
        """
        Generates a boolean value (True/False) based on the schema and model response.

        Returns:
            bool: The generated boolean value.
        """
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        response = self.llm.generate(prompt)
        logits = np.array(response[0].outputs[0].token_ids).ravel()

        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        true_match = np.argmax(np.where(logits == true_token_id, 1, 0))
        false_match = np.argmax(np.where(logits == false_token_id, 1, 0))

        result = true_match < false_match
        self.debug("[generate_boolean]", str(result))

        return result

    def generate_string(self, temperature: Union[float, None] = None) -> str:
        """
        Generates a string value based on the schema and model response.

        Args:
            temperature (Union[float, None], optional): Temperature for randomness. Defaults to None.

        Returns:
            str: The generated string value.
        """
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)

        response = self.llm.generate(
            prompt, SamplingParams(max_tokens=self.max_string_token_length, temperature=temperature or self.temperature)
        )

        response_text = response[0].outputs[0].text
        self.debug("[generate_string]", "|" + response_text + "|")

        if '"' not in response_text:
            return response_text

        return response_text.split('"')[0].strip()

    def generate_object(self, properties: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an object with key-value pairs based on the schema properties.

        Args:
            properties (Dict[str, Any]): The properties of the object as defined in the JSON schema.
            obj (Dict[str, Any]): The object being populated with generated values.

        Returns:
            Dict[str, Any]: The generated object.
        """
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
        """
        Generates a value based on the schema type (number, boolean, string, array, object).

        Args:
            schema (Dict[str, Any]): The schema defining the type of the value to generate.
            obj (Union[Dict[str, Any], List[Any]]): The object or array to append the generated value to.
            key (Union[str, None], optional): The key of the value to generate (used for objects). Defaults to None.

        Returns:
            Any: The generated value.
        """
        schema_type = schema["type"]

        if schema_type == "number":
            obj[key] = self.generation_marker if key else obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            obj[key] = self.generation_marker if key else obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            obj[key] = self.generation_marker if key else obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            obj[key] = new_obj if key else obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        """
        Generates an array based on the schema, filling it with values according to the specified
        item type.

        Args:
            item_schema (Dict[str, Any]): The schema for the individual items in the array.
            obj (Dict[str, Any]): The array to be populated with generated items.

        Returns:
            list: The generated array.
        """
        prompt = self.get_prompt()

        for _ in range(self.max_array_length):
            element = self.generate_value(item_schema, obj)
            obj[-1] = element
            obj.append(self.generation_marker)
            obj.pop()

            response = self.llm.generate(prompt)
            logits = response[0].outputs[0].token_ids
            top_indices = logits[:30]

            if any(self.tokenizer.decode([token_id]) in [",", "]"] for token_id in top_indices):
                break

        return obj

    def get_prompt(self) -> str:
        """
        Generates a prompt for the language model based on the current state of the generated value.

        Returns:
            str: The formatted prompt.
        """
        template = """{prompt}\nформатируй ответ в следующем JSON формате:\n{schema}\nРезультат: {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')

        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        return template.format(prompt=self.prompt, schema=json.dumps(self.json_schema), progress=progress)

    def __call__(self) -> Dict[str, Any]:
        """
        Executes the generation process, starting with an empty object and
        filling it according to the schema.

        Returns:
            Dict[str, Any]: The fully generated object as per the schema.
        """
        self.value = {}
        return self.generate_object(self.json_schema["properties"], self.value)
