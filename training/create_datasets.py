import concurrent.futures
import json
import re
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load dataset
ds = load_dataset("glaiveai/glaive-function-calling-v2")["train"].train_test_split(test_size=0.996)["train"]
print(len(ds))


def extract_function_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extracts function calls from a given text based on specific patterns and converts them into a list of dictionaries.

    Args:
        text (str): The input text containing function calls in a specific format.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a function call with its name and arguments.
    """
    pattern = re.compile(r"<functioncall>\s*({.*?})\s*<\|endoftext\|>", re.DOTALL)
    matches = pattern.findall(text)
    function_calls = []

    for match in matches:
        try:
            function_call = json.loads(match.replace("'", ""))
            function_name = function_call.get("name")
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments.replace("'", ""))
            function_calls.append({function_name: arguments})
        except json.JSONDecodeError:
            pass
    return function_calls


def post_process_dialogue(dialogue: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-processes a dialogue by updating commands with arguments based on extracted queries.

    Args:
        dialogue (List[Dict[str, Any]]): The original dialogue to be processed.
        queries (List[Dict[str, Any]]): The list of extracted function calls and their arguments.

    Returns:
        List[Dict[str, Any]]: The processed dialogue with updated command arguments.
    """
    processed_dialogue = []
    query_index = 0
    for message in dialogue:
        if message["role"] == "ASSISTANT" and isinstance(message["content"], dict):
            command = message["content"].get("command", {})
            command_name = command.get("name")
            if query_index < len(queries) and command_name in queries[query_index]:
                message["content"]["command"]["args"] = queries[query_index][command_name]
                query_index += 1
        processed_dialogue.append(message)
    return processed_dialogue


class Thought(BaseModel):
    """
    Model for representing a thought with various attributes.

    Attributes:
        text (str): Summary of the thought in Russian.
        reasoning (str): Explanation of reasoning in Russian.
        plan (str): Plan of action in Russian.
        criticism (str): Self-criticism in Russian.
        speak (str): Summary to communicate to the user in Russian.
    """

    text: str = Field(..., description="Summary of thought. In Russian")
    reasoning: str = Field(..., description="Explanation of reasoning. In Russian")
    plan: str = Field(..., description="Plan of action - short bulleted list in Russian")
    criticism: str = Field(..., description="Self-criticism. In Russian")
    speak: str = Field(..., description="Summary to communicate to the user. In Russian")


class Command(BaseModel):
    """
    Model for representing a command with a name and optional arguments.

    Attributes:
        name (str): Name of the command to execute.
        args (Optional[Dict[str, Any]]): Arguments for the command.
    """

    name: str = Field(..., description="Name of the command to execute. <functioncall>")
    args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the command.")


class FormattedResponse(BaseModel):
    """
    Model for representing a formatted response consisting of thoughts and a command.

    Attributes:
        thoughts (Thought): The thought process related to the response.
        command (Command): The command to be executed as part of the response.
    """

    thoughts: Thought
    command: Command


class FullResponse(BaseModel):
    """
    Model for representing a full response in a dialogue, including role and content.

    Attributes:
        role (str): Role of the message sender (USER, ASSISTANT, or FUNCTION RESPONSE).
        content (Union[str, FormattedResponse]): Content of the message, which can be a string or a FormattedResponse.
    """

    role: str = Field(..., description="USER or ASSISTANT or FUNCTION RESPONSE")
    content: Union[str, FormattedResponse] = Field(
        ..., description="Content of the message, either a string for user or FormattedResponse for assistant."
    )


class DialogueResponse(BaseModel):
    """
    Model for representing a dialogue consisting of multiple full responses.

    Attributes:
        dialogue (List[FullResponse]): The full dialogue as a list of responses.
    """

    dialogue: List[FullResponse] = Field(..., description="Full dialogue according to schemas")


client = OpenAI(api_key="")


def parse_to_list_of_dicts_general(text: str) -> List[Dict[str, Any]]:
    """
    Parses a given text into a list of dictionaries, handling common JSON formatting issues.

    Args:
        text (str): The input text to be parsed.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries parsed from the text.
    """
    text = re.sub(r"(?<!\\)'(?!s\b)(?=[^']*(?:'[^']*'[^']*)*$)", '"', text)
    text = re.sub(r"\\(')", r"\1", text)
    text = re.sub(
        r'(?<!\\)(")((?:.(?!(?<!\\)"))*.)(")', lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3), text
    )

    if not text.strip().startswith("["):
        corrected_text = re.sub(r"\}\s*\{", "},{", text)
        corrected_text = f"[{corrected_text}]"
    else:
        corrected_text = text

    try:
        parsed_dicts = json.loads(corrected_text)
        if not isinstance(parsed_dicts, list):
            parsed_dicts = [parsed_dicts]
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic text: {corrected_text}")
        try:
            fixed_text = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', corrected_text)
            parsed_dicts = json.loads(fixed_text)
            if not isinstance(parsed_dicts, list):
                parsed_dicts = [parsed_dicts]
        except json.JSONDecodeError:
            print("Failed to fix JSON. Returning empty list.")
            parsed_dicts = []

    return parsed_dicts


def convert_properties_to_args(parameters: Dict[str, Any]) -> str:
    """
    Converts a dictionary of parameters into a string of arguments.

    Args:
        parameters (Dict[str, Any]): The dictionary containing parameter descriptions.

    Returns:
        str: A string representation of the arguments, based on the parameter descriptions.
    """
    args = {}
    if not isinstance(parameters, dict):
        print(f"Warning: parameters is not a dictionary. Type: {type(parameters)}")
        return str(args)

    properties = parameters.get("properties", {})
    if not properties:
        return "No arguments required"

    for key, value in properties.items():
        if isinstance(value, dict) and "description" in value:
            description = value["description"].split(", e.g. ")[0].split(", ")[0].capitalize()
            args[key] = f"<{description}>"
        else:
            args[key] = "<Unknown>"

    return str(args).replace("{", "").replace("}", "")


def translate_system(prompt: str) -> str:
    """
    Translates a system prompt into Russian using an OpenAI model.

    Args:
        prompt (str): The prompt text to be translated.

    Returns:
        str: The translated prompt text in Russian.
    """
    splitted_prompt = prompt.split(" -\n")
    if len(splitted_prompt) == 1:
        return "Ты -- полезный помощник без дополнительных функций \n"

    json_thing = parse_to_list_of_dicts_general(splitted_prompt[-1])

    base_prompt = "Ты -- полезный помощник со следующими функциями:\n "

    for func in json_thing:
        args = convert_properties_to_args(func.get("parameters", {}))
        str_to_add = f"{func['description']}: '{func['name']}', args: {args}"
        base_prompt += str_to_add + "\n"

    messages = [
        {
            "role": "system",
            "content": "ANSWER IN RUSSIAN. You are a helpful translator. Returns only the translated text in Russian. Don't translate words in 'words'. ",
        },
        {"role": "user", "content": base_prompt},
    ]

    completion = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=messages, temperature=0)
    translated_text = completion.choices[0].message.content

    return translated_text


def prepare_sample(example: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """
    Prepares a sample by processing the chat and system prompts and extracting function calls.

    Args:
        example (Dict[str, Any]): The sample example containing chat and system prompts.

    Returns:
        Tuple[Optional[List[Dict[str, Any]]], bool]: A tuple where the first element is the processed dialogue or None, and the second element is a boolean indicating success.
    """
    messages = [
        {
            "role": "system",
            "content": "ANSWER IN RUSSIAN. You are a helpful ASSISTANT that helps with converting answers and translations from English to Russian. Always answer in Russian. If there is no useful command then answer 'NoFunction'",
        },
        {"role": "user", "content": example["chat"]},
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", messages=messages, response_format=DialogueResponse, temperature=0
        )
        dialogue = completion.choices[0].message.parsed

        system_ru = translate_system(example["system"])

        extracted_function_calls = extract_function_calls(example["chat"])
        processed_dialogue = post_process_dialogue(dialogue.model_dump()["dialogue"], extracted_function_calls)
        processed_dialogue[0]["content"] = system_ru + "\n" + processed_dialogue[0]["content"]

        return processed_dialogue, True
    except Exception as e:
        print(f"Error processing sample: {e}")
        print("SYSTEM:", example["system"])
        print("CHAT:", example["chat"])
        print("Extracted function calls:", extracted_function_calls)
        return None, False


def process_dataset(dataset: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Processes a dataset by preparing samples and handling results with concurrency.

    Args:
        dataset (List[Dict[str, Any]]): The list of dataset examples to process.

    Returns:
        List[List[Dict[str, Any]]]: A list of processed results for each dataset example.
    """
    successful_samples = 0
    failed_samples = 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(prepare_sample, example) for example in dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc="Processing samples"):
            try:
                result, success = future.result()
                if success:
                    results.append(result)
                    successful_samples += 1
                else:
                    failed_samples += 1
            except Exception as e:
                print(f"Error processing sample: {e}")
                failed_samples += 1

    print(f"Successful samples: {successful_samples}")
    print(f"Failed samples: {failed_samples}")
    return results


def create_jsonl_from_processed_results(
    processed_results: List[List[Dict[str, Any]]], output_file: str = "output.jsonl"
) -> None:
    """
    Creates a JSONL file from processed results.

    Args:
        processed_results (List[List[Dict[str, Any]]]): The processed results to write to the JSONL file.
        output_file (str): The name of the output JSONL file. Defaults to "output.jsonl".

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, result in enumerate(processed_results):
            json_object = {"id": idx, "source": "example", "messages": []}

            for message in result:
                role = message["role"].lower()
                content = message["content"]

                json_object["messages"].append(
                    {
                        "role": role,
                        "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                    }
                )

            f.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    print(f"JSONL file '{output_file}' has been created successfully.")


def process_and_split_results(
    processed_results: List[List[Dict[str, Any]]],
    train_file: str = "train.jsonl",
    test_file: str = "test.jsonl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Processes and splits results into training and testing sets, and saves them as JSONL files.

    Args:
        processed_results (List[List[Dict[str, Any]]]): The processed results to split and save.
        train_file (str): The file path for the training dataset. Defaults to "train.jsonl".
        test_file (str): The file path for the testing dataset. Defaults to "test.jsonl".
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int): The seed used by the random number generator. Defaults to 42.

    Returns:
        None
    """
    train_data, test_data = train_test_split(processed_results, test_size=test_size, random_state=random_state)
    create_jsonl_from_processed_results(train_data, train_file)
    create_jsonl_from_processed_results(test_data, test_file)


if __name__ == "__main__":
    processed_results = process_dataset(ds)
    process_and_split_results(processed_results, train_file="train_results.jsonl", test_file="test_results.jsonl")
