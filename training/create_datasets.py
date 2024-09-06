import json
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from datasets import load_dataset
import concurrent.futures

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load dataset
ds = load_dataset("glaiveai/glaive-function-calling-v2")['train'].train_test_split(test_size=0.996)['train']
print(len(ds))

# Function to extract queries from the text
def extract_function_calls(text):
    pattern = re.compile(r'<functioncall>\s*({.*?})\s*<\|endoftext\|>', re.DOTALL)
    matches = pattern.findall(text)
    function_calls = []

    for match in matches:
        try:
            function_call = json.loads(match.replace("\'", ''))
            function_name = function_call.get('name')
            arguments = function_call.get('arguments')
            if isinstance(arguments, str):
                arguments = json.loads(arguments.replace("\'", ''))
            function_calls.append({function_name: arguments})
        except json.JSONDecodeError:
            pass
    return function_calls

def post_process_dialogue(dialogue, queries):
    processed_dialogue = []
    query_index = 0
    for message in dialogue:
        if message['role'] == 'ASSISTANT' and isinstance(message['content'], dict):
            command = message['content'].get('command', {})
            command_name = command.get('name')
            if query_index < len(queries) and command_name in queries[query_index]:
                message['content']['command']['args'] = queries[query_index][command_name]
                query_index += 1
        processed_dialogue.append(message)
    return processed_dialogue

# Define models
class Thought(BaseModel):
    text: str = Field(..., description="Summary of thought. In Russian")
    reasoning: str = Field(..., description="Explanation of reasoning. In Russian")
    plan: str = Field(..., description="Plan of action - short bulleted\n- list that conveys\n- long-term plan. In Russian")
    criticism: str = Field(..., description="Self-criticism. In Russian")
    speak: str = Field(..., description="Summary to communicate to the user. In Russian")

class Command(BaseModel):
    name: str = Field(..., description="Name of the command to execute. <functioncall>")
    args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the command.")

class FormattedResponse(BaseModel):
    thoughts: Thought
    command: Command

class FullResponse(BaseModel):
    role: str = Field(..., description="USER or ASSISTANT or FUNCTION RESPONSE")
    content: Union[str, FormattedResponse] = Field(..., description="If user translated in Russian str. If assistant then FormattedResponse translated in russian. Always use FormattedResponse for assistant even there is no function calling")

class DialogueResponse(BaseModel):
    dialogue: List[FullResponse] = Field(..., description="Full dialogue according to schemas")

# Initialize OpenAI client
client = OpenAI(api_key='sk-proj-7k9awwW54lHsonHXDRNQT3BlbkFJbMuWCigNX0GyUhszMT8r')


def parse_to_list_of_dicts_general(text):
    # Step 1: Replace single quotes with double quotes, except those in words
    text = re.sub(r"(?<!\\)'(?!s\b)(?=[^']*(?:'[^']*'[^']*)*$)", '"', text)

    # Step 2: Remove backslashes before single quotes
    text = re.sub(r"\\(')", r"\1", text)

    # Step 3: Escape double quotes within string values
    text = re.sub(r'(?<!\\)(")((?:.(?!(?<!\\)"))*.)(")',
                  lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3), text)

    # Step 4: Add a comma between each JSON object if missing and wrap in a list
    if not text.strip().startswith('['):
        corrected_text = re.sub(r'\}\s*\{', '},{', text)
        corrected_text = f'[{corrected_text}]'
    else:
        corrected_text = text

    # Step 5: Parse the corrected JSON text
    try:
        parsed_dicts = json.loads(corrected_text)
        if not isinstance(parsed_dicts, list):
            parsed_dicts = [parsed_dicts]
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic text: {corrected_text}")
        # Attempt to fix common issues
        try:
            fixed_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', corrected_text)
            parsed_dicts = json.loads(fixed_text)
            if not isinstance(parsed_dicts, list):
                parsed_dicts = [parsed_dicts]
        except json.JSONDecodeError:
            print("Failed to fix JSON. Returning empty list.")
            parsed_dicts = []

    return parsed_dicts


def convert_properties_to_args(parameters):
    args = {}
    if not isinstance(parameters, dict):
        print(f"Warning: parameters is not a dictionary. Type: {type(parameters)}")
        return str(args)

    properties = parameters.get('properties', {})
    if not properties:
        return "No arguments required"

    for key, value in properties.items():
        if isinstance(value, dict) and 'description' in value:
            description = value['description'].split(', e.g. ')[0].split(', ')[0].capitalize()
            args[key] = f"<{description}>"
        else:
            args[key] = "<Unknown>"

    return str(args).replace("{", '').replace("}", '')


def translate_system(prompt):
    splitted_prompt = prompt.split(' -\n')
    if len(splitted_prompt) == 1:
        return 'Ты -- полезный помощник без дополнительных функций \n'

    json_thing = parse_to_list_of_dicts_general(splitted_prompt[-1])

    base_prompt = 'Ты -- полезный помощник со следующими функциями:\n '

    for func in json_thing:
        args = convert_properties_to_args(func.get('parameters', {}))
        str_to_add = f"{func['description']}: '{func['name']}', args: {args}"
        base_prompt += str_to_add + "\n"

    messages = [
        {"role": "system",
         "content": "ANSWER IN RUSSIAN. You are a helpful translator. Returns only the translated text in Russian. Don't translate words in 'words'. "},
        {"role": "user", "content": base_prompt},
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0
    )
    translated_text = completion.choices[0].message.content

    return translated_text



def prepare_sample(example):
    messages = [
        {"role": "system", "content": "ANSWER IN RUSSIAN. You are a helpful ASSISTANT that helps with converting answers and translations from English to Russian. Always answer in Russian. If there is no useful command then answer 'NoFunction'"},
        {"role": "user", "content": example['chat']}
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=DialogueResponse,
            temperature=0
        )
        dialogue = completion.choices[0].message.parsed

        system_ru = translate_system(example['system'])

        extracted_function_calls = extract_function_calls(example['chat'])
        processed_dialogue = post_process_dialogue(dialogue.model_dump()['dialogue'], extracted_function_calls)
        processed_dialogue[0]['content'] = system_ru +'\n' + processed_dialogue[0]['content']

        return processed_dialogue, True
    except Exception as e:
        print(f"Error processing sample: {e}")
        print("SYSTEM:", example['system'])
        print("CHAT:", example['chat'])
        print("Extracted function calls:", extracted_function_calls)
        return None, False


def process_dataset(dataset):
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


def create_jsonl_from_processed_results(processed_results, output_file='output.jsonl'):
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate over each result in processed_results
        for idx, result in enumerate(processed_results):
            # Create a dictionary with the desired schema
            json_object = {
                "id": idx,
                "source": "example",
                "messages": []
            }

            # Populate messages with role and content from the processed result
            for message in result:
                role = message['role'].lower()  # Convert role to lowercase
                content = message['content']

                # Add the message to the messages list
                json_object["messages"].append({
                    "role": role,
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                })

            # Write the JSON object to the file as a JSONL line
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')

    print(f"JSONL file '{output_file}' has been created successfully.")

def process_and_split_results(processed_results, train_file='train.jsonl', test_file='test.jsonl', test_size=0.2,
                              random_state=42):
    """
    Process processed_results, split them into train and test sets, and save them as JSONL files.
    """
    # Perform train-test split
    train_data, test_data = train_test_split(processed_results, test_size=test_size, random_state=random_state)

    # Create JSONL files for training and testing datasets
    create_jsonl_from_processed_results(train_data, train_file)
    create_jsonl_from_processed_results(test_data, test_file)

if __name__ == '__main__':
    processed_results = process_dataset(ds)
    process_and_split_results(processed_results, train_file='train_results.jsonl', test_file='test_results.jsonl')
