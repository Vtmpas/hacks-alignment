import json

def replace_role_in_jsonl(file_path, old_role, new_role):
    # Read the original JSONL file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the role in each line
    updated_lines = []
    for line in lines:
        record = json.loads(line)
        for message in record.get('messages', []):
            if message.get('role') == old_role:
                message['role'] = new_role
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.write('\n'.join(updated_lines))

# Replace 'function response' with 'user' in the specified JSONL files
if __name__ == '__main__':
    replace_role_in_jsonl('/Users/matvejsaprykin/Desktop/hacks-alignment/training/train_results.jsonl', 'function response', 'user')
    replace_role_in_jsonl('/Users/matvejsaprykin/Desktop/hacks-alignment/training/test_results.jsonl', 'function response', 'user')