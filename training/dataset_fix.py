import json


def replace_role_in_jsonl(file_path: str, old_role: str, new_role: str) -> None:
    """
    Replaces occurrences of a specific role with a new role in a JSONL file.

    This function reads a JSONL (JSON Lines) file, replaces all instances of `old_role` with `new_role` in
    the "role" field of each message, and writes the updated records back to the same file.

    Args:
        file_path (str): The path to the JSONL file to be processed.
        old_role (str): The role to be replaced in the JSONL file.
        new_role (str): The new role that will replace the old role.

    Returns:
        None
    """
    # Read the original JSONL file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Replace the role in each line
    updated_lines = []
    for line in lines:
        record = json.loads(line)
        for message in record.get("messages", []):
            if message.get("role") == old_role:
                message["role"] = new_role
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    # Write the updated lines back to the file
    with open(file_path, "w") as file:
        file.write("\n".join(updated_lines))


# Replace 'function response' with 'user' in the specified JSONL files
if __name__ == "__main__":
    """
    Main entry point of the script. Executes the role replacement process for specified JSONL files.

    This script replaces 'function response' with 'user' in the JSONL files located at the specified paths.
    """
    replace_role_in_jsonl(
        "/Users/matvejsaprykin/Desktop/hacks-alignment/training/train_results.jsonl", "function response", "user"
    )
    replace_role_in_jsonl(
        "/Users/matvejsaprykin/Desktop/hacks-alignment/training/test_results.jsonl", "function response", "user"
    )
