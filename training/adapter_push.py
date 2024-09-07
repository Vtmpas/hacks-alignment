from huggingface_hub import HfApi, Repository
import os

# Full path to the adapter
adapter_path = os.path.expanduser("~/hacks-alignment/train_output/trainer/checkpoint-75")

# Define your repository name (replace 'your-username/my-private-repo' with your desired repo path)
repo_name = "vtmpas/hack-ada-lora"  # Replace with your repository path
adapter_repo_url = f"https://huggingface.co/{repo_name}"

# Step 1: Create a private repository on Hugging Face Hub
api = HfApi()
api.create_repo(repo_id=repo_name, exist_ok=True, private=True)  # Create a private repository

# Step 2: Initialize the Repository object to push the adapter
repo = Repository(local_dir=adapter_path, clone_from=adapter_repo_url, use_auth_token=True)

# Step 3: Push the adapter to the private repository
repo.push_to_hub(commit_message="Uploading adapter checkpoint-75")