import os
import subprocess
from contextlib import contextmanager

import torch


@contextmanager
def configure_environment():
    """
    Context manager to set environment variables and configure torch backends.
    """
    # Disable memory-efficient and flash SDP (Sparse Dot Product)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    # Set environment variable to avoid deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        yield
    finally:
        # Optionally reset the environment and settings if needed
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Run the original command
if __name__ == "__main__":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    launch_code = "poetry run python -m turbo_alignment train_sft --experiment_settings_path training/sft.json"
    subprocess.run(launch_code, shell=True)
