import argparse
import os
import subprocess
from contextlib import contextmanager

import torch


@contextmanager
def configure_environment():
    """
    Context manager to set environment variables and configure torch backends.

    This context manager disables memory-efficient and flash sparse dot product (SDP)
    settings for torch CUDA backends and sets an environment variable to avoid
    parallelism warnings from tokenizers. It yields control to the code block within
    the context. After exiting the block, it restores the original settings.

    Yields:
        None
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


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()


# Run the original command
if __name__ == "__main__":
    args = parse_arguments()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    launch_code = f"poetry run python -m turbo_alignment train_sft --experiment_settings_path {args.config}"
    subprocess.run(launch_code, shell=True)  # noqa: S602
