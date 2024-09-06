import torch
import subprocess
from contextlib import contextmanager


@contextmanager
def configure_torch_backends():
    """
    Context manager to disable memory-efficient and flash SDP (Sparse Dot Product)
    in torch.backends.cuda.
    """
    # Disable memory-efficient and flash SDP
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    try:
        yield
    finally:
        # Optionally, you could reset the settings back if needed
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)


# Run the original command
if __name__ == '__main__':
    with configure_torch_backends():
        subprocess.run([
            "poetry", "run", "python", "-m", "turbo_alignment", "train_sft",
            "--experiment_settings_path", "training/sft.json"
        ])