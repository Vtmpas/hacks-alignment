import torch
import subprocess

# Disable memory-efficient and flash SDP (Sparse Dot Product)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Run the original command
if __name__ == '__main__':
    subprocess.run([
        "poetry", "run", "python", "-m", "turbo_alignment", "train_sft",
        "--experiment_settings_path", "training/sft.json"
    ])