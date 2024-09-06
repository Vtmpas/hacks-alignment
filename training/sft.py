import subprocess
from pathlib import Path

if __name__ == '__main__':
    launch_code = 'python -m turbo_alignment train_sft --experiment_settings_path training/sft.json'
    subprocess.run(launch_code, shell=True)