import sys
import subprocess
import os
import deepspeed

# automatically find where DeepSpeed is installed
ds_path = os.path.dirname(deepspeed.__file__)
converter_script = os.path.join(ds_path, "checkpoint/ds_to_universal.py")

# Define your paths
input_folder = "/path/to/checkpoints/source_dp4/global_step100"
output_folder = "/path/to/checkpoints/source_dp4/global_step100_universal"

# Run it
cmd = [
    sys.executable, converter_script,
    "--input_folder", input_folder,
    "--output_folder", output_folder
]

print("Starting conversion...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("Conversion Successful!")
    print(result.stdout)
else:
    print("Conversion Failed!")
    print(result.stderr)