import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Point this to one of your event files (change the path if needed)
# I picked the folder path from your error log
log_dir = "z1_uni_ckpt/tensorboard/bf16/tp2_pp2_dp1_sp1_hd512_nl4_gbsz16_mbsz4_z1_LR_6.0e-3_6.0e-4_bf16_toy_save"

event_acc = EventAccumulator(log_dir)
event_acc.Reload()

print("Available Tags in this file:")
print("---------------------------")
for tag in event_acc.Tags()['scalars']:
    print(f"'{tag}'")