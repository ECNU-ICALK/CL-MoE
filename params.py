import os
import torch
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--task1", type=str, default="")
parser.add_argument("--task2", type=str, default="")
args = parser.parse_args()

# Load finetuned weights
base_path = "/your/catalogue/CLMoE/checkpoints/CL4VQA"
finetuned_model1_path = os.path.join(base_path, args.task1, "llava-1.5-7b-lora/adapter_model.bin")
finetuned_model2_path = os.path.join(base_path, "Only_Pretrain_1.5_MOE_2", args.task2, "llava-1.5-7b-lora/adapter_model.bin")

finetuned_state_dict1 = torch.load(finetuned_model1_path)
finetuned_state_dict2 = torch.load(finetuned_model2_path)

# Define weights
alpha = 0.7

# Read index files
def read_indices(file_path):
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f]

index1 = read_indices(f"/your/catalogue/CLMoE/index_{args.task1}.txt")
index2 = read_indices(f"/your/catalogue/CLMoE/index_{args.task2}.txt")

# Determine indices that are only in task1
exclusive_indices = [i for i in index1 if i not in index2]

# Build the prefix list for targeted LoRA modules
exclusive_prefixes = [f"loraA.{i}." for i in exclusive_indices] + [f"loraB.{i}." for i in exclusive_indices]
default_prefixes = [f"loraA.{i}." for i in range(8)] + [f"loraB.{i}." for i in range(8)]

# Initialize combined model state_dict
combined_state_dict = finetuned_state_dict2.copy()

# Merge parameters
for name, param in finetuned_state_dict1.items():
    if any(name.startswith(prefix) for prefix in exclusive_prefixes):
        combined_state_dict[name] = (1 - alpha) * param + alpha * finetuned_state_dict2[name]
    elif any(name.startswith(prefix) for prefix in default_prefixes):
        combined_state_dict[name] = alpha * param + (1 - alpha) * finetuned_state_dict2[name]

# Copy model structure and save merged weights
source_folder = os.path.join(base_path, "Only_Pretrain_1.5_MOE_2", args.task2, "llava-1.5-7b-lora")
destination_folder = os.path.join(base_path, args.task2)
target_model_folder = os.path.join(destination_folder, "llava-1.5-7b-lora")

# Safely remove old target folder if it exists
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder, exist_ok=True)

shutil.copytree(source_folder, target_model_folder)
torch.save(combined_state_dict, os.path.join(target_model_folder, "adapter_model.bin"))

print("Model merge completed successfully.")
