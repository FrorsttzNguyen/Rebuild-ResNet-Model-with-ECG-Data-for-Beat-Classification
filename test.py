import torch

# Load the checkpoint
checkpoint_path = '/Users/nguyenhien/Library/CloudStorage/OneDrive-VNU-HCMUS/mit_beat_classification/max_checkpoint.pth'
checkpoint = torch.load(checkpoint_path)

# Print the checkpoint contents
print(checkpoint)
