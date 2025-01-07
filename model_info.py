import torch
from models.resnet import resnet34

# Path to the checkpoint
model_path = r"exp_result\resnet34\peta_feat_classifier.pth"

# Initialize the model
model = resnet34()

# Load the checkpoint
checkpoint = torch.load(model_path)

# Adjust the keys in the checkpoint state_dict to match the model's state_dict
state_dict = checkpoint['state_dicts']
new_state_dict = {}

# Remove prefix (e.g., 'backbone.') if it exists
for k, v in state_dict.items():
    new_key = k.replace("backbone.", "")  # Adjust based on the mismatch
    new_key = new_key.replace("classifier.", "")  # Adjust as necessary
    new_state_dict[new_key] = v

# Load the adjusted state_dict
model.load_state_dict(new_state_dict, strict=False)

# Set the model to evaluation mode
model.eval()

# Print the model summary
from torchinfo import summary
summary(model)
