import torch
from prediction import MultimodalClinicalModel   # adjust import if needed
from torchinfo import summary
# Create model
model = MultimodalClinicalModel()

# Count trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable params:", count_trainable_params(model))

# Print model summary
summary(model, input_data=[
    torch.randn(1, 512),
    torch.randn(1, 768),
    torch.randn(1, 4)
])