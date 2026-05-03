import torch
from torchvision.models import mobilenet_v2
import torch.onnx
import warnings
import os

# Suppress some warnings
warnings.filterwarnings("ignore")

print("Loading model for ONNX export...")

# Load checkpoint
checkpoint = torch.load('resources/intel_mobilenetv2.pth', map_location='cpu', weights_only=True)

# Create model
model = mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(num_ftrs, 6)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

print("Exporting model to ONNX...")

# Export to ONNX (using legacy exporter to avoid issues)
torch.onnx.export(
    model,
    dummy_input,
    "resources/intel_mobilenetv2.onnx",
    export_params=True,
    opset_version=13,           # Stable version for Triton
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ SUCCESS! Model exported to: resources/intel_mobilenetv2.onnx")
print("File size:", round(os.path.getsize("resources/intel_mobilenetv2.onnx") / (1024*1024), 2), "MB")