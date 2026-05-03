import json
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

class ModelConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = None
    transforms = None
    labels = None

    @classmethod
    def load_model(cls):
        if cls.model is not None:
            return cls.model

        print("🔄 Loading fine-tuned Intel Image Classification model...")

        # Load checkpoint
        checkpoint = torch.load('resources/intel_mobilenetv2.pth', 
                              map_location=cls.device, 
                              weights_only=True)

        # Create model architecture
        cls.model = mobilenet_v2(pretrained=False)
        num_ftrs = cls.model.classifier[1].in_features
        cls.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_ftrs, 6)
        )

        cls.model.load_state_dict(checkpoint['model_state_dict'])
        cls.model = cls.model.to(cls.device)
        cls.model.eval()

        # Load labels
        with open('resources/labels.json', 'r') as f:
            cls.labels = json.load(f)

        # Simple transforms using torchvision
        cls.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Model loaded successfully on {cls.device}!")
        print(f"Classes: {list(cls.labels.values())}")
        return cls.model


def load_image(image_path):
    """Load image using PIL (more stable)"""
    image = Image.open(image_path).convert('RGB')
    return image


def predict_class(image_path, model_config=ModelConfig, count=3):
    ModelConfig.load_model()

    # Load and transform image
    image = load_image(image_path)
    image = ModelConfig.transforms(image).unsqueeze(0)
    image = image.to(ModelConfig.device)

    with torch.no_grad():
        output = ModelConfig.model(image)
        probs = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    # Get top 3 predictions
    max_indexes = probs.argsort()[-count:][::-1]
    max_values = probs[max_indexes].tolist()
    names = [ModelConfig.labels.get(str(idx), "Unknown") for idx in max_indexes]

    return max_values, names