import os
import shutil

print("Setting up Triton Model Repository...")

# Create directory structure
triton_dir = "model_repository/intel_image_classifier/1"
os.makedirs(triton_dir, exist_ok=True)

# Copy ONNX model
src = "resources/intel_mobilenetv2.onnx"
dst = os.path.join(triton_dir, "model.onnx")

if os.path.exists(src):
    shutil.copy2(src, dst)
    print(f"✅ Copied model to {dst}")
else:
    print("❌ ONNX model not found in resources/")

# Check
print("\nRepository structure:")
print(os.listdir("model_repository"))
print(os.listdir(triton_dir))