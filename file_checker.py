import os

# ✅ Define the model path
MODEL_PATH = "model/playground-v2.5-1024px-aesthetic"

# ✅ List of required files
files_to_check = [
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "tokenizer",
    "scheduler",
    "model_index.json"
]

# ✅ Check for missing files
missing_files = [f for f in files_to_check if not os.path.exists(os.path.join(MODEL_PATH, f))]

if missing_files:
    print("❌ Missing files:", missing_files)
else:
    print("✅ All required files are present!")
