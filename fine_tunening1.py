import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model
print("🔄 Loading BLIP Model... (This may take some time)")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ Model Loaded Successfully!")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to generate captions using BLIP
def generate_caption(image_path):
    print(f"📷 Processing image: {image_path}")  # Ensure this message appears before processing
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate text based on the image
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption.strip()


# Create a list of images from your dataset
image_folder = "dataset/images"
captions = []

# Generate captions for each image
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(image_folder, image_file)
        caption = generate_caption(image_path)
        captions.append({"file": image_path, "caption": caption})

# Save captions to a JSON file
with open("dataset/captions.json", "w") as f:
    json.dump(captions, f, indent=4)

print("✅ Captions generated successfully!")
