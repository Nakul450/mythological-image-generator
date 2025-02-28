import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer  # Import the tokenizer for CLIP

# Set Paths
MODEL_PATH = "model/stable-diffusion-v1-5/"
DATASET_PATH = "dataset/images"
OUTPUT_DIR = "lora_model_output"

# Load Stable Diffusion Model
print("🔄 Loading Stable Diffusion Model...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
print("✅ Model Loaded Successfully!")

# Define target modules for LoRA
target_modules = [
    "conv_in",
    "down_blocks.0.resnets.0.conv1",
    "down_blocks.0.resnets.0.conv2",
    "down_blocks.1.resnets.0.conv1",
    "down_blocks.1.resnets.0.conv2",
    "up_blocks.0.resnets.0.conv1",
    "up_blocks.0.resnets.0.conv2",
    "down_blocks.0.downsamplers.0.conv",
    "down_blocks.1.downsamplers.0.conv",
    "down_blocks.2.downsamplers.0.conv",
    "down_blocks.3.downsamplers.0.conv",
    "up_blocks.1.upsamplers.0.conv",
    "up_blocks.2.upsamplers.0.conv",
    "up_blocks.3.upsamplers.0.conv",
]

lora_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",  # Updated to a valid task type
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=target_modules
)

# Manually add model_type to UNet config
pipe.unet.config["model_type"] = "stable-diffusion"

# Apply LoRA to the model
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()

# Load Dataset
def load_images_from_folder(folder):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = load_images_from_folder(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# Use the custom dataset
print("🔄 Loading Dataset...")
train_dataset = ImageDataset(DATASET_PATH)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Updated batch size to 1
print(f"✅ Loaded {len(train_dataset)} images for training.")

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)

def get_encoder_hidden_states(batch):
    text_inputs = ["Your text prompt here"] * len(batch)  # Replace with actual text inputs
    inputs = tokenizer(text_inputs, padding=True, return_tensors="pt")
    encoder_output = pipe.text_encoder(**inputs)
    return encoder_output.last_hidden_state

# Training Loop
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)

print("🚀 Training LoRA...")
for epoch in range(10):
    for batch in train_loader:
        batch = batch.to("cuda").float()  # Ensure it's float
        if batch.dim() == 5:
            batch = batch.squeeze(1)  # Remove extra dimension if present

        batch = pipe.vae.encode(batch).latent_dist.sample() * 0.18215

        noise = torch.randn_like(batch).to("cuda")
        encoder_hidden_states = get_encoder_hidden_states(batch)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device
        ).long().unsqueeze(-1)

        noise_pred = pipe.unet(
            sample=batch,
            timestep=timesteps.squeeze(),
            encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch {epoch + 1}/10 - Loss: {loss.item()}")

# Save Fine-Tuned LoRA Model
print("💾 Saving fine-tuned LoRA model...")
pipe.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved in {OUTPUT_DIR}!")

