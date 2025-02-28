from diffusers import DiffusionPipeline
import torch

# Load Stable Diffusion v1-5 model
base = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # Use float32 for CPU (not float16)
    use_safetensors=True
)
base.to("cpu")  # Move the model to CPU

# Define how many steps and what % of steps to be run on each expert (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "an ultra-realistic, full-body portrait of a captivating Devi durga with 10 hands wach carrying weapons (trident, swords, battleaxe) , radiating godly power and terrifying grace. showing bangles on hand, bindi, high-quality photo portrait, shot on a Kodak camera, photography, environmental, double eyelid --ar 34:65 --style raw, indian ascent, flying kiss, feeling happy, beautiful legs, wearing violet transparent sari with black silk blouse, includes full body structure,"

# Run the model
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
).images[0]

# Save the generated image
image.save("generated_image.png")
print("✅ Image generated and saved as generated_image.png")
