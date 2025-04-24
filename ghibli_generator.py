import torch
import tkinter as tk
from tkinter import filedialog
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def main():
    # Set device to MPS (Apple Metal) or CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Stable Diffusion model
    print("Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32
    ).to(device)

    # Create file dialog to select an image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )

    if not file_path:
        print("No file selected. Exiting...")
        return

    # Load and prepare the input image
    print("Processing image...")
    init_image = Image.open(file_path).convert("RGB").resize((512, 512))

    # Define the transformation prompt
    prompt = "A beautiful Studio Ghibli-style landscape with vibrant colors and soft lighting"

    # Generate the transformed image
    print("Generating Ghibli-style image...")
    generator = torch.manual_seed(42)
    output_image = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.75,  # Controls how much the AI modifies the image
        guidance_scale=7.5  # Controls how closely the AI follows the prompt
    ).images[0]

    # Save and show the result
    output_path = "ghibli_output.png"
    output_image.save(output_path)
    print(f"Image saved as {output_path}")
    output_image.show()

if __name__ == "__main__":
    main() 