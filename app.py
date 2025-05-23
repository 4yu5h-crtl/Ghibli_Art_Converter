import streamlit as st
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import io
import time
import os
import cv2
import numpy as np
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Studio Ghibli Style Converter",
    page_icon="🎨",
    layout="centered"
)

# Custom CSS for improved dark UI
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #181c22 0%, #23272f 100%) !important;
}

section.main > div {
    max-width: 700px;
    margin: auto;
}

h1, .stTitle {
    color: #6ec1e4 !important;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-weight: 800;
    letter-spacing: 1px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 1.1em;
}
.stButton>button:hover {
    background-color: #45a049;
}

.stFileUploader, .stSlider, .stImage, .stDownloadButton {
    background: #23272f !important;
    color: #f1f1f1 !important;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    padding: 1.5em 1em;
    margin-bottom: 1.5em;
    border: 1px solid #353b48;
}

.stAlert {
    border-radius: 10px;
}

.ghibli-footer {
    text-align: center;
    color: #888;
    font-size: 1em;
    margin-top: 2em;
    margin-bottom: 1em;
}

.ghibli-header-icon {
    font-size: 2.2em;
    vertical-align: middle;
    margin-right: 0.3em;
}

.ghibli-card {
    background: #23272f !important;
    color: #f1f1f1 !important;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
    padding: 2em 1.5em;
    margin-bottom: 2em;
    border: 1px solid #353b48;
}

.stMarkdown h2 {
    color: #6ec1e4;
    font-weight: 700;
}

/* Slider track and handle */
.css-1c7y2kd .stSlider > div[data-baseweb="slider"] > div {
    background: #353b48 !important;
}
.css-1c7y2kd .stSlider > div[data-baseweb="slider"] .css-14xtw13 {
    background: #6ec1e4 !important;
}

/* Download button dark style */
.stDownloadButton > button {
    background: #353b48 !important;
    color: #f1f1f1 !important;
    border-radius: 8px;
    border: 1px solid #6ec1e4;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    background: #6ec1e4 !important;
    color: #23272f !important;
}
</style>
""", unsafe_allow_html=True)

# Header with icon
st.markdown('<div style="text-align:center;"><span class="ghibli-header-icon">🎨</span><span style="font-size:2.2em;font-weight:800;color:#6ec1e4;">Studio Ghibli Style Image Converter</span></div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#bfc9d1; font-size:1.1em; margin-bottom:1.5em;'>This app converts your images to Studio Ghibli-style artwork using Stable Diffusion with ControlNet and Ghibli LoRA.<br>Upload an image, adjust the settings, and get your Ghibli-style illustration!</div>", unsafe_allow_html=True)

# Function to convert image to Canny edge map
def get_canny(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image] * 3, axis=2)
    return Image.fromarray(image)

# Function to load the model
@st.cache_resource
def load_model():
    with st.spinner("Loading ControlNet model... This may take a few minutes the first time."):
        # Load ControlNet model for Canny
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        
        # Load base Stable Diffusion model
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        
        # Load Ghibli LoRA
        with st.spinner("Downloading and loading Ghibli LoRA..."):
            # Check if LoRA file already exists
            lora_path = "ghibli.safetensors"
            if not os.path.exists(lora_path):
                # Download LoRA file - using the correct URL from Civitai
                lora_url = "https://civitai.com/api/download/models/6526"  # Studio Ghibli Style LoRA
                response = requests.get(lora_url)
                with open(lora_path, 'wb') as f:
                    f.write(response.content)
            
            # Load LoRA weights
            pipe.load_lora_weights(".", weight_name="ghibli.safetensors")
            pipe.fuse_lora()
    
    return pipe

# Function to generate Ghibli-style image
def generate_ghibli_image(image, control_image, pipe, strength, guidance_scale, num_inference_steps):
    image = image.convert("RGB")
    image = image.resize((512, 512))  # Ensure proper size
    
    prompt = "Studio Ghibli style illustration, soft watercolor shading, pastel colors, cinematic lighting, gentle brush strokes, by Hayao Miyazaki"
    negative_prompt = "blurry, deformed, ugly, extra limbs, neon, oversaturated, 3d render"
    
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    ).images[0]
    
    generation_time = time.time() - start_time
    return result, generation_time

# GPU warning/info
if torch.cuda.is_available():
    st.success("✅ GPU is available! Processing will be fast.")
else:
    st.warning("⚠️ GPU not available. Processing will be slow. Consider using a machine with GPU support.")

# Main card for upload and controls
with st.container():
    st.markdown('<div class="ghibli-card">', unsafe_allow_html=True)
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Advanced settings in an expander
    with st.expander("Advanced Settings"):
        strength = st.slider(
            "Stylization Strength", 
            min_value=0.3, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Higher values will make the output more stylized but may lose some details from the original image."
        )
        
        guidance_scale = st.slider(
            "Guidance Scale", 
            min_value=1.0, 
            max_value=15.0, 
            value=8.0, 
            step=0.5,
            help="Higher values make the image follow the prompt more closely but may reduce quality."
        )
        
        num_inference_steps = st.slider(
            "Inference Steps", 
            min_value=20, 
            max_value=50, 
            value=40, 
            step=1,
            help="More steps generally produce better quality but take longer to generate."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Model loading (outside UI containers for caching)
pipe = load_model()

# Image processing logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns for original and edge map
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    # Generate Canny edge map
    control_image = get_canny(image)
    with col2:
        st.image(control_image, caption="Canny Edge Map", use_column_width=True)
    
    if st.button("Generate Ghibli Style Image", use_container_width=True):
        with st.spinner("Generating your Ghibli-style image... This may take a minute or two."):
            progress = st.progress(0)
            for i in range(1, 5):
                time.sleep(0.2)
                progress.progress(i * 20)
            
            result_img, gen_time = generate_ghibli_image(
                image, 
                control_image, 
                pipe, 
                strength, 
                guidance_scale, 
                num_inference_steps
            )
            
            progress.progress(100)
            
            # Show result card only after generation
            with st.container():
                st.markdown('<div class="ghibli-card">', unsafe_allow_html=True)
                st.subheader("Ghibli Style Result")
                st.image(result_img, caption="Ghibli Style Illustration", use_column_width=True)
                st.success(f"Image generated in {gen_time:.2f} seconds!")
                
                # Save the image to a buffer
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                # Download button
                st.download_button(
                    label="Download Ghibli Style Image",
                    data=byte_im,
                    file_name=f"ghibli_style_{uploaded_file.name}",
                    mime="image/png"
                )
                
                # Upscaling option
                st.subheader("Upscale Image (Optional)")
                st.info("For better quality, you can upscale the generated image using Real-ESRGAN.")
                if st.button("Upscale Image", use_container_width=True):
                    st.warning("Upscaling functionality requires integration with a Real-ESRGAN API or Hugging Face Space.")
                    st.info("In a production environment, you would implement the upscaling here.")
                
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='text-align:center; color:#aaa; font-size:1.1em;'>Upload an image and click 'Generate Ghibli Style Image' to see the result here.</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("""
<div class='ghibli-footer'>
    <p style='font-size:0.95em;'>Models used: <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>Stable Diffusion v1.5</a>, 
    <a href='https://huggingface.co/lllyasviel/sd-controlnet-canny' target='_blank'>ControlNet Canny</a>, and 
    <a href='https://civitai.com/models/6526/studio-ghibli-style-lora' target='_blank'>Studio Ghibli Style LoRA</a></p>
</div>
""", unsafe_allow_html=True) 
