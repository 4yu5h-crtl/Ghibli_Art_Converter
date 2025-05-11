import streamlit as st
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionImg2ImgPipeline
from PIL import Image
import io
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Ghibli Style Converter",
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
st.markdown('<div style="text-align:center;"><span class="ghibli-header-icon">🎨</span><span style="font-size:2.2em;font-weight:800;color:#6ec1e4;">Ghibli Style Image Converter</span></div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#bfc9d1; font-size:1.1em; margin-bottom:1.5em;'>This app converts your images to Ghibli-style artwork using the Stable Diffusion model.<br>Upload an image, adjust the stylization strength, and get your Ghibli-style portrait!</div>", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    model_id = "nitrosocke/Ghibli-Diffusion"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    with st.spinner("Loading model... This may take a few minutes the first time."):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.enable_attention_slicing()  # Optimize memory usage
    return pipe

# Function to generate Ghibli-style image
def generate_ghibli_image(image, pipe, strength):
    image = image.convert("RGB")
    image = image.resize((512, 512))  # Ensure proper size
    prompt = "Ghibli-style anime painting, soft pastel colors, highly detailed, masterpiece"
    start_time = time.time()
    result = pipe(prompt=prompt, image=image, strength=strength).images[0]
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
    strength = st.slider(
        "Stylization Strength", 
        min_value=0.3, 
        max_value=0.8, 
        value=0.6, 
        step=0.05,
        help="Higher values (closer to 0.8) will make the output more stylized but may lose some details from the original image."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Model loading (outside UI containers for caching)
pipe = load_model()

# Image processing logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    if st.button("Generate Ghibli Style Image", use_container_width=True):
        with st.spinner("Generating your Ghibli-style image... This may take a minute or two."):
            progress = st.progress(0)
            for i in range(1, 5):
                time.sleep(0.2)
                progress.progress(i * 20)
            result_img, gen_time = generate_ghibli_image(image, pipe, strength)
            progress.progress(100)
            # Show result card only after generation
            with st.container():
                st.markdown('<div class="ghibli-card">', unsafe_allow_html=True)
                st.subheader("Ghibli Style Result")
                st.image(result_img, caption="Ghibli Style Portrait", use_column_width=True)
                st.success(f"Image generated in {gen_time:.2f} seconds!")
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Ghibli Style Image",
                    data=byte_im,
                    file_name=f"ghibli_portrait_{uploaded_file.name}",
                    mime="image/png"
                )
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='text-align:center; color:#aaa; font-size:1.1em;'>Upload an image and click 'Generate Ghibli Style Image' to see the result here.</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("""
<div class='ghibli-footer'>
    <p style='font-size:0.95em;'>Model used: <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></p>
</div>
""", unsafe_allow_html=True) 