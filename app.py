import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Studio Ghibli Art Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 Studio Ghibli Art Generator")
st.markdown("""
Transform your images into beautiful Studio Ghibli-style artwork using AI!
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Strength slider
    strength = st.slider(
        "Transformation Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher values will transform the image more dramatically"
    )
    
    # Guidance scale slider
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=0.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="Controls how closely the AI follows the prompt"
    )
    
    # Custom prompt
    prompt = st.text_area(
        "Style Prompt",
        value="A beautiful Studio Ghibli-style landscape with vibrant colors and soft lighting",
        help="Describe the style you want to achieve"
    )

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image to transform",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to transform into Ghibli style"
)

# Initialize the model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32
    ).to(device)
    return pipe

# Main processing
if uploaded_file is not None:
    # Display original image
    st.subheader("Original Image")
    original_image = Image.open(uploaded_file)
    st.image(original_image, use_column_width=True)
    
    # Process button
    if st.button("Transform Image"):
        with st.spinner("Generating Ghibli-style image..."):
            # Load model
            pipe = load_model()
            
            # Prepare image
            init_image = original_image.convert("RGB").resize((512, 512))
            
            # Generate image
            generator = torch.manual_seed(42)
            output_image = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale
            ).images[0]
            
            # Display result
            st.subheader("Generated Image")
            st.image(output_image, use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Generated Image",
                data=byte_im,
                file_name="ghibli_output.png",
                mime="image/png"
            )

# Add some helpful information
st.markdown("""
### Tips for Best Results:
- Use high-quality images for better results
- Adjust the transformation strength to control how much the image changes
- Try different prompts to achieve different styles
- Higher guidance scale values will make the AI follow the prompt more closely

### Note:
The first run will download the Stable Diffusion model, which may take a few minutes.
""") 