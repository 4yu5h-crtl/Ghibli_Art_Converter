# Studio Ghibli Art Generator

This project allows you to transform your images into Studio Ghibli-style artwork using Stable Diffusion through a user-friendly web interface.

## Requirements

- Python 3.8 or higher
- PyTorch
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser
3. Upload an image using the file uploader
4. Adjust the settings in the sidebar:
   - Transformation Strength: Controls how much the AI modifies the image
   - Guidance Scale: Controls how closely the AI follows the prompt
   - Style Prompt: Customize the transformation style
5. Click "Transform Image" to generate the Ghibli-style version
6. Download the generated image using the download button

## Features

- User-friendly web interface
- Real-time image preview
- Adjustable transformation parameters
- Direct download of generated images
- Helpful tips and guidance

## Troubleshooting

If you experience performance issues:
- Try reducing the image size before uploading
- Lower the transformation strength
- Ensure you have enough GPU memory if using a GPU
- The first run will download the Stable Diffusion model, which is several gigabytes in size

## Note

The application uses Streamlit's caching to improve performance. The model will only be loaded once per session. 