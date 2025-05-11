# Ghibli Style Image Converter

This application converts your images to Ghibli-style artwork using the Stable Diffusion model. It provides a user-friendly interface built with Streamlit.

![Ghibli Style Converter](https://i.imgur.com/example.png)

## Features

- Upload any image and convert it to Ghibli-style artwork
- Adjust the stylization strength to control how much the original image is transformed
- Download the generated Ghibli-style image
- User-friendly interface with real-time feedback

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Internet connection (for downloading the model)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Ghibli_Art_Converter.git
   cd Ghibli_Art_Converter
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Upload an image using the file uploader

4. Adjust the stylization strength using the slider (0.3-0.8, recommended: 0.6)

5. Click the "Generate Ghibli Style Image" button

6. Wait for the image to be processed (this may take a few minutes depending on your hardware)

7. Download the generated Ghibli-style image using the download button

## Troubleshooting

### Common Issues

1. **Import Errors**: If you encounter import errors related to numpy, transformers, or diffusers, make sure you're using the exact package versions specified in the requirements.txt file. These versions have been tested for compatibility.

2. **CUDA/GPU Issues**: If you have a GPU but the app doesn't detect it, make sure you have the correct CUDA drivers installed for your GPU.

3. **Memory Errors**: If you encounter out-of-memory errors, try reducing the image size or using a machine with more RAM.

4. **Model Download Issues**: If the model fails to download, check your internet connection and try again. The model is about 4GB in size.

### Fixing "numpy has no attribute 'dtypes'" Error

If you encounter the error `module 'numpy' has no attribute 'dtypes'`, it's likely due to incompatible package versions. The requirements.txt file has been updated with compatible versions to fix this issue. Make sure to:

1. Create a fresh virtual environment
2. Install the exact package versions from the requirements.txt file
3. Avoid installing other packages that might conflict with these versions

## Notes

- The first time you run the application, it will download the Ghibli Diffusion model, which is about 4GB in size. This may take some time depending on your internet connection.
- Processing images without a GPU will be significantly slower.
- The application uses the `nitrosocke/Ghibli-Diffusion` model from Hugging Face.