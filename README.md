# 🎨 Ghibli Style Image Converter

Transform your photos into magical Ghibli-style artwork using the power of AI! This app provides a beautiful, user-friendly interface built with Streamlit.

![Ghibli Style Converter](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjN5ZnB3NWViaTlpNHV0enFxdzZ3aGRxY2lhb3pxbzVvbWRvcGtzbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ylpMP58IO1arS/giphy.gif)

---

## ✨ Features

- 📤 **Upload any image** and convert it to Ghibli-style artwork
- 🎚️ **Adjust stylization strength** to control how much the original image is transformed
- ⬇️ **Download** the generated Ghibli-style image
- ⚡ **Real-time feedback** and progress indicators
- 🖥️ **Modern, dark-themed interface**

---


## 🛠️ Requirements

- 🐍 Python 3.8 or higher
- 💻 CUDA-compatible GPU (recommended for faster processing)
- 🌐 Internet connection (for downloading the model)

---

## 🚀 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/ghibli-converter.git
   cd ghibli-converter
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```
3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Open your web browser** and navigate to the URL displayed in the terminal (usually http://localhost:8501)
3. **Upload an image** using the file uploader
4. **Adjust the stylization strength** using the slider (0.3-0.8, recommended: 0.6)
5. **Click** the "Generate Ghibli Style Image" button
6. **Wait** for the image to be processed (this may take a few minutes depending on your hardware)
7. **Download** the generated Ghibli-style image using the download button

---

## 🧩 Troubleshooting

### Common Issues

- ❗ **Import Errors:** Ensure you're using the exact package versions specified in the requirements.txt file.
- ⚡ **CUDA/GPU Issues:** If you have a GPU but the app doesn't detect it, check your CUDA drivers.
- 🧠 **Memory Errors:** Try reducing the image size or use a machine with more RAM.
- 🌐 **Model Download Issues:** Check your internet connection. The model is about 4GB in size.

### Fixing "numpy has no attribute 'dtypes'" Error

If you encounter the error `module 'numpy' has no attribute 'dtypes'`, it's likely due to incompatible package versions. The requirements.txt file has been updated with compatible versions to fix this issue. Make sure to:

1. Create a fresh virtual environment
2. Install the exact package versions from the requirements.txt file
3. Avoid installing other packages that might conflict with these versions

---

## ℹ️ Notes

- 🕒 The first time you run the application, it will download the Ghibli Diffusion model (~4GB). This may take some time.
- 🐢 Processing images without a GPU will be significantly slower.
- 🤗 The application uses the [`nitrosocke/Ghibli-Diffusion`](https://huggingface.co/nitrosocke/Ghibli-Diffusion) model from Hugging Face.
