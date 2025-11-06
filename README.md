
# 🩺 Computer Vision Kidney Disease Predictor

This project is a **Flask-based deep learning web application** that classifies **kidney CT scan images** into four categories — **Cyst, Normal, Stone, and Tumor** — using a pre-trained **Convolutional Neural Network (CNN)** model built with **TensorFlow and Keras**.

The application provides an interactive web interface where users can upload medical images and receive real-time predictions, supported by probability distributions for each class.

---

![screenshot](https://github.com/user-attachments/assets/e8279c8c-bf12-425b-b985-8fcadddcd2ac)



## 🚀 Features
- 🧬 Deep learning–based kidney CT classification  
- 🖼️ Image upload via a web UI built with HTML, CSS, and JavaScript  
- ⚙️ Flask backend serving a trained Keras model (`.h5` file)  
- 📊 JSON output with class probabilities and top prediction  
- 💻 Lightweight and easy to run locally (no Docker required)

---

## 📂 Project Structure
Computer-vision-Kidney-disease-predictor/
│
├─ models/                              # Stores model assets
│   ├─ class_names.json                 # List of output class labels
│   └─ kidney_model_best.h5 (local only)# Trained CNN model (not uploaded to GitHub)
│
├─ src/                                 # Core backend logic
│   └─ app.py                           # Main Flask app serving predictions
│
├─ static/                              # Front-end static assets (CSS & JS)
│   ├─ script.js                        # Handles uploads and AJAX prediction requests
│   └─ style.css                        # Web interface styling
│
├─ templates/                           # HTML templates (Flask Jinja2)
│   └─ index.html                       # Main web UI page for uploading and viewing results
│
├─ tests/                               # Testing and standalone prediction scripts
│   └─ test_predict.py                  # CLI test for model inference
│
├─ .gitignore                           # Files & folders excluded from Git tracking
├─ LICENSE                              # Open-source MIT License (© 2025 Zainab Jamil)
├─ README.md                            # Project documentation and usage instructions
├─ kidney_tumor_code.py                 # (Optional) Additional model/code file for reference or training
└─ requirements.txt                     # Python dependencies list


---

## 🧩 Model Details
- Framework: **TensorFlow / Keras**
- Input size: **224 × 224 × 3**
- Output classes:  
  1. Cyst  
  2. Normal  
  3. Stone  
  4. Tumor  

You can retrain or fine-tune your model and replace `models/kidney_model_best.h5` with your updated version.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/zanjamil30-ops/Computer-vision-Kidney-disease-predictor.git
cd Computer-vision-Kidney-disease-predictor

2. Create and Activate a Virtual Environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python src/app.py
Then open your browser and go to 👉 http://127.0.0.1:5000/

🧠 How It Works

Upload an image from your local system.

The backend preprocesses the image (resizing, scaling).

The trained CNN model predicts the probabilities for each class.

The top class and full probability distribution are displayed in the browser.


🧪 Testing the Model (Command Line)

You can test predictions directly without the Flask UI:

python tests/test_predict.py


Update the script’s IMG_PATH variable with the path to your test image.

📸 Front-End Overview

index.html: Simple, responsive upload interface

script.js: Handles image preview and AJAX call to /predict endpoint

style.css: Modern, minimal styling


⚖️ License

This project is licensed under the MIT License — see the LICENSE
 file for details.

👩‍💻 Author

Zainab Jamil
📍 GitHub: @zanjamil30-ops

https://www.linkedin.com/in/zainab-jamil-b73824329/

🌟 Acknowledgements

TensorFlow & Keras for deep learning support

Flask for lightweight model serving

PIL (Pillow) for image processing

“Empowering medical imaging with AI for faster and smarter diagnostics.”
