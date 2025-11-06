import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

# -------------------------
# Config
# -------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

MODEL_FILENAME = "kidney_model_best (1).h5"  # adjust if you rename model file
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)  # must match training
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# -------------------------
# Helpers
# -------------------------
def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def model_contains_rescaling_layer(m):
    """Return True if model contains a Keras Rescaling layer (so we should NOT scale again)."""
    try:
        for layer in m.layers:
            if layer.__class__.__name__ == "Rescaling":
                return True
    except Exception:
        pass
    return False

# Resampling compatibility for Pillow versions
try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_MODE = Image.ANTIALIAS

def preprocess_image(pil_img: Image.Image, target_size=IMG_SIZE, do_scale=True):
    """
    Convert PIL image to model input:
      - convert to RGB
      - resize with appropriate resampling for Pillow versions
      - optionally scale to [0,1] if do_scale==True
      - returns numpy array shape (1,H,W,3) dtype float32
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size, resample=RESAMPLE_MODE)
    arr = np.asarray(pil_img).astype(np.float32)
    if do_scale:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def load_class_names():
    """Try to read class_names.json. If missing or invalid, return numeric labels inferred from model output."""
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
            else:
                app.logger.warning("class_names.json exists but content is not a string list; ignoring.")
        except Exception as e:
            app.logger.warning("Failed to read class_names.json: %s", e)
    # fallback: numeric labels based on model output
    out_shape = model.output_shape
    try:
        num = int(out_shape[-1])
    except Exception:
        num = 1
    return [str(i) for i in range(num)]

# -------------------------
# Load model (once)
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Put your .h5 next to app.py or change MODEL_FILENAME.")

print("Loading model from:", MODEL_PATH)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}. Exception: {e}")

# Print model summary to console (helpful for debugging)
try:
    model.summary(print_fn=lambda s: print(s))
except Exception:
    pass

# Decide whether to scale inputs in preprocess (if model already includes Rescaling layer)
MODEL_HAS_RESCALING = model_contains_rescaling_layer(model)
if MODEL_HAS_RESCALING:
    print("Detected a Rescaling layer inside the model. Will NOT rescale inputs in preprocess (model expects [0,255] input).")
else:
    print("No Rescaling layer detected. preprocess will divide pixel values by 255.0 (model expects [0,1] input).")

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Validate upload
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: png, jpg, jpeg, bmp"}), 400

    # Read image
    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": "Cannot read the uploaded image", "detail": str(e)}), 400

    # (re)load class names here so edits take effect without restarting
    class_names = load_class_names()

    # Preprocess according to whether model expects [0,1]
    try:
        x = preprocess_image(pil_img, target_size=IMG_SIZE, do_scale=(not MODEL_HAS_RESCALING))
    except Exception as e:
        return jsonify({"error": "Preprocessing failed", "detail": str(e)}), 500

    # Predict
    try:
        preds = model.predict(x)
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    # Ensure preds shape
    try:
        probs = preds[0].tolist()
    except Exception as e:
        return jsonify({"error": "Unexpected model output shape", "detail": str(e)}), 500

    # build dict of label->prob (round for readability)
    label_probs = {}
    for i, p in enumerate(probs):
        label = class_names[i] if i < len(class_names) else str(i)
        label_probs[label] = float(round(float(p), 6))

    # top prediction
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    top_prob = float(round(float(probs[top_idx]), 6))

    response = {
        "prediction": top_label,
        "probability": top_prob,
        "all_probabilities": label_probs
    }
    return jsonify(response)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

