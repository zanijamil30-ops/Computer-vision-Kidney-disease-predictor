import json, os, numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "kidney_model_best (1).h5"
CLASS_JSON = "class_names.json"
IMG_PATH = "path/to/a/known_sample.jpg"  # replace with a sample image path

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

if os.path.exists(CLASS_JSON):
    classes = json.load(open(CLASS_JSON))
else:
    out_shape = model.output_shape
    n = int(out_shape[-1]) if isinstance(out_shape, (list,tuple)) else 1
    classes = [str(i) for i in range(n)]
print("Class order:", classes)

img = Image.open(IMG_PATH).convert("RGB").resize((224,224))
arr = (np.asarray(img).astype('float32') / 255.0)[None, ...]  # scale if model expects [0,1]
pred = model.predict(arr)[0]
idx = int(pred.argmax())
print("Predicted index:", idx)
print("Predicted label:", classes[idx])
print("Probabilities:", dict(zip(classes, [float(round(p,6)) for p in pred])))
