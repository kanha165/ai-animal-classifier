from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import io
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Animal Classification API")

# load model
model = tf.keras.models.load_model("multi_animal_model.h5")

# load class names
class_names = np.load("class_names.npy")

def preprocess_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize((150,150))

        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except UnidentifiedImageError:
        return None


@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # DEBUG (VERY IMPORTANT)
    print("Filename:", file.filename)
    print("Content-Type:", file.content_type)

    # Validate file type
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    contents = await file.read()

    if not contents:
        return {"error": "Empty file received"}

    img = preprocess_image(contents)

    if img is None:
        return {"error": "Invalid image format"}

    predictions = model.predict(img)

    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "prediction": str(class_names[class_index]),
        "confidence": round(confidence, 3)
    }