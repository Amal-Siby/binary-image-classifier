import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr

# Load model
model = tf.keras.models.load_model("my_model.h5")
IMG_SIZE = 160

def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    return f"{label} ({confidence * 100:.2f}% confident)"

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a Cat or Dog Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="🐾 Cat vs Dog Classifier",
    description="Upload an image of a cat or dog and this smart classifier will tell you what it sees with a confidence score.",
    theme="huggingface",
    allow_flagging="never"
).launch()
