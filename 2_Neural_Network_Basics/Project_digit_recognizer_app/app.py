from flask import Flask, render_template, request, jsonify
import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

import os

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates")
)


MODEL_PATH = hf_hub_download(
    repo_id="MohammadMinhasMustafa/best-mnist-model",
    filename="best_mnist_model.keras"
)

model = load_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['image']
    image_data = data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Simple centering without cropping
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array.astype('float32') / 255.0

    # Just reshape — most MNIST models work fine without perfect centering
    img_array = img_array.reshape(1, 28, 28)  # or (1, 28, 28, 1) if your model expects channel
    
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    return jsonify({'digit': digit, 'confidence': round(confidence, 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)