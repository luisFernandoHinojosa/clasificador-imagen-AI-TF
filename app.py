from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = ResNet50(weights='imagenet')


@app.route('/upload', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return jsonify({'error': 'No se subio ninguna imagen'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se selecciono ninguna imagen'}), 400

    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    imagenet_id, label, score = decoded_predictions[0]

    return jsonify({'label': label, 'score': f"{score * 100:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
