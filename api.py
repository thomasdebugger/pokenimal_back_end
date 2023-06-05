import io
from flask import Flask, request
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

# Charger le modèle MobileNetV2 pré-entraîné
model = MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si une image est incluse dans la requête
    if 'image' not in request.files:
        return "No image found", 400

    # Charger et prétraiter l'image
    img_file = request.files['image']
    img_bytes = img_file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Faire les prédictions
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # Renvoyer les prédictions
    response = []
    for pred in decoded_predictions:
        response.append({
            'label': pred[1],
            'score': str(pred[2])
        })

    return {'predictions': response}

if __name__ == '__main__':
    app.run()
