from flask import Flask, render_template, jsonify, request
from tensorflow import keras
from PIL import Image

import tensorflow_hub as hub
import numpy as np
import cv2

app = Flask(__name__)
modelo = keras.models.load_model('src/modelo_clasificacion_nubes.h5', custom_objects={'KerasLayer':hub.KerasLayer})

def clasificarNube(imagen):
    img = Image.open(imagen)
    img = np.array(img).astype(float)/255
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():

    imagen = request.files['imagen']

    clasificacion = clasificarNube(imagen=imagen)

    print(clasificacion)

    if clasificacion == 1:
        return jsonify({"mensaje": "Hay probabilidad de lluvia"})
    else:
        return jsonify({"mensaje": "No hay probabilidad de lluvia"})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)