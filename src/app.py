from distutils.file_util import move_file
import json
from flask import Flask, render_template, jsonify, request
from tensorflow import keras
from PIL import Image

import tensorflow_hub as hub
import numpy as np
import cv2

app = Flask(__name__)
modelo_v1 = keras.models.load_model('src/modelo_clasificacion_nubes.h5', custom_objects={'KerasLayer':hub.KerasLayer})
modelo_v2 = keras.models.load_model('src/modelo_clasificacion_cielo.h5', custom_objects={'KerasLayer':hub.KerasLayer})

def clasificar(modelo, imagen):
    img = Image.open(imagen)
    img = np.array(img).astype(float)/255
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/v2')
def segundo_modelo():
    return render_template('index2.html')

@app.route('/clasificar_nubes', methods=['POST'])
def clasificarNubes():

    imagen = request.files['imagen']

    clasificacion = clasificar(modelo=modelo_v1, imagen=imagen)

    if clasificacion == 1:
        return jsonify({"mensaje": "Hay probabilidad de lluvia"})
    else:
        return jsonify({"mensaje": "No hay probabilidad de lluvia"})   

@app.route('/clasificar_cielo', methods=['POST'])
def clasificarCielo():

    imagen = request.files['imagen']

    clasificacion = clasificar(modelo=modelo_v2, imagen=imagen)

    if clasificacion == 1:
        return jsonify({"mensaje": "Está nublado"})
    elif clasificacion == 2:
        return jsonify({"mensaje": "Está soleado"})
    else:
        return jsonify({"mensaje": "Es de noche"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)