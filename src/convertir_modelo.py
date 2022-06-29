import tensorflowjs as tfjs
from tensorflow import keras
import tensorflow_hub as hub

modelo = keras.models.load_model('src/modelo_clasificacion_nubes.h5', custom_objects={'KerasLayer':hub.KerasLayer})

tfjs.converters.save_keras_model(modelo, 'modelo_tfjs')