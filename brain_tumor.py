import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
global graph,model
graph = tf.compat.v1.get_default_graph
with tf.compat.v1.Session() as ses:

    def model_predict_cnn(img_path):
        model = load_model(('brain_tumor_detector.h5'), custom_objects={'KerasLayer': hub.KerasLayer})
        #img_path="./static/Y1.jpg"
        test_image = image.load_img(img_path, target_size = (240,240))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if result <= 0.5:
            result = "The Person has no Brain Tumor"
        else:
            result = "The Person has Brain Tumor"

        return result

preds = model_predict_cnn("./static/Y1.jpg")
print(preds)
