from flask import Flask, render_template, redirect, request
import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

import brain_tumor
#app = Flask(__name__)

res=""
app=Flask(__name__,template_folder='template')
@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def marks():
    global res
    if request.method == 'POST':
        f = request.files['userfile']

        path = "./static/{}".format(f.filename)
        f.save(path)
        res=brain_tumor.model_predict_cnn(path)
        print("shashi:"+str(res))

        '''caption = caption_it.caption_this_image(path)
        print("shashi : " + caption)'''
        result_dic ={
            'image' :path,
            'caption' :res,
            'audiopath': path
        }
    return render_template("index.html", result_c =result_dic)


if __name__ == '__main__':
    app.run(debug=True)
