from flask import Flask,request,send_file
from flasgger import Swagger
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from segment import segment_image
from PIL import Image


app=Flask(__name__) # it's a common step to start with this
Swagger(app) # pass the App to Swagger

current_directory = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_directory, "model.h5")

model = tensorflow.keras.models.load_model(model_path)
@app.route('/') # must be written to define the root page or main page to display
# this will display a web page having welcome all in it
def welcome():
    return "Welcome All"

# a page for predicting one sample, can be used through Postman

@app.route('/detect',methods=["POST"]) # by default it's GET method because we will pass our features as parameters
def detect_A_sample():
    """
    Let's segment image
    ---
    parameters:
        
        - name: image
          in: formData
          type: file
          required: true
    produces:
        - image/*
    responses:
        200:
            description: ok
            content:
                image/jpg: {}

    """
    image_path = os.path.join(current_directory, "image.jpg")
    if os.path.exists(image_path):
        # Delete the file
        os.remove(image_path)
    image = request.files.get("image")
    image = Image.open(image)
    image = np.array(image.resize((192, 256)))
    image = image.reshape((1, 192, 256,3))

    segment_image(image, image_path, model)
    image = open(image_path, 'rb')
    return  send_file(image, mimetype='image/jpg')

    

if __name__=='__main__':
    app.run()