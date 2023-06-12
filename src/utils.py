import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os


def load_image(image_path, target_size=(224, 224)):
    # Load and preprocess the image from the given file path.

    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    return image


def load_model(model_path):
    # Load the pre-trained model from the given file path.
    model = tf.keras.models.load_model(model_path)
    return model


def predict_logo(model, image):
    # Predict the company logo from the given image.

    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_logo = predictions.argmax()
    return predicted_logo

def get_predicted(logos,index):
    # Return the predicted logo from the logos list
    return logos[index]

def get_classes(folder):
    # Return the class names from the folder
    return os.listdir(folder)