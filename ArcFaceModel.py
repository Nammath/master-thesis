from deepface.basemodels import ArcFace

import os
from pathlib import Path
import gdown
import numpy as np

import matplotlib.pyplot as plt
import cv2

from deepface.commons import functions

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
else:
	from tensorflow import keras
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation


from keras.models import model_from_json
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


def getArcFaceModel():
    model = ArcFace.loadModel()
    model.load_weights("C:/Users/Daniel/.deepface/weights/arcface_weights.h5")
    return model

def getArcFaceWindowSize():
    return (112,112)