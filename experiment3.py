from deepface.basemodels import Facenet, VGGFace, OpenFace
from deepface.commons import distance as dst
from deepface.commons import functions

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
import PIL
import pandas as pd
from chefboost import Chefboost as chef

#includes

from VGGModel import getVGGModel, getVGGWindowSize
from OpenFaceModel import getOpenFaceModel, getOpenFaceWindowSize
from FaceNetModel import getFaceNetModel, getFaceNetWindowSize
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import sys

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))