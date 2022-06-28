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
from FaceNet512Model import *
from ArcFaceModel import *
from DeepFaceModel import *
from DeepIDModel import *

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tqdm
import sys

f = open('out.txt', 'w')

#PARAMETERS

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"

backends = ["opencv", "ssd", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "DeepID"]
#models = ["VGG-Face", "Facenet", "OpenFace"]
dist = ["cosine", "euclidean", "euclidean_l2"]

MODEL = "VGG-Face"
BACKEND = "retinaface"
DIST = "cosine"

distance_positive_euc = []
distance_negative_euc = []

distance_positive_eucl2 = []
distance_negative_eucl2 = []

distance_positive_cos = []
distance_negative_cos = []
decisions = []

j = 0

def getModel(model_name):
	if model_name == "VGG-Face":
		model = getVGGModel()
	if model_name == "Facenet":
		model = getFaceNetModel()
	if model_name == "OpenFace":
		model = getOpenFaceModel()
	if model_name == "Facenet512":
		model = getFaceNet512Model()
	if model_name == "ArcFace":
		model = getArcFaceModel()
	if model_name == "DeepFace":
		model = getDeepFaceModel()
	if model_name == "DeepID":
		model = getDeepIDModel()
	return model

def getWindow(model_name):
	if model_name == "VGG-Face":
		return getVGGWindowSize()
	if model_name == "Facenet":
		return getFaceNetWindowSize()
	if model_name == "OpenFace":
		return getOpenFaceWindowSize()
	if model_name == "Facenet512":
		return getFaceNet512WindowSize()
	if model_name == "ArcFace":
		return getArcFaceWindowSize()
	if model_name == "DeepFace":
		return getDeepFaceWindowSize()
	if model_name == "DeepID":
		return getDeepIDWindowSize()

def findDistance(img1_embedding, img2_embedding, metric):
	if metric == 'cosine':
		distance = dst.findCosineDistance(img1_embedding, img2_embedding)
	elif metric == 'euclidean':
		distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
	elif metric == 'euclidean_l2':
		distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))
	return distance


VGG_Cosine = 0.31
VGG_Euclidean = 0.45
VGG_Euclidean_l2 = 0.78

OpenFace_Cosine = 0.32
OpenFace_Euclidean = 0.93
OpenFace_Euclidean_l2 = 0.93

FaceNet_Cosine = 0.61
FaceNet_Euclidean = 13.83
FaceNet_Euclidean_l2 = 1.09

Facenet512_Cosine = 0.57
DeepFace_Cosine = 0.20
ArcFace_Cosine = 0.73
DeepID_Cosine = 0.05

euclidean_predicted = []
cosine_predicted = []
euclidean_l2_predicted = []

actual = []

# databases = ["D:/Praca Magisterska/DB/compress10/", "D:/Praca Magisterska/DB/compress20/", "D:/Praca Magisterska/DB/compress50/", 
# 			"D:/Praca Magisterska/DB/gauss/", "D:/Praca Magisterska/DB/motionblur_hor_5/", "D:/Praca Magisterska/DB/motionblur_hor_10/", 
# 			"D:/Praca Magisterska/DB/motionblur_ver_5/", "D:/Praca Magisterska/DB/motionblur_ver_10/", "D:/Praca Magisterska/DB/snp/"]

#databases = ["D:/Praca Magisterska/DB/motionblur_ver_5/", "D:/Praca Magisterska/DB/motionblur_ver_10/", "D:/Praca Magisterska/DB/snp/"]
databases = ["D:/Praca Magisterska/DB/lfw/lfw/"]

DBSIZE = 500

def checkDistance(modelName, distanceName, distance):
	if model_name == "VGG-Face":
		if distanceName == "euclidean":
			if distance < VGG_Euclidean:
				return True
			else:
				return False
		if distanceName == "cosine":
			if distance < VGG_Cosine:
				return True
			else:
				return False
		if distanceName == "euclidean_l2":
			if distance < VGG_Euclidean_l2:
				return True
			else:
				return False
		
	if model_name == "Facenet":
		if distanceName == "euclidean":
			if distance < FaceNet_Euclidean:
				return True
			else:
				return False
		if distanceName == "cosine":
			if distance < FaceNet_Cosine:
				return True
			else:
				return False
		if distanceName == "euclidean_l2":
			if distance < FaceNet_Euclidean_l2:
				return True
			else:
				return False
	if model_name == "OpenFace":
		if distanceName == "euclidean":
			if distance < OpenFace_Euclidean:
				return True
			else:
				return False
		if distanceName == "cosine":
			if distance < OpenFace_Cosine:
				return True
			else:
				return False
		if distanceName == "euclidean_l2":
			if distance < OpenFace_Euclidean_l2:
				return True
			else:
				return False
	if model_name == "Facenet512":
		if distance < Facenet512_Cosine:
			return True
		return False
	if model_name == "DeepFace":
		if distance < DeepFace_Cosine:
			return True
		return False
	if model_name == "ArcFace":
		if distance < ArcFace_Cosine:
			return True
		return False
	if model_name == "DeepID":
		if distance < DeepID_Cosine:
			return True
		return False

import csv
with open("pairsDevTestFull.txt", 'r') as csvfile:
	trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

for database in databases:
	print("Done!")
	for model_name in tqdm.tqdm(models):
		model = getModel(model_name)
		for i, row in enumerate(trainrows):
			if i < DBSIZE:
				filename1 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[1]))
				filename2 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[2]))
			else:
				filename1 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[1]))
				filename2 = "{0}/{1}_{2:04d}.jpg".format(row[2], row[2], int(row[3]))
			
			img1 = functions.preprocess_face(f"{database}{filename1}", target_size = getWindow(model_name), detector_backend='mtcnn', enforce_detection=False)
			img2 = functions.preprocess_face(f"{database}{filename2}", target_size = getWindow(model_name), detector_backend='mtcnn', enforce_detection=False)

			img1_embed = model.predict(img1)[0]
			img2_embed = model.predict(img2)[0]

			distance_euc = findDistance(img1_embed, img2_embed, metric='euclidean')
			distance_euc_l2 = findDistance(img1_embed, img2_embed, metric='euclidean_l2')
			distance_cos = findDistance(img1_embed, img2_embed, metric='cosine')

			# #euclidean
			# if(checkDistance(model_name, "euclidean", distance_euc)):
			# 	euclidean_predicted.append(1)
			# else:
			# 	euclidean_predicted.append(0)

			# #euclidean_l2
			# if(checkDistance(model_name, "euclidean_l2", distance_euc_l2)):
			# 	euclidean_l2_predicted.append(1)
			# else:
			# 	euclidean_l2_predicted.append(0)

			#cosine
			if(checkDistance(model_name, "cosine", distance_cos)):
				cosine_predicted.append(1)
			else:
				cosine_predicted.append(0)
			if i < DBSIZE:
				actual.append(1)
			else:
				actual.append(0)

		print(f"Matrix for cosine, {model_name}, database - {database}", file=f)
		matrix = confusion_matrix(actual,cosine_predicted, labels=[1,0])
		print('Confusion matrix : \n',matrix, file=f)
		tp, fn, fp, tn = confusion_matrix(actual,cosine_predicted,labels=[1,0]).reshape(-1)
		print('Outcome values : \n', tp, fn, fp, tn, file=f)
		matrix = classification_report(actual,cosine_predicted,labels=[1,0])
		print('Classification report : \n',matrix, file=f)

		actual.clear()
		cosine_predicted.clear()
		euclidean_predicted.clear()
		euclidean_l2_predicted.clear()

		# print(f"Matrix for euclidean, {model_name}, database - {database}", file=f)
		# matrix = confusion_matrix(actual,euclidean_predicted, labels=[1,0])
		# print('Confusion matrix : \n',matrix, file=f)
		# tp, fn, fp, tn = confusion_matrix(actual,euclidean_predicted,labels=[1,0]).reshape(-1)
		# print('Outcome values : \n', tp, fn, fp, tn, file=f)
		# matrix = classification_report(actual,euclidean_predicted,labels=[1,0])
		# print('Classification report : \n',matrix, file=f)


		# print(f"Matrix for euclidean_l2, {model_name}, database - {database}", file=f)
		# matrix = confusion_matrix(actual,euclidean_l2_predicted, labels=[1,0])
		# print('Confusion matrix : \n',matrix, file=f)
		# tp, fn, fp, tn = confusion_matrix(actual,euclidean_l2_predicted,labels=[1,0]).reshape(-1)
		# print('Outcome values : \n', tp, fn, fp, tn, file=f)
		# matrix = classification_report(actual,euclidean_l2_predicted,labels=[1,0])
		# print('Classification report : \n',matrix, file=f)


		


