from deepface.basemodels import Facenet, VGGFace, OpenFace
from deepface.commons import distance as dst
from deepface.commons import functions

import os
from pathlib import Path
import gdown
import numpy as np

import matplotlib.pyplot as plt
import cv2
import tqdm

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
import time
#PARAMETERS

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"

backends = ["opencv", "ssd", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "DeepID"]
#models = ["DeepFace"]
dist = ["cosine", "euclidean", "euclidean_l2"]

SET_LENGTH = 1100

distance_positive_euc = []
distance_negative_euc = []

distance_positive_eucl2 = []
distance_negative_eucl2 = []

distance_positive_cos = []
distance_negative_cos = []
decisions = []

i = 0
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

def calculateFit(df, decision, model_name):
	"""
	temp_df1 = df[['values_positive']]
	temp_df1["Decision"] = "Yes"
	temp_df1 = temp_df1.rename({'values_positive': 'distance'}, axis='columns')

	temp_df2 = df[['values_negative']]
	temp_df2["Decision"] = "No"
	temp_df2 = temp_df2.rename({"values_negative": "distance"}, axis='columns')

	temp_df = pd.concat([temp_df1, temp_df2]).reset_index(drop = True)

	config = {'algorithm': 'C4.5'}
	model = chef.fit(temp_df, config)

	print(model)
	"""
	print(model_name)
	X = np.reshape(df, (-1, 1))
	y = np.reshape(decision, (-1, 1))
	clf = tree.DecisionTreeClassifier(random_state=0, max_depth=1)
	clf = clf.fit(X, y)
	print(clf.score(X, y))
	r = export_text(clf)
	print(r)

# import csv
# with open('pairsDevTrain.txt', 'r') as csvfile:
# 	trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

fig, axes = plt.subplots(1, 7)
fig.suptitle("Models")

import csv
with open('pairsDevTrainFull.txt', 'r') as csvfile:
 	trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

start = time.time()

for idx, model_name in enumerate(models):
	model = getModel(model_name)
	for row in tqdm.tqdm(trainrows):
		if i < SET_LENGTH:
			filename1 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[1]))
			filename2 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[2]))
		else:
			filename1 = "{0}/{1}_{2:04d}.jpg".format(row[0], row[0], int(row[1]))
			filename2 = "{0}/{1}_{2:04d}.jpg".format(row[2], row[2], int(row[3]))
		
		img1 = functions.preprocess_face(f"{DBPATH}{filename1}", target_size = getWindow(model_name), detector_backend='mtcnn', enforce_detection=False)
		img2 = functions.preprocess_face(f"{DBPATH}{filename2}", target_size = getWindow(model_name), detector_backend='mtcnn', enforce_detection=False)

		img1_embed = model.predict(img1)[0]
		img2_embed = model.predict(img2)[0]

		distance_euc = findDistance(img1_embed, img2_embed, metric='euclidean')
		distance_euc_l2 = findDistance(img1_embed, img2_embed, metric='euclidean_l2')
		distance_cos = findDistance(img1_embed, img2_embed, metric='cosine')
		if i < SET_LENGTH:
			distance_positive_cos.append(distance_cos)
			decisions.append(1)
			distance_positive_euc.append(distance_euc)
			distance_positive_eucl2.append(distance_euc_l2)
		else:
			distance_negative_cos.append(distance_cos)
			decisions.append(0)
			distance_negative_euc.append(distance_euc)
			distance_negative_eucl2.append(distance_euc_l2)
		i += 1
	
	
	

	d_cos = {"values_positive" : distance_positive_cos, "values_negative": distance_negative_cos}
	d_euc = {"values_positive" : distance_positive_euc, "values_negative": distance_negative_euc}
	d_eucl2 = {"values_positive" : distance_positive_eucl2, "values_negative": distance_negative_eucl2}

	distance_df_cos = pd.DataFrame(data = d_cos)
	distance_df_euc = pd.DataFrame(data = d_euc)
	distance_df_eucl2 = pd.DataFrame(data = d_eucl2)

	distance_df_cos.plot.kde(ax = axes[idx]).set_title(f'{model_name} cosine')
	#distance_df_euc.plot.kde(ax = axes[1]).set_title('euclidean')
	#distance_df_euc.plot.kde(ax = axes[2]).set_title('euclidean_l2')
	

	combined_dist_positive_cos = distance_positive_cos + distance_negative_cos
	combined_dist_positive_euc = distance_positive_euc + distance_negative_euc
	combined_dist_positive_eucl2 = distance_positive_eucl2 + distance_negative_eucl2
	calculateFit(combined_dist_positive_cos, decisions, model_name)
	#calculateFit(combined_dist_positive_euc, decisions)
	#calculateFit(combined_dist_positive_eucl2, decisions)
	#print(temp_df.head())

	distance_positive_cos.clear()
	distance_negative_cos.clear()

	distance_positive_euc.clear()
	distance_negative_euc.clear()

	distance_positive_eucl2.clear()
	distance_negative_eucl2.clear()

	decisions.clear()

	d_cos.clear()
	d_euc.clear()
	d_eucl2.clear()
	i = 0

end = time.time()
print(end - start)
plt.show()
