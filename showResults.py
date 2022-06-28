import os
from pathlib import Path
import gdown
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

values = [
    [0.89, 0.94, 0.95, 0.68, 0.62, 0.94, 0.62],
    [0.86, 0.91, 0.92, 0.65, 0.63, 0.90, 0.61], 
    [0.87, 0.93, 0.94, 0.68, 0.63, 0.93, 0.61], 
    [0.88, 0.93, 0.95, 0.67, 0.63, 0.94, 0.62], 
    [0.82, 0.84, 0.92, 0.69, 0.63, 0.76, 0.62], 
    [0.87, 0.92, 0.93, 0.68, 0.62, 0.93, 0.61], 
    [0.85, 0.87, 0.90, 0.68, 0.62, 0.88, 0.62], 
    [0.87, 0.88, 0.94, 0.66, 0.62, 0.93, 0.61], 
    [0.78, 0.54, 0.83, 0.63, 0.66, 0.67, 0.62], 
    [0.63, 0.59, 0.74, 0.64, 0.65, 0.54, 0.61], #arcface, recall 0.98 :o
]

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "DeepID"]
dbs = ["Original", "Compress 10", "Compress 20", "Compress 50", "Gaussian noise", 
    "Horizontal motion blur - window size 5", "Horizontal motion blur - window size 10",
    "Vertical motion blur - window size 5", "Vertical motion blur - window size 10", "S&P Noise"]

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


for idx, value in enumerate(values):
    x_pos = [i for i, _ in enumerate(models)]
    plt.figure(figsize=(11,5))
    plt.bar(x_pos, value, color='green')

    plt.xlabel("Model name")
    plt.ylabel("Accuracy [%]")
    plt.title(f"Database - {dbs[idx]}")
    
    addlabels(x_pos, value)

    plt.xticks(x_pos, models)

    plt.show()
