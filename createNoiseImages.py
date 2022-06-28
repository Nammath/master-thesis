import cv2
import pytesseract
import os
import sys
import re
from PIL import Image
import PIL
import glob
import numpy as np
from skimage.util import random_noise

path = "D:/Praca Magisterska/DB/snp/"

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"


for i, root in enumerate(os.walk(DBPATH)):
    print(f"Root: {root[0]} ")
    if i > 0:
        directory_path = os.path.basename(os.path.normpath(root[0]))
        os.mkdir(f"{path}/{directory_path}")
        for filename in os.listdir(root[0]):
            picture = cv2.imread(f"{root[0]}/{filename}")
            noise_img = random_noise(picture, mode='s&p', seed=None, amount=0.1)
            noise_img = np.array(255*noise_img, dtype = 'uint8')
            #img = Image.fromarray(noise_img)
            #img.save(f"{path}/{directory_path}/{filename}")
            cv2.imwrite(f"{path}/{directory_path}/{filename}", noise_img)