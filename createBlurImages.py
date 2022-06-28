import cv2
import pytesseract
import os
import sys
import re
from PIL import Image
import PIL
import glob
import cv2
import numpy as np

path = "D:/Praca Magisterska/DB/motionblur_hor_10/"

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"

kernel_size = 10
kernel_v = np.zeros((kernel_size, kernel_size))
kernel_h = np.copy(kernel_v)
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
kernel_v /= kernel_size
kernel_h /= kernel_size

for i, root in enumerate(os.walk(DBPATH)):
    print(f"Root: {root[0]} ")
    if i > 0:
        directory_path = os.path.basename(os.path.normpath(root[0]))
        os.mkdir(f"{path}/{directory_path}")
        for filename in os.listdir(root[0]):
            picture = cv2.imread(f"{root[0]}/{filename}")
            horizontal_mb = cv2.filter2D(picture, -1, kernel_h)
            cv2.imwrite(f"{path}/{directory_path}/{filename}", horizontal_mb)