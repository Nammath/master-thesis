import cv2
import pytesseract
import os
import sys
import re
from PIL import Image
import PIL
import glob

path = "D:/Praca Magisterska/DB/compress10/"

DBPATH = "D:/Praca Magisterska/DB/lfw/lfw/"


for i, root in enumerate(os.walk(DBPATH)):
    print(f"Root: {root[0]} ")
    if i > 0:
        directory_path = os.path.basename(os.path.normpath(root[0]))
        os.mkdir(f"{path}/{directory_path}")
        for filename in os.listdir(root[0]):
            picture = Image.open(f"{root[0]}/{filename}")
            picture.save(f"{path}/{directory_path}/{filename}", quality=10)