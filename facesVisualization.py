import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

databases = ["D:/Praca Magisterska/DB/lfw/lfw/", "D:/Praca Magisterska/DB/compress10/", "D:/Praca Magisterska/DB/compress20/", 
            "D:/Praca Magisterska/DB/compress50/", "D:/Praca Magisterska/DB/gauss/", 
            "D:/Praca Magisterska/DB/motionblur_hor_5/", "D:/Praca Magisterska/DB/motionblur_hor_10/", 
			"D:/Praca Magisterska/DB/motionblur_ver_5/", "D:/Praca Magisterska/DB/motionblur_ver_10/", "D:/Praca Magisterska/DB/snp/"]

names = ["Original database", "Compression ratio = 10", "Compression ratio = 20", "Compression ratio = 50", 
        "Gaussian noise", "Horizontal motion blur, window size = 5", "Horizontal motion blur, window size = 10",
        "Vertical motion blur, window size = 5", "Vertical motion blur, window size = 10", "S&P noise"]

filename1 = "George_W_Bush/George_W_Bush_0067.jpg"
filename2 = "Robert_Downey_Jr/Robert_Downey_Jr_0001.jpg"
filename3 = "Zinedine_Zidane/Zinedine_Zidane_0006.jpg"

for i, database in enumerate(databases):
    picture1 = mpimg.imread(f"{database}/{filename1}")
    picture2 = mpimg.imread(f"{database}/{filename2}")
    picture3 = mpimg.imread(f"{database}/{filename3}")
    fig = plt.figure()
    fig.suptitle(names[i], fontsize=16)
    ax = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(picture1)
    ax.set_xlabel("George W Bush")
    ax = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(picture2)
    ax.set_xlabel("Robert Downey Jr")
    ax = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(picture3)
    ax.set_xlabel("Zinedine Zidane")
    plt.show()