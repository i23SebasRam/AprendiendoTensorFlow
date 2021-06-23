#La gracia de este codigo es generar nombres aleatorios y despues guardarlos en otra carpeta.

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import tensorflow

path_base = "D:/RosBag/RosBag/imagenes"
img = Image.open(path_base + '/img000001.jpg')
imgplt = plt.imshow(img)
#plt.show()

from tensorflow import keras
from tensorflow.keras.preprocessing.image import DirectoryIterator,ImageDataGenerator





#image_data_generator = ImageDataGenerator()
#ImagenesSelec = image_data_generator.flow_from_directory(directory = path_base,shuffle = True, seed = 42, batch_size = 256,)
#ImagenesSelec = DirectoryIterator(directory = path_base, image_data_generator, shuffle = True, seed = 42, batch_size = 256)

pathData = "D:/RosBag/RosBag/dataSet"


import glob 
import random
import os
import shutil

cosa = glob.glob("D:/RosBag/RosBag/imagenes/*[0-1000].*")
cosita = random.sample(cosa,250)

for i in cosita:
    shutil.copy(i,pathData)





