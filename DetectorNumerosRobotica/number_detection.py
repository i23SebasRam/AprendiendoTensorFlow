
import tensorflow as tf # Se requiere instalar tensorflow
import cv2
import numpy as np
import matplotlib.pyplot as plt


# La siguiente función permite ingresar una imagen RGB de cualquier tamaño como primer parámetro, y el 
# segundo paraámetro corresponde al modelo de clasificación.
# La función retorna el número que contiene la imagen.

# Para resultados óptimos el número que contiene la imagen de entrada de bes negro y con fondo blanco.

def number_recognition(image, model):
	image = cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
	if len(np.shape(image))>2:
	  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = image/255
	#image = (image-1)*-1
	#image = image>0.8
	image = image.astype(np.float64)
	cv2.imshow('number vision', image)
	cv2.waitKey(3)
	image = np.expand_dims(image,axis=0)
	image = np.expand_dims(image,axis=-1)
	prediction = np.argmax(model.predict([image]))
	return prediction



# Ejemplo:

# Para importar el modelo de detección.

path_model = 'D:/AprendiendoTensorFlow/DetectorNumerosRobotica/number_detector.h5' # Ingresar el path del modelo.

model = tf.keras.models.load_model(path_model)

path = "C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robotica/prueba8.jpg"

Imagen = cv2.imread(path) # Se carga la imagen a detectar
plt.imshow(Imagen)

numero = number_recognition(Imagen, model) # Se utiliza la función.
print(numero)
