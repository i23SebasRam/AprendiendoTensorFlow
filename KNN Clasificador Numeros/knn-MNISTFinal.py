from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
#import pandas as pd
#from sklearn.linear_model import SGDClassifier
import cv2

path = "C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robotica/prueba.jpg"
img = cv2.imread(path)


mnist = fetch_openml('mnist_784', version=1)
x,y = mnist['data'],mnist['target']

xlist = x.values
y = y.astype(np.uint8)
ylist = y.values 
ylist = ylist.reshape(-1,1)

def hagoDosCifras(xlist,ylist,vecesIzquierda,porcen):
    Hx,Wx = xlist.shape
    #No necesitamos la de y porque son las mismas dimensiones
    cantDatos = int(Hx*porcen)
    xlist = xlist[:cantDatos]
    #Me deberia de tirar la fila desde cero hasta la cantidad de datos que queremos.
    ylist = ylist[:cantDatos]
    #Me deberia de tirar los labels de la misma cantidad.
    xNuevo = np.zeros((cantDatos*vecesIzquierda,28*56),dtype=np.uint8)
    yNuevo = np.zeros((cantDatos*vecesIzquierda,11),dtype=np.uint8)
    
    #La ultima columna de yNuevo es cuando los dos numeros son iguales.
    for i in range(vecesIzquierda):
        numIzquierda = xlist[i].reshape(28,28)
        for j in range(cantDatos):
        #Aca tomo el numero de la izquierda
            numDerecha = xlist[j].reshape(28,28)
            xSemiNuevo = np.concatenate((numIzquierda,numDerecha),axis=1)
            #xSemiNuevo[28,56]
            xNuevo[j+cantDatos*i] = xSemiNuevo.reshape(1,-1)
            if ylist[i] == ylist[j]:
                #Si son el mismo numero.
                yNuevo[j+cantDatos*i,ylist[i]] = 1
                yNuevo[j+cantDatos*i,10] = 1
            else:
                yNuevo[j+cantDatos*i,ylist[i]] = 1
                yNuevo[j+cantDatos*i,ylist[j]] = 1
    return xNuevo,yNuevo

x2,y2 = hagoDosCifras(xlist,ylist,30,0.05)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x2,y2,test_size=0.2,random_state=42)

xtrain,xtest,ytrain,ytest = xtrain/255,xtest/255,ytrain,ytest

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=39,p=2,weights='distance')
knn_clf.fit(xtrain,ytrain)

import time
inicio=time.time()
knn_clf.predict(xtest[7].reshape(1,-1))
fin=time.time()
print(fin-inicio)
#0.30017
#Probar el modelo
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(knn_clf,xtrain,ytrain,cv=3)

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score

y_predict = np.zeros((1000,11))
for i in range(1000):
    y_predict[i,:] = knn_clf.predict(xtest[i].reshape(1,-1))


f1_score(ytest[:1000],y_predict,average='samples')
#0.936633
accuracy_score(ytest[:1000],y_predict,normalize=False)
#0.835



#Otro modelo

knn_clf2 = KNeighborsClassifier(n_neighbors=39,p=2)
knn_clf2.fit(xtrain,ytrain)

import time
inicio=time.time()
knn_clf2.predict(xtest[7].reshape(1,-1))
fin=time.time()
print(fin-inicio)
#0.2858

y_predict2 = np.zeros((1000,11))
for i in range(1000):
    y_predict2[i,:] = knn_clf2.predict(xtest[i].reshape(1,-1))
    

f1_score(ytest[:1000],y_predict2,average='samples')
#0.8718333
accuracy_score(ytest[:1000],y_predict2,normalize=False)
#0.644


#Image
img = "C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robotica/prueba8.jpg"
img = cv2.imread(img)
img2 = cv2.resize(img,(28,56),interpolation=cv2.INTER_AREA)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
norm_img = np.zeros((10,10))
(thresh, blackAndWhiteImage) = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)
th3 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
final_img = cv2.normalize(img2, norm_img,0,255, cv2.NORM_MINMAX)

cv2.imshow('cosita',img2)
cv2.imwrite("C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robotica/prueba2.png",th3)
cv2.waitKey(0)

knn_clf.predict(th3.reshape(1,-1))


captura = cv2.VideoCapture('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robotica/96.mp4')
while (captura.isOpened()):
  contador = 59
  ret, imagen = captura.read()
  if ret == True:
    contador = contador + 1
    if contador == 60:
        #imagen2 = cv2.flip(imagen,-1)
        imagen2 = cv2.resize(imagen,(56,28),interpolation=cv2.INTER_AREA)
        imagen2 = cv2.cvtColor(imagen2,cv2.COLOR_BGR2GRAY)
        imagen2 = cv2.adaptiveThreshold(imagen2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
        txt = knn_clf.predict(imagen2.reshape(1,-1))
        cv2.putText(imagen,str(txt),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        contador = 0
    cv2.imshow('video', imagen2)
    if cv2.waitKey(30) == ord('s'):
      break
  else: break
captura.release()
cv2.destroyAllWindows()

import matplotlib.pyplot as pt
cosa = np.apply_along_axis(sum,0,ytrain)
_ = plt.hist(cosa,bins='auto')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['0','1','2','3','4','5','6','7','8','9','10']
students = [23,17,35,29,12]
ax.bar(langs,cosa)
plt.show()
