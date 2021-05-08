from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
import cv2

mnist = fetch_openml('mnist_784', version=1)
x,y = mnist['data'],mnist['target']

xlist = x.values
y = y.astype(np.uint8)
ylist = y.values 
ylist = ylist.reshape(-1,1)

def hagoDosCifras(xlist,ylist,vecesIzquierda,porcen):
    Hx = xlist.shape
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

x2,y2 = hagoDosCifras(xlist,ylist,20,0.4)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x2,y2,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(xtrain,ytrain)