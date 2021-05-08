from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
import cv2

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
x,y = mnist['data'],mnist['target']

xlist = x.values
y = y.astype(np.uint8)
ylist = y.values 
ylist = ylist.reshape(-1,1)

datos = np.append(xlist,ylist,axis = 1)
datospd = pd.DataFrame(data=datos)
datospd.to_csv("D:/AprendiendoTensorFlow/KNN Clasificador Numeros/datosMNIST.csv")

digito = xlist[0,:]
digito = digito.reshape(28,28)

plt.imshow(digito,cmap="binary")
plt.axis("off")
plt.show()

x_train,x_test,y_train,y_test = xlist[:60000],xlist[60000:],y[:60000],y[60000:]
x_train,x_test,y_train,y_test = x[:60000],x[60000:],y[:60000],y[60000:]
#Para un modelo de clasificacion binario
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train_5)

sgd_clf.predict(x.values[1].reshape(1,-1))


from sklearn.neighbors import KNeighborsClassifier
#Clasificador multilabel

img = cv2.imread('C:/Users/pc/Pictures/prueba2.jpeg')
imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBW = 255 - imgBW
(tr,imgBW) = cv2.threshold(imgBW,127, 255, cv2.THRESH_BINARY)
imgBW = imgBW.reshape(1,-1)

y_train_0 = (y_train==0)
y_train_1 = (y_train==1)
y_train_2 = (y_train==2)
y_train_3 = (y_train==3)
y_train_4 = (y_train==4)
y_train_5 = (y_train==5)
y_train_6 = (y_train==6)
y_train_7 = (y_train==7)
y_train_8 = (y_train==8)
y_train_9 = (y_train==9)
y_train_large = np.c_[y_train_0,y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6,y_train_7,y_train_8,y_train_9]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train,y_train_large)

knn_clf.predict(x.values[0].reshape(1,-1))
knn_clf.predict(imgBW)


#Juntar dos imagenes
total = np.zeros((28,56))
izquierda = x.values[0].reshape(28,28)
derecha = x.values[1].reshape(28,28)
#derecha = np.roll(derecha,-2)
total = np.concatenate((izquierda,derecha),axis=1)
total1 = total.reshape(1,-1)
total2 = total1.reshape(28,56)

plt.imshow(total2,cmap="binary")
plt.axis("off")
plt.show()

label = np.zeros((9))
label[y[0]] = 1
label[y[1]] = 1

## Quiero hacer una funcion que me expanda los datos para entrenar el modelo, con datos de dos cifras

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

pruebaX,pruebaY = hagoDosCifras(xlist,ylist,5,0.001)

prueba = pruebaX[348].reshape(28,56)
print(pruebaY[348])
plt.imshow(prueba,cmap="binary")
plt.axis("off")
plt.show()

pruebaX,pruebaY = hagoDosCifras(xlist,ylist,20,0.4)

