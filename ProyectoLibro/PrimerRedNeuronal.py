# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,Y_train_full),(X_test,Y_test) = fashion_mnist.load_data()

X_val, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
Y_val, Y_train = Y_train_full[:5000], Y_train_full[5000:]

class_names = ["T-Shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle_boot"]

model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28,28]), keras.layers.Dense(300,activation="relu"),keras.layers.Dense(100,activation="relu"), keras.layers.Dense(10,activation="softmax")])


model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_val,Y_val))


model.evaluate(X_test,Y_test)
