import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

#Paths en donde se encuentran los datos.
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("D:/","AprendiendoTensorFlow","datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#FUncion que descarga los datos y despues los agrega a la carpeta que ya hay.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
#fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)

#Hacemos una funcion que nos lee el dataset en csv y lo pasa a un dataframe(pandas)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data(housing_path=HOUSING_PATH)

#El histograma de todo los datos que tenemos.
"""
housing.hist(bins = 50, figsize=(20,15))
plt.show()
"""

#Una forma de hacer un split de los datos que tenemos, es decir separar los datos para entrenamiento y test.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)

#Aca lo que hacemos es una subdivision de los datos que tenemos, para entender las proporciones de ese label.
#Las dividimos en 5 clasificaciones.
housing["income_cat"] = pd.cut(housing["median_income"],bins = [0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
#housing["income_cat"].hist()
#plt.show()

#Se realiza el split con ayuda de sklearn, y asi se respeta la misma proporcionalidad de los datos en los dos test set, como deberia de ser.
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
"""
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
"""

#Devolvemos al estado original los datos
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace = True)

#Vamos a iniciar el proceso de machine learning

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Usamos diferentes alternativas para cuando un atributo le hacen falta datos
#Alternativa 1 - Quitar los distritos a los cuales les falte ese atributo.
"""
housing.dropna(subset=["total_bedrooms"])
"""

#Alternativa 2 - Quitar todo el atributo
"""
housing.drop("total_bedrooms",axis=1)
"""

#Alternativa 3 - Completarlos con el promedio de los datos

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace = True)


#El siguiente paso es volver los datos que son textuales a numericos, porque
#asi funciona machine learning, para esto tenemos dos formas de hacer las cosas


#Primer forma - la hacemos dependiendo segun cuantas clasificaciones hayan
# el problema es que los modelos normalmente entre mas distanciados estan los
#numeros quiere decir que son menos semejantes entre si.

from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing[["ocean_proximity"]]
ordinal_enconder = OrdinalEncoder()
housing_cat_encoded = ordinal_enconder.fit_transform(housing_cat)
"""
print(housing_cat_encoded[:10])  
"""

#Ahora usamos otro metodo que se llama onehotencoder en donde lo que hace es que
#crea un vector con la cantidad total de clasificaciones que hay de ese atributo
#y procede a poner en uno el valor que le corresponda y cero los otros

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()
print(housing_cat_1hot)
