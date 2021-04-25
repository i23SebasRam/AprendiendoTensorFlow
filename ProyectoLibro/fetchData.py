import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

#Paths en donde se encuentran los datos.
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#FUncion que descarga los datos y despues los agrega a la carpeta que ya hay.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
"""
fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
"""
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

# Para cuando hace falta un valor podemos usar librerias de sklearn

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis = 1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index = housing_num.index)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
            
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#Antes de iniciar a entrenar modelos, es comun realizar una normalizacion o una
#estandarizacion porque los datos pueden tener un escalamiento distinto.

#Normalizacion: los datos quedan de 0 a 1.
#Estandarizacion: le resta el promedio y despues lo divide por la desviacion estandar.


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler',StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)



#Creo que esto lo que hace es organizar los datos tales como los necesito,
#es decir que haga el oneHotcode para las que son texto y los numeros.

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
                ("num", num_pipeline,num_attribs),
                ("cat",OneHotEncoder(),cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)

#Y ahora nos mandamos nuestra regresion lineal con los datos que tenemos

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)




