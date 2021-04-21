import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data(housing_path=HOUSING_PATH)

# Este es un plot basico.
"""
housing.plot(kind = "scatter", x = "longitude", y = "latitude",alpha = 0.1)
"""

# Este es un plot de alguien pro.

"""
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
plt.show()
"""


# Ahora es un analisis de la correlacion entre los atributos que tenemos.

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Se hace un scatter de diferentes correlaciones, especialmente de aquellas que tenian mas correlacion.

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

# Se crean nuevos atributos al sistema con los datos que ya tenemos.

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

