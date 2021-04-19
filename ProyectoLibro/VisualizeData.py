import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

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
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# Ahora es un analisis de la correlacion entre los atributos que tenemos.

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
