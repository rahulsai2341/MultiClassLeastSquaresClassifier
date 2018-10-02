import numpy as np
import sklearn
import pandas
import matplotlib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataframe = pandas.read_csv(url, names=names, engine='python')
print(dataframe)
