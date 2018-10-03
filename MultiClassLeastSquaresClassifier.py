import numpy as np
from sklearn import datasets
import pandas
import matplotlib

#url = r"C:\Users\Rahul\Documents\UFL\Courses\MIS\Homework4\iris.csv"
#names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
#dataframe = pandas.read_csv(url, names=names, engine='python')
#print(dataframe)
#dataframe_training = dataframe.iloc[:75,0:5]
#dataframe_test = dataframe.iloc[75:,0:4]
#print(dataframe_training)
#print(dataframe_test)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train = X[:75].T
X_test = X[75:]
Y_train = Y[:75]
#print(Y_train.shape)
Y_train_mat = np.zeros((3,75))
#print(X_train.shape)
#Convert single column of Y's to 3*75 matrix
for x in range(0,75):
    if Y_train[x]==0:
        Y_train_mat[0][x]=1
    elif Y_train[x]==1:
        Y_train_mat[1][x]=1
    elif Y_train[x]==2:
        Y_train_mat[2][x]=1
#print(Y_train_mat.shape)
Y_train_mat  = np.asmatrix(Y_train_mat)
#added row of ones at the end
X_train = np.asmatrix(np.vstack((X_train,np.ones((1,75)))))
#print((X_train[:,1]))
#print(np.transpose(Y_train_mat[:,1]))
xy = np.zeros((5,3))
#Finding sum of XY^t for all patterns
for i in range(0,75):
    xy = xy + np.matmul(X_train[:,i],(np.transpose(Y_train_mat[:,i])))
print(xy)

