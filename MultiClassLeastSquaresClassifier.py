import numpy as np
from sklearn import datasets
import matplotlib

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train = X[:75].T
X_test = X[75:]
Y_train = Y[:75]
#print(Y_train)
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
xyt = np.zeros((5,3))


#Finding sum of XY^t for all patterns
for i in range(0,75):
    xyt = xyt + np.matmul(X_train[:,i],(np.transpose(Y_train_mat[:,i])))
#print("xyt = ")
#print(xyt)

# finding inverse of sum of x.x^t for all input patterns
xxt = np.zeros((5,5))
for i in range(0,75):
    xxt = xxt + np.matmul(X_train[:,i],np.transpose(X_train[:,i]))
#print("xxt = ")
#print(xxt)
a = np.diagflat([0.001,0.001,0.001,0.001,0.001])
#print("a = ")
#print(a)
xxt_inv = np.linalg.inv(xxt+a)
#print("xxt_inv = ")
#print(xxt_inv)

#find weight matrix
W = np.matmul(xxt_inv,xyt)#5X3
#print("W = ")
#print(W)
W1 = W[:,0]#5X1
W2 = W[:,1]
W3 = W[:,2]
#print("X_test = ")
#print(X_test)#75X4
out = np.zeros((3))

#Testing
X_test_1 = np.transpose(np.asmatrix(np.hstack((X_test[1,:],1))))
#print(X_test_1)
#print(X_test_1.shape)
out[0] = np.matmul(np.transpose(W1),X_test_1)
out[1] = np.matmul(np.transpose(W2),X_test_1)
out[2] = np.matmul(np.transpose(W3),X_test_1)
print(out)