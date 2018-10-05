import numpy as np
from sklearn import datasets

#Load the dataset
iris = datasets.load_iris()
X = (iris.data)
Y = (iris.target)
for i in [12,30,50]:
    #fix the training set percentage
    training_percentage = i
    train_values = (training_percentage / 100) * 50

    #Divide X and y values based on training percentage
    X_test = np.vstack((X[int(train_values):50,:],X[50+int(train_values):100,:],X[100+int(train_values):150,:]))
    X_train = np.transpose(np.vstack((X[:int(train_values),:],X[50:50+int(train_values),:],X[100:100+int(train_values),:])))
    Y_test = np.hstack((Y[int(train_values):50],Y[50+int(train_values):100],Y[100+int(train_values):150]))
    Y_train = np.hstack((Y[:int(train_values)],Y[50:50+int(train_values)],Y[100:100+int(train_values)]))

    #Convert single column of Y's to 3*75 matrix
    Y_train_mat = np.zeros((3,int(train_values)*3))
    for x in range(0,int(train_values)*3):
        if Y_train[x]==0:
            Y_train_mat[0][x]=1
        elif Y_train[x]==1:
            Y_train_mat[1][x]=1
        elif Y_train[x]==2:
            Y_train_mat[2][x]=1
    Y_train_mat  = np.asmatrix(Y_train_mat)

    #Add row of ones at the end of training array
    X_train = np.asmatrix(np.vstack((X_train,np.ones((1,int(train_values)*3)))))

    #Finding sum of XY^t for all patterns
    xyt = np.zeros((5,3))
    for i in range(0,int(train_values)*3):
        xyt = xyt + np.matmul(X_train[:,i],(np.transpose(Y_train_mat[:,i])))

    #Finding inverse of sum of x.x^t for all input patterns
    xxt = np.zeros((5,5))
    for i in range(0,int(train_values)*3):
        xxt = xxt + np.matmul(X_train[:,i],np.transpose(X_train[:,i]))

    #Computing XXT + lamda
    lamda = np.diagflat([0.001,0.001,0.001,0.001,0.001])
    xxt_inv = np.linalg.inv(xxt+lamda)

    #find weight matrix
    W = np.matmul(xxt_inv,xyt)#5X3
    W1 = W[:,0]#5X1
    W2 = W[:,1]
    W3 = W[:,2]

    #Testing
    X_test_rows,X_test_col = X_test.shape
    Y_test_rows = Y_test.size
    out = np.zeros((3))
    classifier_pred = np.zeros((Y_test_rows))
    #Calculate WTX and find maximum output of the three W's
    for i in range(0, X_test_rows):
        X_test_1 = np.transpose(np.asmatrix(np.hstack((X_test[i,:],1))))
        out[0] = np.matmul(np.transpose(W1),X_test_1)
        out[1] = np.matmul(np.transpose(W2),X_test_1)
        out[2] = np.matmul(np.transpose(W3),X_test_1)
        max = np.argmax(out)
        classifier_pred[i] = int(max)

    #Computing confusion matrix
    confusion_mat = np.zeros((3,3))
    for i in range(0,Y_test_rows):
        confusion_mat[int(Y_test[i])][int(classifier_pred[i])] = confusion_mat[int(Y_test[i])][int(classifier_pred[i])] + 1;
    print(confusion_mat)