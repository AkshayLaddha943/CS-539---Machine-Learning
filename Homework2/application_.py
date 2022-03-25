import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
loss = None
yhat_test = None
yhat_train = None
dL_dW = None
z = []
#w = np.mat(np.zeros(X.shape[1]))
#w = w.T
def epoch_result(alpha, n_epoch):
    w = train(Xtrain, Ytrain, alpha, n_epoch)
    yhat_train = compute_yhat(Xtrain, w)
    Ltrain = compute_L(yhat_train, Ytrain)
    z.append(Ltrain)
    yhat_test = compute_yhat(Xtest, w)
    Ltest = compute_L(yhat_test, Ytest)
    print("With alpha: ", alpha)
    print("Training data loss is:", Ltrain)
    print("Test data loss: ", Ltest)
    print("epoch count: ", n_epoch, '\n')


#########################################

#keeping alpha = 0.25
epochs = range(50,200,25)

for i in epochs:
    epoch_result(alpha = 0.25, n_epoch=i)
    
  
for k in np.arange(0.1, 0.8, 0.1):
    epoch_result(alpha = k, n_epoch=100)
