import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression
    In this problem, you will implement the linear regression method based upon gradient descent.
    Xw  = y
    You could test the correctness of your code by typing `nosetests -v test.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
'''

#--------------------------
def compute_Phi(x,p):
    '''
        Compute the feature matrix Phi of x. We will construct p polynoials, the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    length_x = len(x)
    Phi = np.mat(np.zeros([length_x, p]))
    for i in range(p):
        for j in range(length_x):
            s = x[j]
            Phi[j, i] = math.pow(s,i)


    #########################################
    return Phi 

#--------------------------
def compute_yhat(Phi, w):
    '''
        Compute the linear logit value (predicted value) of all data instances. z = <w, x>
        Here <w, x> represents the dot product of the two vectors.
        Input:
            Phi: the feature matrix of all data instance, a float numpy matrix of shape n by p. 
            w: the weights parameter of the linear model, a float numpy matrix of shape p by 1. 
        Output:
            yhat: the logit value (predicted value) of all instances, a float numpy matrix of shape n by 1
        Hint: you could solve this problem using 1 line of code. Though using more lines of code is also okay.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    yhat = np.dot(Phi, w)

    #########################################

    return yhat

    #--------------------------
def compute_L(yhat,y):
    '''
        Compute the loss function: mean squared error. In this function, divide the original mean squared error by 2 for making gradient computation simple. Remember our loss function in the slides.  
        Input:
            yhat: the predicted sample labels, a numpy vector of shape n by 1.
            y:  the sample labels, a numpy vector of shape n by 1.
        Output:
            L: the loss value of linear regression, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    summ = 0
    a = len(y)
    for i in range(0,a):
        diff = yhat[i] - y[i]
        squ_diff = diff ** 2
        summ = summ + squ_diff
    L = summ/(2*a)


    #########################################
    return L 



def compute_dL_dw(y, yhat, Phi):
    '''
        Compute the gradients of the loss function L with respect to (w.r.t.) the weights w. 
        Input:
            Phi: the feature matrix of all data instances, a float numpy matrix of shape n by p. 
               Here p is the number of features/dimensions.
            y: the sample labels, a numpy vector of shape n by 1.
            yhat: the predicted sample labels, a numpy vector of shape n by 1.
        Output:
            dL_dw: the gradients of the loss function L with respect to the weights w, a numpy float matrix of shape p by 1. 

    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dL_dw = np.mat(np.zeros([np.shape(Phi)[1],1]))
    for i in range(np.shape(Phi)[1]):
        for j in range(len(y)):
            y_pred = y[j] 
            dL_dw[i, 0] = dL_dw[i, 0] + Phi[j, i] * (yhat[j] - y_pred) 
            
        dL_dw[i, 0] = dL_dw[i, 0]/len(y)
        
    

    #########################################
    return dL_dw


#--------------------------
def update_w(w, dL_dw, alpha = 0.001):
    '''
       Given the instances in the training data, update the weights w using gradient descent.
        Input:
            w: the current value of the weight vector, a numpy float matrix of shape p by 1.
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a numpy float matrix of shape p by 1. 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a numpy float matrix of shape p by 1.
        Hint: you could solve this problem using 1 line of code
    '''
    
    #########################################
    ## INSERT YOUR CODE HERE
    w = w - alpha * dL_dw


    #########################################
    return w


#--------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
       Given a training dataset, train the linear regression model by iteratively updating the weights w using the gradient descent
        We repeat n_epoch passes over all the training instances.
        Input:
            X: the feature matrix of training instances, a float numpy matrix of shape (n by p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer matrix of shape n by 1. 
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a numpy float matrix of shape p by 1. 
    '''

    # initialize weights as 0
    w = np.mat(np.zeros(X.shape[1])).T
    
    #Phi = X
    for _ in range(n_epoch):
        
    #########################################
    ## INSERT YOUR CODE HERE
       yhat = compute_yhat(X, w)
       dL_dW =compute_dL_dw(Y, yhat, X)
       w = update_w(w, dL_dW, alpha)
       
    # Back propagation: compute local gradients 
    
        

        
        
    # update the parameters w
        

     #########################################
    return w


