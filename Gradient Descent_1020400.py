#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Your assignment is to complete the LogisticRegression class below using Gradient Descent.

You are free to add any additional class methods or optional parameters,
but do not change the arguments that have been pre added.

Once done, copy and paste the class in this cell into a textfile and save it as: 
"gd_{college_id_num}.py" (replace {college_id_num} with your college id number) and submit.
For example, if your college id is 1234567, you should name your file gd_1234567.py
    - Do NOT inlcude the curly braces: {}
    - DO include the underscore: _
    - Do NOT capitalize gd
    - DO ensure that you are saving as a .py file
    - ONLY copy the text from this cell, DO NOT copy the text from other cells
    - DO test your code to ensure it works before you submit it.
"""
import numpy as np

class LogisticRegression:
    def __init__(self, max_iter = 50000, lr = 0.01, fit_intercept = True):
        self.max_iter = max_iter
        self.lr = lr
        self.fit_intercept = fit_intercept
    
    def sigmoid(self, x): 
        """
        I've completed this method for you. It takes in a value x and computes 
        the sigmoid function: f(x) = 1/(1 + exp(-x)) = exp(x)/(1 + exp(x)).
        Note that there are two equivalent forms of the function. If x is
        positive, the first form is more compuationally stable, and if x is 
        negative the second one is more computationally stable.
        Note also that this method expects x as a numpy array, and returns 
        an equal sized array with the sigmoid of each element of x.
        """
        pos, neg = x >= 0, x < 0
        sig = np.empty_like(x, dtype = np.float64)
        sig[pos] = 1 / (1 + np.exp(-x[pos]))
        sig[neg] = np.exp(x[neg])/(1 + np.exp(x[neg]))
        return sig
    
    def logit(self, X):
        """
        Return an array of logits for each row of X.
        
        The logit of each row of X is the dot product of that row and the weights (coefficients)
        We haven't yet created the weight variable, so when writing this method,
        assume that there is a class attribute called `self.w` that is a single dimension
        array with a weight corresponding to each column of X, plus one extra weight in the
        beginning if self.fit_intercept is True.
        """
        
        #  1. Convert X to a numpy array
        
        #  2. Check if the width of X is less than the length of self.w.
        #     If so, add a column of 1's as the first column of X
        
        #  3. Return the array of logits
        
        X = np.array(X)
        if X.shape[1] < len(self.w):
            X = self.add_intercept(X)
        return np.dot(X, self.w)
    
    def prob(self, X):
        #  Return an array of probabilities that y_hat = 1 for
        #  each row of X. You can do this by computing the sigmoids 
        #  of the logits of each row of X
        return self.sigmoid(self.logit(X))
    
    def pred(self, X):
        #  Return an array of predictions of y for each row of X
        #  We predict a row as 1 if the prob that it is 1 is strictly 
        #  greater than 0.5, otherwise we predict 0
        return (self.prob(X) > 0.5).astype(int)
    
    def logloss(self, X, y):
        #  Return the average of the log-loss of all rows of X and y.
        #  log loss of row_i = -y_i * log(prob(X_i)) - (1 - y_i) * log(1 - prob(X_i))
        y_hat = self.prob(X)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def grad(self, X, y):
        #  Return the gradient of the log loss with respect to each weight
        #  This will be an array of the same length as self.w.
        #  Don't worry about the intercept, because we're treating that as a regular 
        #  coeficient because we added a 1 column if self.add_intercept is True
        #  The formula for the partial derviatve for each w_j is the dot product
        #  of the j'th column of X and (prob(X) - y) you should divide this result 
        #  by the count of rows in X
        y_hat = self.prob(X)
        return np.dot(X.T, (y_hat - y)) / len(X)
    
    def add_intercept(self, X):
        #  Use this method to add an column of 1's as the first column of X
        # if self.fit_intercept is True. Be careful not to do this twice to the same X!
        if self.fit_intercept:
            return np.column_stack((np.ones(len(X)), X))
        return X

    def accuracy(self, X, y):
        # I filled this out for you! It returns the accuracy.
        return (self.pred(X) == y).mean()
    
    def fit(self, X, y):
        """
        This is the meat of this class. Here you will fit a Logistic Regression
        with the help of the previous methods, Gradient Descent, and your brains!
        """
        #  Filled this out for you we'll use these to collect
        #  losses and accuracies at each epoch
        self.losses, self.accuracies = [], []
        
        #  1. Make sure X and y are numpy arrays, and add add a column of
        #     ones as the first column of X if self.fit_intercept = True
        
        #  2. Initialize self.w as an array of 0's, there should be one element
        #     each column of X
        
        # 3. Write a loop to iterate as many times as self.max_iter
        #    In each iteration of the loop, compute the loss and accuracy and 
        #    append them to the self.losses and self.accuracies lists.
        #    Compute the gradient and decrement self.w by the product of
        #    the gradient and self.lr
        
        X = np.array(X)
        X = self.add_intercept(X)
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            loss = self.logloss(X, y)
            acc = self.accuracy(X, y)
            self.losses.append(loss)
            self.accuracies.append(acc)
            gradient = self.grad(X, y)
            self.w -= self.lr * gradient
        
        return self
    

