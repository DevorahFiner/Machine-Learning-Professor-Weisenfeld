#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Your assignment is to complete the KNN class below.

You are free to add any additional class methods or optional parameters,
but do not change the required argument k.

This method should use euclidean distance to find neighbors 
and the mean of y values of neighbors to to predict.

The only imported library you can use in this class is `numpy` aliased as `np`. 
Do not write this import yourself, the testing module will do that import.
Once done, copy and paste the class in this cell into a textfile and save it as: 
"knn_{college_id_num}.py" (replace {college_id_num} with your college id number) and submit.
"""

class KNN:
    def __init__(self, k: int):
        self.k = k # number of neighbors to use, don't change this
    
    def fit(self, X, y):
        """
        This method gives the KNN class enough information to be able
        to predict an unknown X parameter (provided in the `predict` method)
        using the data contained in the training `X` and `y` parameters 
        of this method. 
        
        The input parameters of this method are
            X: the features, a two-dimensional array-like object with shape [n, d]
            y: the target, a one dimensional array-like object with length n
        
        Remember to convert `X` and `y` to a numpy array.
        
        (Do not try to do the actual prediction here)
        """
        self.X = np.array(X)  # convert X and y to numpy arrays
        self.y = np.array(y)

        return self # returning self is convenient when chaining methods
    
    def predict(self, X):
        """
        This method takes a single parameter
            X: features, a two-dimensional array-like object with shape [m, d]
        
        Output: a numpy array of length m with the KNN prediction for each row of `X`
        
        Remember to convert `X` to a numpy array.
        """
        # type your code here:
           
        #Basic Steps:
            # for each row in `X`, find the k nearest rows in `self.X`
            # get the `self.y` values that correspond to those nearest neighbors
            # compute the mean of those selected values, and collect in a container
            # output a numpy array containing those means for each row of `X`
            
        X = np.array(X)  # convert X to a numpy array
        y_pred = []  # to store predictions
        
        for x in X:
            # compute Euclidean distances between x and all points in self.X
            distances = np.sqrt(np.sum((self.X - x)**2, axis=1))
            
            # find indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            
            # get corresponding y values of nearest neighbors
            nearest_y = self.y[nearest_indices]
            
            # compute mean of nearest y values and append to predictions
            y_pred.append(np.mean(nearest_y))
        
            
        return np.array(y_pred)  # you should return a numpy array containing a prediction for each y
    
    def score(self, X, y): # Outputs the R^2 of the prediction. Don't change this method.
        y_pred = self.predict(X) # get the prediction
        y_true = np.array(y)
        return 1 - ((y_true - y_pred)** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

