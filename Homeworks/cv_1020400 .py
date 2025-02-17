#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Your assignment is to complete the CV class below.

You are free to add any additional class methods or optional parameters,
but do not change the required argument k and model.

The only imported library you can use in this class is `numpy` aliased as `np`. 
Once done, copy and paste the class in this cell into a textfile and save it as: 
"cv_{college_id_num}.py" (replace {college_id_num} with your college id number) and submit.
For example, if your college id is 1234567, you should name your file cv_1234567.py
    - Do NOT inlcude the curly braces: {}
    - DO include the underscore: _
    - Do NOT capitalize cv
    - DO ensure that you are saving as a .py file
    - ONLY copy the text from this cell, DO NOT copy the text from other cells
    - DO test your code to ensure it works before you submit it.
"""
import numpy as np  # Leave this in place

class CV:
    # Do not change the input parameters, but feel free to add
    def __init__(self, k: int, model, shuffle: bool=True, random_seed = None): 
        self.k = k  # int: number of folds
        self.model = model  # instantiated model object with `fit`, `predict`, and `score` methods
        self.shuffle = shuffle  # bool: determines whether to shuffle samples before cross-validating
        self.random_seed = random_seed  # int or None (default): ensures your model returns the same results as mine
    
    def fit(self, X, y):
        """
        This method creates the k-folds, does the training and scoring for each fold,
        saves the model scores in an attribute list and then refits the model on the entire dataset.
        
        The input parameters of this method are
            X: the features, a two-dimensional array-like object with shape [n, d]
            y: the target, a one dimensional array-like object with length n
        
        """
        np.random.seed(self.random_seed)
        
        # Type your code below here. Follow the instructions carefully!
        
        # Convert X, y to np.array
        X, y = np.array(X), np.array(y)
        
        # Create a list called fold_ids that for each row of X that indidcate what fold that row belongs to.
        # For example, if len(X) = 11 and k = 4 we want: fold_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        # because 11/4 is 2 with a remainder of 3, so we assign two samples to each of the four folds, and
        # add an additional 1 to each of the first three folds to account for the remainder. 
        # Note your code must follow the above pattern for it to pass the auto tester. If you're not sure if 
        # it is working properly, add fold_ids as a class attribute, instantiate this class with k=4 and
        # call the fit method on an X with 11 rows. Check the fold_ids attribute that you created to ensure
        # that it matches the above list. (or you can add a temporary print line, or step through using a debugger)
        
        # Create a list called fold_ids that for each row of X that indicates what fold that row belongs to.
        # Adjust for the case where len(X) is not perfectly divisible by k.
        remainder = len(X) % self.k
        fold_sizes = np.full(self.k, len(X) // self.k)
        fold_sizes[:remainder] += 1
        fold_ids = np.concatenate([np.full(size, i) for i, size in enumerate(fold_sizes)])

        # Shuffle the list fold_ids if self.shuffle is set to True 
        # (I wrote this for you, don't change this part):
        if self.shuffle:
            np.random.shuffle(fold_ids)

        
        # Create empty list called scores, make it an attribute of the class (Don't change this line):
        self.scores = []
        
        # loop through all the folds, train self.model on the other folds 
        # and append the score against the active fold to the self.scores list
        for i in range(self.k):
            train_mask = fold_ids != i
            test_mask = fold_ids == i
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            self.model.fit(X_train, y_train)
            self.scores.append(self.model.score(X_test, y_test))


        
        # Now retrain the model on the entire X and save the model to self
        self.model.fit(X, y)
        
        return self # returning self is convenient when chaining methods
    
    def predict(self, X): # Predicts using the fully trained model
        """
        This method takes a single parameter
            X: features, a two-dimensional array-like object with shape [m, d]
        
        and predicts y using a the fitted model
        """            
        X = np.array(X)
        return  self.model.predict(X)
    
    def score(self, X, y): # Outputs the R^2 of the prediction. Don't change this method.
        """
        Outputs the model's score method on X and y
        """
        X, y = np.array(X), np.array(y)
        return self.model.score(X, y)

