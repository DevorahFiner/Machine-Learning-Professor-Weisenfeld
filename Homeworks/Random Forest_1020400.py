#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Your assignment is to complete the DecisionTree and RandomForest classes below.

You are free to add any additional class methods or optional parameters,
but do not change the arguments that have been pre added.

Once done, copy and paste the class in this cell into a textfile and save it as: 
"rf_{college_id_num}.py" (replace {college_id_num} with your college id number) and submit.
For example, if your college id is 1234567, you should name your file rf_1234567.py
    - Do NOT inlcude the curly braces: {}
    - DO include the underscore: _
    - Do NOT capitalize rf
    - DO ensure that you are saving as a .py file
    - ONLY copy the text from this cell, DO NOT copy the text from other cells
    - DO test your code to ensure it works before you submit it.
"""

import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, feature_subsample_size=None, random_seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample_size = feature_subsample_size
        self.tree = None
        self.random_seed = random_seed

    def fit(self, X, y): # Written for you!
        np.random.seed(self.random_seed)
        X, y = np.array(X), np.array(y)
        self.n_classes = len(np.unique(y))
        self.n_features_total = X.shape[1]
        self.n_features_to_use = X.shape[1] if self.feature_subsample_size is None else self.feature_subsample_size
        self.tree = self._grow_tree(X, y)
        return self

    def predict(self, X):
        # return a one dimensional numpy array with the predicted class for each row of X 
        predictions = []
        for x in X:
            node = self.tree
            while 'probs' not in node:
                if x[node['feature_index']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(max(node['probs'], key=node['probs'].get))
        return np.array(predictions)

    def predict_proba(self, X):

        # return a two dimensional numpy array with the probabilities that row i of X is class j
        
        # Step 1: Convert X to a numpy array
        X = np.array(X)
        # Step 2: create an empty list of probabilities
        probabilities = []
        # Step 3: Loop through each row of X
        #          For each row of X, step through the tree until you get to a leaf node
        #           Once you get to a leaf node, append the list of class probabilities for this row
        #           to the list you created
        for x in X:
            node = self.tree
            while 'probs' not in node:
                if x[node['feature_index']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            probabilities.append(list(node['probs'].values()))

        # Step 4: return a numpy array version of the list of probabilities

        return np.array(probabilities)
        

    
    def _grow_tree(self, X, y, depth=0):
        # This is a recursive function that I wrote for you. Note the structure of each tree node
        # So you can use that in your predict_proba method. Also note how the methods _best_criteria
        # and _split are used. You'll have to write those.
        
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return {'probs': {i: sum(y == i) / len(y) if len(y) > 0 else 0 for i in range(self.n_classes)}}

        feat_idxs = np.random.choice(self.n_features_total, self.n_features_to_use, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {'feature_index': best_feat, 'threshold': best_thresh, 'left': left, 'right': right}

    def _best_criteria(self, X, y, feat_idxs):
        # Your goal is to find the best splitting criteria given the parameters
        # feat_idxs are the column indexes of X that you are allowed to use
        
        # Step 1: Initialize best_gain, split_idx, and split_thresh variables
        #         whenever you find a better gain than your best_gain, you should
        #         update all three of these values.
        #         split_idx is the index of the column in X that you will use to split
        #         split_thresh is the value in that column of X that you will use to split
        
        best_gain, split_idx, split_thresh = 0, None, None
        
        # Step 2: Loop through each index in feat_idx
        #            Loop through each possible value in that column
        #            compute the information_gain if you were to split on that value on that column
        #            (hint call the _infomration_gain method for that). If that information_gain
        #            is greater than best_gain, update best_gain, split_idx, and split_thresh
        
        for idx in feat_idxs:
            for threshold in np.unique(X[:, idx]):
                gain = self._information_gain(y, X[:, idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = threshold
        
        # Step 3 return:

        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        
        # Compute and return the information gain by splitting X_column by split_thresh
        
        # Step 1: Compute the entropy without splitting (Hint, use self._entropy)
        
        base_entropy = self._entropy(y)
        left_idxs, right_idxs = X_column <= split_thresh, X_column > split_thresh
        
        # Step 2: Compute the entropy for each split and take the weighted average of the
        #         two entropies, weighted by the number of samples in each split

        left_weight = len(y[left_idxs]) / len(y)
        right_weight = len(y[right_idxs]) / len(y)
        new_entropy = left_weight * self._entropy(y[left_idxs]) + right_weight * self._entropy(y[right_idxs])
        
        # Step 3: return the difference between the original and new entropy

        return base_entropy - new_entropy

    def _split(self, X_column, split_thresh):
        # return a tuple of row indices (left_idxs, right_idxs), where 
        # -  left_idxs are the indices of all values in X_column <= split_thresh
        # -  right_idxs are the indices of all values in X_column > split_thresh
        return np.where(X_column <= split_thresh)[0], np.where(X_column > split_thresh)[0]
    
    def _entropy(self, y):
        # Compute and return the entropy of y
        # note there can be more than two classes of y
        counter = Counter(y)
        return -sum((count / len(y)) * np.log2(count / len(y)) for count in counter.values() if count > 0)

    
    def score(self, X, y):
        # Written for you!
        X, y = np.array(X), np.array(y)
        y_pred = self.predict(X)
        return (y == y_pred).mean()


class RandomForest:
    def __init__(self, num_trees=100, max_depth=None, min_samples_split=2, feature_subsample_size=None, random_seed=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample_size = feature_subsample_size
        self.trees = []
        self.random_seed = random_seed

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        # create a class attribute called self.trees that contains a list of fitted decision tree objects
        # each fitted on a different _bootstrap_sample from X, y (hint: use that method)
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                feature_subsample_size=self.feature_subsample_size, random_seed=self.random_seed)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self


    def predict(self, X):
        # Get and return the class with the maximum proability for each row as a single dimension numpy array
        predictions = [tree.predict(X) for tree in self.trees]
        return np.array([Counter(pred).most_common(1)[0][0] for pred in np.transpose(predictions)])


    def predict_proba(self, X):
        # Return the average probability for all classes across all trees (rows by classes)
        probabilities = [tree.predict_proba(X) for tree in self.trees]
        return np.mean(probabilities, axis=0)


    def _bootstrap_sample(self, X, y):
        # Randomly select rows from X with replacement
        # You should select the same number of rows as X has
        # Remember to include the corresponing y value for each row of X
        # Written for you!
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def score(self, X, y):
        # Written for you!
        X, y = np.array(X), np.array(y)
        y_pred = self.predict(X)
        return (y == y_pred).mean()

