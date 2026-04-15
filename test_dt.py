import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report
)
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# DT functions 
# Decision Tree from Alfred

from multiprocessing import Pool, cpu_count


# DT parameter search space
DT_SPACE = [
    Integer(2, 20, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 20, name='min_samples_leaf'),
    Categorical(['gini', 'entropy'], name='purity_measure'),
]


#The following decision tree implementation allows for splitting based on entropy or gini index

class DTNode:
    """Represents a node in the decision tree."""    
    def __init__(self, feature=None, threshold=None, left=None, right=None, cellType=None, numSamples=None, impurity=None):
        self.feature = feature          #Index of feature that is being split with
        self.threshold = threshold      #Threshold value for split
        self.left = left                #Left subtree
        self.right = right              #Right subtree
        self.cellType =  cellType       #(Leaves only): Cell type prediction  
        self.numSamples = numSamples    #Number of samples at this node
        self.impurity = impurity        #Purity measure based on the feature being split on
    
    def isLeaf(self):
        """Check if node is a leaf."""
        return self.cellType is not None #Only leaves have cellType label

class DecisionTree:
    """Decision Tree for classification and regression."""
    def __init__(self, purityMeasure=None, maxDepth = None, minSamplesSplit = None, minSamplesLeaf=None, numJobs=-1):
        self.purityMeasure = purityMeasure                      #Purity measure (gini or entropy) to compute infogain with
        self.maxDepth = maxDepth                                #Max depth tree can grow to (None = unlimited)
        self.minSamplesSplit = minSamplesSplit                  #Minimum samples required to split a node
        self.minSamplesLeaf = minSamplesLeaf                    #Minimum samples required at a leaf node
        self.numJobs = numJobs if numJobs > 0 else cpu_count()  #Number of cpus to use
        self.tree = None
        self.numFeatures = None
        self.classes = None


    def _growTree(self, X, y, depth=0):
        """Recursively grow the decision tree"""

        numSamples = X.shape[0]
        numClasses = len(np.unique(y))

        #Stopping criteria
        if (self.maxDepth is not None and depth >= self.maxDepth) or numSamples < self.minSamplesSplit or numClasses == 1:
            return self._createLeaf(y)
        
        #Try all splits
        bestSplit = self._findBestSplit(X, y)
        
        if bestSplit is None:
            return self._createLeaf(y)
        
        feature, threshold, leftIndex, rightIndex = bestSplit
        
        # Recursively build left and right subtrees
        leftSubtree = self._growTree(X[leftIndex], y[leftIndex], depth + 1)
        rightSubtree = self._growTree(X[rightIndex], y[rightIndex], depth + 1)
        
        # Calculate impurity
        impurity = self._impurity(y)
        
        return DTNode(
            feature=feature,
            threshold=threshold,
            left=leftSubtree,
            right=rightSubtree,
            numSamples=numSamples,
            impurity=impurity
        )
    
    def _findBestSplit(self, X, y):
        """Find the best feature and threshold to split a node"""
        numSamples, numFeatures = X.shape
        bestInfoGain = -np.inf
        bestSplit = None

        if self.numJobs > 1:
            results = []
            with Pool(self.numJobs) as pool:
                for feature in range(numFeatures):
                    result = pool.apply_async(self._evaluateFeature,(feature, X, y, self.minSamplesLeaf))
                    results.append(result)

                splits = [result.get() for result in results]
        
        else:
            splits = [self._evaluateFeature(feature, X, y, self.minSamplesLeaf) for feature in range(numFeatures)]

        
        for feature, threshold, leftIndex, rightIndex, infoGain in splits:
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestSplit = (feature, threshold, leftIndex, rightIndex)

        return bestSplit


    def _evaluateFeature(self, feature, X, y, minSamplesLeaf):
        """Evaluate a single feature for splitting."""

        #Get unique feature values
        uniqueVals = np.unique(X[:, feature])

        if len(uniqueVals) == 1:
            return (feature, None, None, None, -np.inf)
    
        #Thresholds to test = Midpoint for all consecutive unique feature values
        thresholds = (uniqueVals[:-1] + uniqueVals[1:])/2

        #Initialize variables to track best threshold
        bestInfoGain = -np.inf
        bestSplit = (feature, None, None, None, -np.inf)

        #Test all threhsholds
        for threshold in thresholds:
            leftIndex = X[:, feature] <= threshold
            rightIndex = ~leftIndex

            #Check if children have >minSample
            numLeft = np.sum(leftIndex)
            numRight = np.sum(rightIndex)

            if numLeft < minSamplesLeaf or numRight < minSamplesLeaf:
                continue

            #Compute infoGain for split    
            infoGain = self._infoGain(y, y[leftIndex], y[rightIndex])

            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestSplit = (feature, threshold, leftIndex, rightIndex, infoGain)

        return bestSplit

    def _impurity(self, y):
        """Compute impurity based on criterion."""
        if self.purityMeasure == "entropy":
            return entropy(y)
        elif self.purityMeasure == "gini":
            return gini(y)
        

    def _infoGain(self, node, leftChild, rightChild):
        """Compute the information gain resulting from a feature split for any purity measure"""

        nodeLength = len(node)
        leftLength = len(leftChild)
        rightLength = len(rightChild)

        #If either left or right child has no samples -> Splitting on this feature yields no info gain
        if leftLength == 0 or rightLength == 0:
            return 0.0
    
        nodeImpurity = self._impurity(node)
        
        leftImpurity = self._impurity(leftChild)
        rightImpurity = self._impurity(rightChild)

        #Weighted average of children's impurities
        childImpurity = (leftLength/nodeLength) * leftImpurity + (rightLength/nodeLength) * rightImpurity

        #Information Gain = (Node's Impurity) - (Children's Weighted Average Impurity)
        return nodeImpurity - childImpurity


    def _createLeaf(self, y):
        """Create a leaf node."""

        #Pick most common class via majority vote
        value = Counter(y).most_common(1)[0][0]
        return DTNode(cellType=value, numSamples=len(y), impurity=0)
    

    def _traverseTree(self, x, node):
        """Traverse tree to find prediction for sample x."""

        #Base case
        if node.isLeaf():
            return node.cellType
        
        #Recusively traverse tree
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)
        else:
            return self._traverseTree(x, node.right)
    

    #Methods
    def fit(self, X, y):
        """Build decision tree classifier"""
        X = np.array(X)
        y = np.array(y)
        
        self.numFeatures = X.shape[1]
        self.classes = np.unique(y)
        
        #Grow tree
        self.tree = self._growTree(X, y)
        return self

    def predict(self, X):
        """Predict class/value for all samples in data"""
        X = np.array(X)
        return np.array([self._traverseTree(x, self.tree) for x in X])

    def score(self, X, y):
        """Calculate accuracy of predicitions."""
        yPred = self.predict(X)
        return np.mean(yPred == y)


def entropy(labels):
    """Calculate conditional entropy"""
    unique, labelCounts = np.unique(labels, return_counts=True)
    labelProbs = labelCounts/len(labels)

    #Entropy = -Summation of labelProbs*log_2(labelProbs)
    return -np.sum(labelProbs * np.log2(labelProbs + 1e-10)) #Prevent underflow


def gini(labels):
    """Calculate Gini impurity"""
    unique, counts = np.unique(labels, return_counts=True)
    labelProbs = counts/len(labels)

    #Gini = 1 - Summation of labelProbs**2
    return 1.0 - np.sum(labelProbs**2)


def trainAndTestTree(XTrain, XTest, yTrain, yTest, purityMeasure='entropy', 
                        maxDepth=10, minSamplesSplit=5, minSamplesLeaf=2, numJobs=-1,
                        outputPath='trainedTree.pkl'):
    """Train a decision tree classifier and save it to disk."""
    
    print("Training Decision Tree...")
    #Create tree with paramters
    tree = DecisionTree(
        purityMeasure=purityMeasure,
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minSamplesLeaf=minSamplesLeaf,
        numJobs=numJobs
    )
    
    #Train tree
    tree.fit(XTrain, yTrain)
    
    #Evaluate tree accuracy using training and testing data
    trainAccuracy = tree.score(XTrain, yTrain)
    print(f"Training Accuracy: {trainAccuracy}")
    
    testAccuracy = tree.score(XTest, yTest)
    print(f"Testing Accuracy: {testAccuracy}")

# GP Bayesian optimization over the Decision Tree hyperparameter space.
def optimize_dt(X_train, y_train, n_calls=5, random_state=42):

    rng       = np.random.RandomState(random_state)
    n         = len(X_train)
    fold_size = n // 5
    indices   = rng.permutation(n)

    def manual_cv(max_depth, min_samples_split, min_samples_leaf, purity_measure):
        fold_accs = []
        for k in range(5):
            val_idx   = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = np.concatenate([indices[:k * fold_size],
                                        indices[(k + 1) * fold_size:]])
            Xtr = X_train[train_idx]
            ytr = y_train[train_idx]
            Xva = X_train[val_idx]
            yva = y_train[val_idx]
            clf = DecisionTree(
                purityMeasure=purity_measure,
                maxDepth=max_depth,
                minSamplesSplit=min_samples_split,
                minSamplesLeaf=min_samples_leaf,
            )
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xva)
            fold_accs.append(np.mean(preds == yva))
        return float(np.mean(fold_accs))

    @use_named_args(DT_SPACE)
    def objective(max_depth, min_samples_split, min_samples_leaf, purity_measure):
        return -manual_cv(int(max_depth), int(min_samples_split),
                          int(min_samples_leaf), purity_measure)

    print(f"Optimizing Decision Tree with GP ({n_calls} calls)")
    result = gp_minimize(
        objective, DT_SPACE,
        n_calls=n_calls,
        random_state=random_state,
        n_initial_points=5,
        acq_func='EI',
        verbose=False,
    )
    best_params = {
        'max_depth':         int(result.x[0]),
        'min_samples_split': int(result.x[1]),
        'min_samples_leaf':  int(result.x[2]),
        'purity_measure':    result.x[3],
    }
    print(f"Best CV accuracy: {-result.fun:.4f}")
    print(f"Best params: {best_params}")
    return best_params, result

def load_data(pca_filepath, clusters_filepath, test_size=0.3, random_state=42):
    # Loads PCA and cluster files, merges by cell barcode, and returns 70/30 train/test splits.

    pca_df = pd.read_csv(pca_filepath,index_col=0)
    clusters_df = pd.read_csv(clusters_filepath,index_col=0)

    merged = pca_df.join(clusters_df, how='inner')
    if merged.empty:
        raise ValueError("No shared barcodes between files")

    label_col = clusters_df.columns[0]
    X = merged.drop(columns=[label_col]).values.astype(np.float32)
    y = merged[label_col].values.astype(int)

    # drop rows with any null features
    valid_mask = ~np.isnan(X).any(axis=1)
    X, y = X[valid_mask], y[valid_mask]

    print(f"Cells loaded: {X.shape[0]}")
    print(f"PCA features: {X.shape[1]}")
    print(f"Clusters: {sorted(np.unique(y).tolist())}")

    label_names = sorted(np.unique(y).tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    print(f"Train count: {X_train.shape[0]}")
    print(f"Test count: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, label_names

if __name__ == "__main__":

    # create outputs 
    # File paths 
    pca_file = "pca_coordinates.csv" 
    cluster_file = "cell_clusters.csv" 
    test_set = 0.30   
    seed = 42

    # Number of GM optimization runs 
    n_calls_SVM = 30
    n_calls_RF = 30
    n_calls_DT = 20

    # Creates output directory
    outputs = "results"
    os.makedirs(outputs, exist_ok=True)
    print(f"Outputs will be saved to: {os.path.abspath(outputs)}/")

    X_train, X_test, y_train, y_test, LABEL_NAMES = load_data(
    pca_file, cluster_file,
    test_size=test_set,
    random_state=seed,
    )

    dt_best_params, dt_gp_result = optimize_dt(
    X_train, y_train,
    n_calls=n_calls_DT,
    random_state=seed,
    )
