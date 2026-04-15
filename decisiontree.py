import numpy as np
import pickle
import os
import time
import pandas as pd
from collections import Counter
from multiprocessing import Pool, cpu_count

#The following decision tree implementation allows for splitting based on entropy or gini index

class Node:
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
        
        return Node(
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
        return Node(cellType=value, numSamples=len(y), impurity=0)
    

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
    
    #Save tree via pickle
    with open(outputPath, 'wb') as f:
        pickle.dump(tree, f)
    
    print(f"Tree saved to {outputPath}")
    
    return tree
 
 
def loadTree(inputFilePath):
    """Load a previously trained decision tree in pickle format"""
    
    with open(inputFilePath, 'rb') as f:
        tree = pickle.load(f)
    
    print(f"Tree loaded from {inputFilePath}")
    return tree


def dtAnnotateCells(tree, X, outputName='cellTypeAnnotations.csv'):
    """Predict cell types for new single cells"""
    
    #Make predictions
    predictedTypes = tree.predict(X)
    
    #Create pd DataFrame to store results after annotaiton
    results = pd.DataFrame({
        'cellID': range(len(X)),
        'predictedCellType': predictedTypes
    })
    
    #Save results to CSV
    results.to_csv(outputName, index=False)
    print(f"{len(predictedTypes)} cells annotated and saved to {outputName}")
    
    return results
 
 
if __name__ == "__main__":
    pass